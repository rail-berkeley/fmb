import functools as ft
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from orca.model.clip import CLIPTextTokenizer, CLIPVisionTokenizer, clip_weights_loader
from orca.model.transformer import MlpBlock
from orca.model.vision import encoders
from orca.model.proprio import mlps
from einops import rearrange

EPS = 1e-6


# adapted from https://github.com/google-research/robotics_transformer/blob/master/tokenizers/token_learner.py
class TokenLearner(nn.Module):
    num_tokens: int
    bottleneck_dim: int = 64
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        if len(inputs.shape) == 4:
            inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1])
        x = nn.LayerNorm()(inputs)
        x = MlpBlock(
            mlp_dim=self.bottleneck_dim,
            out_dim=self.num_tokens,
            dropout_rate=self.dropout_rate,
        )(x, deterministic=not train)
        x = jnp.transpose(x, (0, 2, 1))
        x = nn.softmax(x, axis=-1)
        return jnp.einsum("bna,baf->bnf", x, inputs)


# adapted from https://github.com/google-research/robotics_transformer/blob/master/tokenizers/image_tokenizer.py
class ImageTokenizer(nn.Module):
    encoder: str
    encoder_kwargs: dict = None
    use_token_learner: bool = False
    num_tokens: int = 8  # this is not enforced unless use_token_learner is True
    conditioning_type: str = "none"

    @nn.compact
    def __call__(
        self,
        observations,
        goals=None,
        train: bool = True,
    ):
        # observations["image"] is (batch, obs_horizon, height, width, channel)
        b, t, h, w, c = observations["image"].shape
        if self.conditioning_type == "none":
            # late-fusion architecture, image encoder doesn't see task and obs together
            image = observations["image"]
            image = jnp.reshape(image, (b * t, h, w, c))
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "goal_image":
            # early-fusion goal-image only architecture, concatenate obs and goal image channel-wise
            image = jnp.concatenate(
                [observations["image"][:, -1], goals["image"]], axis=-1
            )
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "goal_image_no_obs":
            image = goals["image"]
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "film_language":
            # encode task and pass into encoder with FiLM
            image = observations["image"]
            image = jnp.reshape(image, (b * t, h, w, c))
            lang = goals["language"]
            lang = lang[:, None, :].repeat(t, axis=1)
            lang = jnp.reshape(lang, (b * t, -1))
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(
                image, cond_var=lang
            )
            image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))

        if self.use_token_learner:
            image_tokens = jnp.reshape(
                image_tokens, (b * t, -1, image_tokens.shape[-1])
            )
            image_tokens = TokenLearner(num_tokens=self.num_tokens)(
                image_tokens, train=train
            )
            image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))
        return image_tokens

class UnmaskedMultiImageTokenizer(nn.Module):
    encoder: str
    encoder_kwargs: dict = None
    use_token_learner: bool = False
    projected_num_tokens: int = 8  # this is not enforced unless use_token_learner is True
    conditioning_type: str = "none"
    img_obs_keys: str = "image_wrist_1:image_wrist_2:image_side_1:image_side_2"
    num_primitive: int = 5
    num_peg: int = 11
    mask_peg_id: bool = True

    def setup(self):
        img_view_count = len(self.img_obs_keys.split(":"))
        if self.use_token_learner:
            self.num_tokens = self.projected_num_tokens * img_view_count
        else:
            if self.encoder == 'resnetv1-34-bridge':
                self.num_tokens = 64 * img_view_count
            elif self.encoder == 'resnetv1-34-bridge-avg-pool':
                self.num_tokens = 1 * img_view_count
        self.image_encoder = encoders[self.encoder](**self.encoder_kwargs)

    def index_to_onehot(self, index, total_num):
        one_hot = jax.nn.one_hot(index, total_num)
        return one_hot
    @nn.compact
    def __call__(
        self,
        observations,
        train: bool = True,
    ):
        # observations["image_xxx_x"] is (batch, height, width, channel)
        parsed_img_keys = self.img_obs_keys.split(":")
        b, h, w, c = observations[parsed_img_keys[-1]].shape
        all_image_tokens = []
        if self.conditioning_type == "none":
            for img_key in parsed_img_keys:
                image = observations[img_key]
                image_tokens = self.image_encoder(image)
                image_tokens = jnp.reshape(image_tokens, (b, 1, -1, image_tokens.shape[-1]))
                all_image_tokens.append(image_tokens)
            all_image_tokens = jnp.concatenate(all_image_tokens, axis=1)
        elif self.conditioning_type == "film":
            # encode task and pass into encoder with FiLM
            peg_onehot = self.index_to_onehot(observations['peg_id'], self.num_peg)
            cond = nn.Dense(features=512)(peg_onehot) # b, 512
            for img_key in parsed_img_keys:
                image = observations[img_key]
                image_tokens = self.image_encoder(image, cond_var=cond)
                image_tokens = jnp.reshape(image_tokens, (b, 1, -1, image_tokens.shape[-1]))
                all_image_tokens.append(image_tokens)
            all_image_tokens = jnp.concatenate(all_image_tokens, axis=1)
        else:
            raise NotImplementedError

        b, t, n, d = all_image_tokens.shape

        if self.use_token_learner:
            all_image_tokens = jnp.reshape(
                all_image_tokens, (b * t, -1, d) # b*4, 64, 512
            )
            all_image_tokens = TokenLearner(num_tokens=self.projected_num_tokens)(
                all_image_tokens, train=train
            )

        all_image_tokens = jnp.reshape(all_image_tokens, (b, -1, d))
        return all_image_tokens

class MultiImageTokenizer(nn.Module):
    encoder: str
    encoder_kwargs: dict = None
    use_token_learner: bool = False
    projected_num_tokens: int = 8  # this is not enforced unless use_token_learner is True
    conditioning_type: str = "none"
    img_obs_keys: str = "image_wrist_1:image_wrist_2:image_side_1:image_side_2"
    num_primitive: int = 5
    num_peg: int = 11
    mask_peg_id: bool = True

    def setup(self):
        img_view_count = len(self.img_obs_keys.split(":"))
        if self.use_token_learner:
            self.num_tokens = self.projected_num_tokens * (img_view_count - 1)
        else:
            if self.encoder == 'resnetv1-34-bridge':
                self.num_tokens = 64 * (img_view_count - 1)
            elif self.encoder == 'resnetv1-34-bridge-avg-pool':
                self.num_tokens = 1 * (img_view_count - 1)
        self.image_encoder = encoders[self.encoder](**self.encoder_kwargs)

    def index_to_onehot(self, index, total_num):
        one_hot = jax.nn.one_hot(index, total_num)
        return one_hot
    @nn.compact
    def __call__(
        self,
        observations,
        train: bool = True,
    ):
        # observations["image_xxx_x"] is (batch, height, width, channel)
        parsed_img_keys = self.img_obs_keys.split(":")
        b, h, w, c = observations[parsed_img_keys[-1]].shape
        all_image_tokens = []
        insert_mask = observations['primitive_id'] == 3
        if self.conditioning_type == "none":
            for img_key in parsed_img_keys:
                image = observations[img_key]
                image_tokens = self.image_encoder(image)
                image_tokens = jnp.reshape(image_tokens, (b, 1, -1, image_tokens.shape[-1]))
                all_image_tokens.append(image_tokens)
            all_image_tokens = jnp.concatenate(all_image_tokens, axis=1)
        elif self.conditioning_type == "film":
            # encode task and pass into encoder with FiLM
            peg_onehot = self.index_to_onehot(observations['peg_id'], self.num_peg)
            cond = nn.Dense(features=512)(peg_onehot) # b, 512
            if self.mask_peg_id:
                cond = jnp.where(
                    insert_mask[..., None],
                    cond,
                    jnp.zeros_like(cond)
                )
            for img_key in parsed_img_keys:
                image = observations[img_key]
                image_tokens = self.image_encoder(image, cond_var=cond)
                image_tokens = jnp.reshape(image_tokens, (b, 1, -1, image_tokens.shape[-1]))
                all_image_tokens.append(image_tokens)
            all_image_tokens = jnp.concatenate(all_image_tokens, axis=1)
        else:
            raise NotImplementedError

        b, t, n, d = all_image_tokens.shape # b, v, n, d
        insert_mask = jnp.broadcast_to(insert_mask[..., None, None, None], (b, 3, n, d))
        # if insert, taking 0, 1, 3
        # if not insert, taking 0, 1, 2
        all_image_tokens = jnp.where(
            insert_mask,
            all_image_tokens[:, [0, 1, 2], :, :], # b, 3, n, d
            all_image_tokens[:, [0, 1, 3], :, :] # b, 3, n, d
        )
        b, t, n, d = all_image_tokens.shape

        if self.use_token_learner:
            all_image_tokens = jnp.reshape(
                all_image_tokens, (b * t, -1, d) # b*3, 64, 512
            )
            all_image_tokens = TokenLearner(num_tokens=self.projected_num_tokens)(
                all_image_tokens, train=train
            )

        all_image_tokens = jnp.reshape(all_image_tokens, (b, -1, d))
        return all_image_tokens

class LanguageTokenizer(nn.Module):
    encoder: str = None
    encoder_kwargs: dict = None
    num_tokens: int = 1

    @nn.compact
    def __call__(
        self,
        observations,
        goals=None,
        train: bool = True,
    ):

        # add a time dimension to language
        if goals["language"].ndim == 2:
            tokens = goals["language"][:, None, :]
        else:
            tokens = goals["language"]

        return tokens

class StateTokenizer(nn.Module):
    mlp: str = None
    mlp_kwargs: dict = None
    num_tokens: int = 1

    @nn.compact
    def __call__(
        self,
        observations,
    ):  
        # this will encode state into one token
        b, t, d = observations['proprio'].shape
        
        state = observations['proprio']
        state = jnp.reshape(state, (b * t, d))
        state_tokens = mlps[self.mlp](**self.mlp_kwargs)(state)
        state_tokens = jnp.reshape(state_tokens, (b, -1, state_tokens.shape[-1]))

        return state_tokens
class UnMaskedStateTokenizer(nn.Module):
    mlp: str = None
    mlp_kwargs: dict = None
    num_token_per_state: int = 2

    def setup(self):
        # self.ee_vel_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.ee_ft_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.ee_pos_xy_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.ee_pos_z_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.ee_quat_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.num_tokens = self.num_token_per_state * 4

    @nn.compact
    def __call__(
        self,
        observations,
        train: bool = True,
    ):
        ee_vel, ee_pose, ee_ft = observations['ee_vel'], observations['ee_pose'], observations['ee_ft']
        ee_pos_xy, ee_pos_z, ee_quat = ee_pose[..., :2], ee_pose[..., 2:3], ee_pose[..., 3:]
        # ee_vel_token = self.ee_vel_tokenizer(ee_vel)[:, None, ...]
        ee_ft_token = self.ee_ft_tokenizer(ee_ft)[:, None, ...]
        ee_pos_xy_token = self.ee_pos_xy_tokenizer(ee_pos_xy)[:, None, ...]
        ee_pos_z_token = self.ee_pos_z_tokenizer(ee_pos_z)[:, None, ...]
        ee_quat_token = self.ee_quat_tokenizer(ee_quat)[:, None, ...]

        state_tokens = jnp.concatenate([ee_ft_token, ee_pos_xy_token, ee_pos_z_token, ee_quat_token], axis=1)
        state_tokens = rearrange(state_tokens, 'b t (n d) -> b (t n) d', n=self.num_token_per_state)
        return state_tokens
class MaskedStateTokenizer(nn.Module):
    mlp: str = None
    mlp_kwargs: dict = None
    num_token_per_state: int = 2

    def setup(self):
        # self.ee_vel_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.ee_ft_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.ee_pos_xy_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.ee_pos_z_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.ee_quat_tokenizer = mlps[self.mlp](**self.mlp_kwargs)
        self.num_tokens = self.num_token_per_state * 4

    @nn.compact
    def __call__(
        self,
        observations,
        train: bool = True,
    ):
        ee_vel, ee_pose, ee_ft = observations['ee_vel'], observations['ee_pose'], observations['ee_ft']
        ee_pos_xy, ee_pos_z, ee_quat = ee_pose[..., :2], ee_pose[..., 2:3], ee_pose[..., 3:]
        # ee_vel_token = self.ee_vel_tokenizer(ee_vel)[:, None, ...]
        ee_ft_token = self.ee_ft_tokenizer(ee_ft)[:, None, ...]
        ee_pos_xy_token = self.ee_pos_xy_tokenizer(ee_pos_xy)[:, None, ...]
        ee_pos_z_token = self.ee_pos_z_tokenizer(ee_pos_z)[:, None, ...]
        ee_quat_token = self.ee_quat_tokenizer(ee_quat)[:, None, ...]

        insert_mask = observations['primitive_id'] == 3
        grasp_mask = observations['primitive_id'] == 0
        insert_mask = jnp.broadcast_to(insert_mask[..., None, None], ee_ft_token.shape)
        grasp_mask = jnp.broadcast_to(grasp_mask[..., None, None], ee_ft_token.shape)


        ee_ft_token = jnp.where( # only insertion needs ee ft
            insert_mask,
            ee_ft_token,
            jnp.zeros_like(ee_ft_token)
        )

        # ee_vel_token = jnp.where(
        #     insert_mask, # only insertion needs ee vel
        #     ee_vel_token,
        #     jnp.zeros_like(ee_vel_token)
        # )

        ee_quat_token = jnp.where(
            grasp_mask,   # only grasp doesn't need quat
            jnp.zeros_like(ee_quat_token),
            ee_quat_token
        )

        ee_pos_xy_token = jnp.where(
            jnp.logical_or(insert_mask, grasp_mask), # insertion and grasp don't need xy
            jnp.zeros_like(ee_pos_xy_token),
            ee_pos_xy_token
        )

        # everything needs z
        state_tokens = jnp.concatenate([ee_ft_token, ee_pos_xy_token, ee_pos_z_token, ee_quat_token], axis=1)
        state_tokens = rearrange(state_tokens, 'b t (n d) -> b (t n) d', n=self.num_token_per_state)
        return state_tokens

class UnifiedObservationTokenizer(nn.Module):
    image_tokenizer_kwargs: dict = None
    state_tokenizer_kwargs: dict = None

    def setup(self):
        self.image_tokenizer = MultiImageTokenizer(
            **self.image_tokenizer_kwargs
        )
        self.state_tokenizer = MaskedStateTokenizer(
            **self.state_tokenizer_kwargs
        )
        self.num_tokens = self.image_tokenizer.num_tokens + self.state_tokenizer.num_tokens


    @nn.compact
    def __call__(
        self,
        observations,
        train: bool = True,
    ):
        image_tokens = self.image_tokenizer(observations, train=train)
        state_tokens = self.state_tokenizer(observations, train=train)
        tokens = jnp.concatenate([image_tokens, state_tokens], axis=1)
        return tokens

class UnifiedUnMaskedObservationTokenizer(nn.Module):
    image_tokenizer_kwargs: dict = None
    state_tokenizer_kwargs: dict = None

    def setup(self):
        self.image_tokenizer = UnmaskedMultiImageTokenizer(
            **self.image_tokenizer_kwargs
        )
        self.state_tokenizer = UnMaskedStateTokenizer(
            **self.state_tokenizer_kwargs
        )
        self.num_tokens = self.image_tokenizer.num_tokens + self.state_tokenizer.num_tokens


    @nn.compact
    def __call__(
        self,
        observations,
        train: bool = True,
    ):
        image_tokens = self.image_tokenizer(observations, train=train)
        state_tokens = self.state_tokenizer(observations, train=train)
        tokens = jnp.concatenate([image_tokens, state_tokens], axis=1)
        return tokens


class UnifiedTaskOneHotTokenizer(nn.Module):
    num_primitive: int = 5 # number of primitives / pegs
    num_peg: int = 5 # number of pegs
    embed_dim: int = 512
    mlp: str = None
    mlp_kwargs: dict = None
    mask_peg_id: bool = True
    num_primitive_tokens: int = 1
    num_peg_tokens: int = 1

    def setup(self):
        self.task_token_projection_primitive = mlps[self.mlp](**self.mlp_kwargs)
        self.task_token_projection_peg = mlps[self.mlp](**self.mlp_kwargs)
        self.num_tokens = self.num_primitive_tokens + self.num_peg_tokens


    def index_to_onehot(self, index, total_num):
        one_hot = jax.nn.one_hot(index, total_num)
        return one_hot
    @nn.compact
    def __call__(self, observations, train: bool = True):
        primitive_id = observations['primitive_id']
        primitive_one_hot = self.index_to_onehot(primitive_id, self.num_primitive)
        primitive_tokens = self.task_token_projection_primitive(primitive_one_hot)[:, None, ...]
        primitive_tokens = rearrange(primitive_tokens, 'b t (n d) -> b (t n) d', n=self.num_primitive_tokens)

        peg_id = observations['peg_id']
        peg_one_hot = self.index_to_onehot(peg_id, self.num_peg)
        peg_tokens = self.task_token_projection_peg(peg_one_hot)[:, None, ...]
        peg_tokens = rearrange(peg_tokens, 'b t (n d) -> b (t n) d', n=self.num_peg_tokens)

        if self.mask_peg_id:
            insert_mask = observations['primitive_id'] == 3
            insert_mask = jnp.broadcast_to(insert_mask[..., None, None], peg_tokens.shape)
            peg_tokens = jnp.where(
                insert_mask,
                peg_tokens,
                jnp.zeros_like(peg_tokens)
            ) # only insertion needs peg id
        task_tokens = jnp.concatenate([primitive_tokens, peg_tokens], axis=1)
        return task_tokens

class DummyTaskTokenizer(nn.Module):
    def setup(self):
        self.num_tokens = 1
        self.embed_dim = 512
    @nn.compact
    def __call__(self, observations, train: bool = True):
        task_token = jnp.zeros((observations['primitive_id'].shape[0], 1, self.embed_dim))
        return task_token

class ActionTokenizer(nn.Module):
    action_dim: int
    vocab_size: int
    normalization_type: str = "bounds"
    low: float = 0
    high: float = 1

    def setup(self):
        if self.normalization_type == "bounds":
            self.thresholds = jnp.linspace(self.low, self.high, self.vocab_size + 1)
        elif self.normalization_type == "normal":
            self.thresholds = norm.ppf(jnp.linspace(EPS, 1 - EPS, self.vocab_size + 1))
        else:
            raise ValueError

    def __call__(self, actions, mode: str = "tokenize"):
        if mode == "tokenize":
            if self.normalization_type == "bounds":
                actions = jnp.clip(actions, self.low + EPS, self.high - EPS)
            actions = actions[..., None]
            token_one_hot = (actions < self.thresholds[1:]) & (
                actions >= self.thresholds[:-1]
            ).astype(jnp.uint8)
            action_tokens = jnp.argmax(token_one_hot, axis=-1)
            return action_tokens
        elif mode == "detokenize":
            action_tokens = actions
            one_hot = jax.nn.one_hot(action_tokens, self.vocab_size)
            bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
            actions = jnp.sum(one_hot * bin_avgs, axis=-1)
            return actions


tokenizers = {
    "obs-tokenizer": ft.partial(
        ImageTokenizer,
        conditioning_type="none",
    ),
    "goal-tokenizer": ft.partial(
        ImageTokenizer,
        conditioning_type="goal_image_no_obs",
    ),
    "goal-obs-tokenizer": ft.partial(
        ImageTokenizer,
        conditioning_type="goal_image",
    ),
    "obs-film-language-tokenizer": ft.partial(
        ImageTokenizer,
        conditioning_type="film_language",
    ),
    "language-tokenizer": LanguageTokenizer,
    "masked-state-tokenizer": MaskedStateTokenizer,

    "fmb-unified-obs-tokenizer": UnifiedObservationTokenizer,
    "fmb-unified-unmasked-obs-tokenizer": UnifiedUnMaskedObservationTokenizer,
    "fmb-unified-task-tokenizer": UnifiedTaskOneHotTokenizer,
    "dummy-task-tokenizer": DummyTaskTokenizer,
}

weights_loaders = {
    "clip": clip_weights_loader,
}

if __name__ == "__main__":
    import jax
    import numpy as np

    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    action = np.broadcast_to(action, [2, 2, 7])
    tokenizer = ActionTokenizer(
        action_dim=7, vocab_size=256, normalization_type="normal"
    )
    params = tokenizer.init(jax.random.PRNGKey(0), action)
    action_tokens = tokenizer.apply(params, action)
    detokenized_actions = tokenizer.apply(params, action_tokens, mode="detokenize")

    print(action)
    print(action_tokens)
    print(detokenized_actions)
