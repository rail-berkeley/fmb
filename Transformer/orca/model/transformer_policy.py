# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.tokenizers import ActionTokenizer
from orca.model.transformer import Transformer
from orca.typing import PRNGKey, Sequence


class TransformerPolicy(nn.Module):
    """
    Transformer that models trajectories.
    """

    observation_tokenizers: Sequence[nn.Module]
    task_tokenizers: Sequence[nn.Module]
    vocab_size: int = 256
    token_embedding_size: int = 512
    num_layers: int = 4
    mlp_dim: int = 1024
    num_heads: int = 8
    dropout_rate: float = 0.1
    time_sequence_length: int = 1
    action_dim: int = 7
    normalization_type: str = "bounds"
    num_views: int = 2

    def setup(self):
        self.transformer = Transformer(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )

        self.action_tokenizer = ActionTokenizer(
            action_dim=self.action_dim,
            vocab_size=self.vocab_size,
            normalization_type=self.normalization_type,
        )

        self.tokens_per_action = self.action_dim
        self.tokens_per_obs = sum(tok.num_tokens for tok in self.observation_tokenizers)
        self.tokens_per_task = sum(tok.num_tokens for tok in self.task_tokenizers)
        self.attention_mask = self.generate_masks()
        self.vocab_proj = nn.Dense(self.vocab_size)

    def __call__(
        self,
        observations,
        goals,
        actions,
        train: bool = False,
    ):
        output = self.transformer_call(
            observations,
            goals,
            self.attention_mask,
            train=train,
        )
        
        # get the output for the last action
        # TODO (homer): supervise all action predictions
        action_logits = output[:, -1, -self.tokens_per_action :]
        action_logprob = jax.nn.log_softmax(action_logits, axis=-1)

        action_labels = self.action_tokenizer(actions, mode="tokenize")
        action_labels_one_hot = jax.nn.one_hot(action_labels, self.vocab_size)

        action_loss = -jnp.sum(action_logprob * action_labels_one_hot, axis=-1).mean()

        action_pred = jnp.argmax(action_logits, axis=-1)
        accuracy = (action_pred == action_labels).mean()

        action_values = self.action_tokenizer(action_pred, mode="detokenize")
        action_mse = jnp.square(actions - action_values).sum(axis=-1).mean()

        return {"loss": action_loss, "mse": action_mse, "accuracy": accuracy}

    def transformer_call(
        self,
        observations,
        goals,
        attention_mask,
        train: bool = False,
        obs_only_input: bool = True,
    ):
        task_tokens, obs_tokens, action_tokens = self.get_tokens(
            observations, goals
        )
        
        if obs_only_input:
            input_tokens = obs_tokens
        else:
            input_tokens = self.assemble_input_tokens(
                task_tokens, obs_tokens, action_tokens
            )
        

        input_tokens = jnp.reshape(obs_tokens, (obs_tokens.shape[0], -1, obs_tokens.shape[-1]))

        output = self.transformer(
            input_tokens, attention_mask=attention_mask, train=train
        )
        
        output = self.vocab_proj(output)
        # remove output corresponding to task
        output = output[:, self.tokens_per_task :]
        # unfold time sequence length from token sequence length
        return jnp.reshape(
            output,
            (
                output.shape[0],
                -1,
                self.tokens_per_time_step,
                self.vocab_size,
            ),
        )

    def predict_action(
        self,
        observations,
        goals,
        train: bool = False,
        argmax: bool = False,
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ):
        output = self.transformer_call(
            observations,
            goals,
            self.attention_mask,
            train=train,
        )

        # use the last action as the predicted action
        action_logits = output[:, -1, -self.tokens_per_action :]

        if argmax:
            action_tokens = jnp.argmax(action_logits, axis=-1).astype(jnp.int32)
        else:
            action_tokens = jax.random.categorical(
                rng, action_logits / temperature, axis=-1
            ).astype(jnp.int32)
        return self.action_tokenizer(action_tokens, mode="detokenize")

    def generate_masks(self):
        """
        Generate attention mask for transformer call.
        """
        # we don't attend to actions, so its [obs]
        self.tokens_per_time_step = self.tokens_per_obs 
        # self.total_tokens = self.tokens_per_time_step * self.time_sequence_length * self.num_views

        #TODO(fangchen) add mask according to different input modalities
        causal_mask = jnp.ones((self.tokens_per_time_step, self.tokens_per_time_step))
        
        return causal_mask

    def get_tokens(self, observations, goals):
        """
        Tokenize obervation/action history and task (either goal image or language).
        """

        obs_tokens = [tok(observations) for tok in self.observation_tokenizers]
        obs_tokens = [
            jnp.reshape(t, (t.shape[0], -1, t.shape[-1])) if len(t.shape) > 3 else t 
            for t in obs_tokens
        ]
        obs_tokens = jnp.concatenate(obs_tokens, axis=-2)
        assert obs_tokens.shape[-2] == self.tokens_per_obs

        if len(self.task_tokenizers) > 0:
            # a list of (batch, tokens_per_X, token_embedding_size)
            task_tokens = [
                tok(observations, goals, train=train) for tok in self.task_tokenizers
            ]
            # (batch, tokens_per_task, token_embedding_size)
            task_tokens = jnp.concatenate(task_tokens, axis=-2)
            assert task_tokens.shape[-2] == self.tokens_per_task
        else:
            task_tokens = jnp.zeros(
                (obs_tokens.shape[0], self.tokens_per_task, self.token_embedding_size)
            )

        # (batch, time_seq_len, tokens_per_action, token_embedding_size)
        # TODO we don't currently attend to actions so just use zeros here
        action_tokens = jnp.zeros(
            (
                *observations["image_wrist_1"].shape[:2],
                self.tokens_per_action,
                self.token_embedding_size,
            )
        )
        return task_tokens, obs_tokens, action_tokens

    def assemble_input_tokens(self, task_tokens, obs_tokens, action_tokens):
        """
        Folds time sequence dim into token sequence dim. Then prepends task tokens.
        """

        # (batch, seq_length, tokens_per_time_step, token_embedding_size)

        tokens = jnp.concatenate([obs_tokens, action_tokens], axis=2)
        # (batch, seq_length * tokens_per_time_step, token_embedding_size)
        tokens = jnp.reshape(tokens, (tokens.shape[0], -1, tokens.shape[-1]))
        tokens = jnp.concatenate([task_tokens, tokens], axis=1)
        return tokens

    def get_action_index_for_token(self, k):
        """
        Returns the action index associated with the token at given position `k`.

        If k is not an action token then it returns None.
        If k is part of the first action in the sequence then returns 0 etc.
        """
        if k < 0 or k >= self.total_tokens:
            return None

        if k % self.tokens_per_time_step < self.tokens_per_obs:
            return None
        else:
            return int(k / self.tokens_per_time_step)
