from functools import partial
from typing import Any, Callable, Sequence, Tuple
from ml_collections import ConfigDict

import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax

from .jax_utils import JaxRNG
from flax.core.frozen_dict import FrozenDict

class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param('value', lambda x:self.init_value)

    def __call__(self):
        return self.value


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = '256-256'

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split('-')]
        for h in hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim)(x)


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: Any
    norm: Any
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: Any
    norm: Any
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: Any
    num_filters: int = 64
    pooling: Callable = nn.max_pool
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: Any = nn.Conv

    @nn.compact
    def __call__(self, x):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.GroupNorm,
            num_groups=32,
            dtype=self.dtype,
        )

        x = conv(
            self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init"
        )(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        x = self.pooling(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)
        # x = jnp.mean(x, axis=(1, 2))
        # x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x



ResNetModules = {
    'ResNet18': partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock),
    'ResNet34': partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock),
    'ResNet50': partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock),
    'ResNet101': partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock),
    'ResNet152': partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock),
    'ResNet200': partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock),
    'ResNet18Depth': partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, pooling=nn.avg_pool),
    'ResNet34Depth': partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock, pooling=nn.avg_pool),
    'ResNet50Depth': partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock, pooling=nn.avg_pool),
    'ResNet101Depth': partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock, pooling=nn.avg_pool),
    'ResNet152Depth': partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock, pooling=nn.avg_pool),
    'ResNet200Depth': partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock, pooling=nn.avg_pool),
}


class ResNetPolicy(nn.Module):
    output_dim: int
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.resnet_type = ('ResNet18', )
        config.spatial_aggregate = 'average'
        config.mlp_arch = '256-256'
        config.state_injection = 'full'
        config.state_projection_dim = 64
        config.share_resnet_between_views = True
        config.shape_projection_dim = 64


        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    @nn.nowrap
    def rng_keys():
        return ('params',)

    @staticmethod
    @nn.nowrap
    def get_weight_decay_exclusions():
        return ('bias')

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

    @nn.compact
    def __call__(self, state, images, shape_vec=None, primitive_vec=None):
        features = []
        for i in range(len(images)):
            x = images[i]
            if self.config.share_resnet_between_views and images[i].shape[-1] == 1:
                z = ResNetModules[self.config.resnet_type[-1]]()(x)
            elif self.config.share_resnet_between_views and images[i].shape[-1] == 3:
                z = ResNetModules[self.config.resnet_type[0]]()(x)
            else:
                z = ResNetModules[self.config.resnet_type[i]]()(x)

            if self.config.spatial_aggregate == 'average':
                z = jnp.mean(z, axis=(1, 2))
            elif self.config.spatial_aggregate == 'flatten':
                z = z.reshape(z.shape[0], -1)
            else:
                raise ValueError('Unsupported spatial aggregation type!')
            features.append(z)

        if self.config.state_injection == 'full':
            features.append(nn.Dense(self.config.state_projection_dim)(state))
        elif self.config.state_injection == 'z_only':
            features.append(nn.Dense(self.config.state_projection_dim)(jnp.concatenate((state[:, 2:3], state[:, 7:]), axis=1)))
        elif self.config.state_injection == 'no_xy':
            features.append(nn.Dense(self.config.state_projection_dim)(state[:, 2:]))
        elif self.config.state_injection == 'none':
            pass
        else:
            raise ValueError(f'Unsupported state_injection: {self.config.state_injection}!')
        if shape_vec is not None:
            features.append(nn.Dense(self.config.shape_projection_dim)(shape_vec))
        if primitive_vec is not None:
            features.append(nn.Dense(self.config.shape_projection_dim)(primitive_vec))
        x = jnp.concatenate(features, axis=1)
        return FullyConnectedNetwork(self.output_dim, self.config.mlp_arch)(x)


class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, features):
        """ 
        features is B x H x W X C
        """
        kernel = self.param('kernel',
                            nn.initializers.lecun_normal(),
                            (self.height, self.width, self.channel, self.num_features),
                            self.param_dtype)

        batch_size = features.shape[0]
        # assert len(features.shape) == 4
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2))
        features = jnp.reshape(features, [batch_size, -1])
        return features

class MobileNetEncoder(nn.Module):
    output_dim: int
    mobilenet: Callable[..., Callable]
    params: FrozenDict
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = ConfigDict()
        config.mlp_arch = '256-256'
        config.state_injection = 'full'
        config.state_projection_dim = 64
        config.use_shape_vec = False
        config.shape_projection_dim = 64
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    @nn.nowrap
    def rng_keys():
        return ('params',)

    @staticmethod
    @nn.nowrap
    def get_weight_decay_exclusions():
        return ('bias')

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

    @nn.compact
    def __call__(self, state, images, shape_vec=None, training=False) -> jnp.ndarray:
        features = []
        for i in range(len(images)):
            x = self.mobilenet.apply(self.params, images[i], mutable=False, training=False)
            # x = jax.lax.stop_gradient(x)
            x = SpatialLearnedEmbeddings(*(x.shape[1:]), 16)(x)
            x = nn.Dropout(0.5)(x, deterministic=not training)
            features.append(x)

        if self.config.state_injection == 'full':
            features.append(nn.Dense(self.config.state_projection_dim)(state))
        elif self.config.state_injection == 'z_only':
            features.append(nn.Dense(self.config.state_projection_dim)(jnp.concatenate((state[:, 2:3], state[:, 7:]), axis=1)))
        elif self.config.state_injection == 'none':
            pass
        else:
            raise ValueError(f'Unsupported state_injection: {self.config.state_injection}!')
        
        if self.config.use_shape_vec and shape_vec is not None:
            features.append(nn.Dense(self.config.shape_projection_dim)(shape_vec))
        elif self.config.use_shape_vec and shape_vec is None:
            raise ValueError('use_shape_vec is True but shape_vec is None!')
        x = jnp.concatenate(features, axis=1)
        return FullyConnectedNetwork(self.output_dim, self.config.mlp_arch)(x)
    

class TanhGaussianResNetPolicy(nn.Module):
    output_dim: int
    config_updates: ... = None
    model: str = 'ResNet'

    @staticmethod
    @nn.nowrap
    def get_default_config(model='ResNet', updates=None):
        if model=='ResNet':
            return ResNetPolicy.get_default_config(updates)
        elif model=='MobileNet':
            return MobileNetEncoder.get_default_config(updates)

    @staticmethod
    @nn.nowrap
    def rng_keys():
        return ('params', 'noise')

    @staticmethod
    @nn.nowrap
    def get_weight_decay_exclusions():
        return ('bias')

    def setup(self):
        self.config = self.get_default_config(self.model, self.config_updates)
        if self.model=='ResNet':
            self.backbone = ResNetPolicy(self.output_dim * 2, self.config)
        elif self.model=='MobileNet':
            from jeffnet.linen import create_model, EfficientNet
            MobileNet, mobilenet_variables = create_model('tf_efficientnet_b0', pretrained=True)
            self.backbone = MobileNetEncoder(output_dim=self.output_dim * 2, mobilenet=MobileNet, params=mobilenet_variables)

    def log_prob(self, state, action, images, shape_vec=None, primitive_vec=None, return_mean=False):
        mean, log_std = jnp.split(self.backbone(state, images, shape_vec=shape_vec, primitive_vec=primitive_vec), 2, axis=-1)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        log_probs = action_distribution.log_prob(action)
        if return_mean:
            return log_probs, mean
        return log_probs

    def __call__(self, state, images, shape_vec=None, primitive_vec=None, deterministic=False):
        mean, log_std = jnp.split(self.backbone(state, images, shape_vec=shape_vec, primitive_vec=primitive_vec), 2, axis=-1)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(
                seed=self.make_rng('noise')
            )

        return samples, log_prob

class TanhGaussianResNetMixedPolicy(nn.Module):
    output_dim: int
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        return ResNetPolicy.get_default_config(updates)

    @staticmethod
    @nn.nowrap
    def rng_keys():
        return ('params', 'noise')

    @staticmethod
    @nn.nowrap
    def get_weight_decay_exclusions():
        return ('bias')

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.backbone = ResNetPolicy(self.output_dim * 2 - self.output_dim//7, self.config)

    def log_prob(self, state, action, images, shape_vec=None, primitive_vec=None, return_mean=False):
        backbone_result = self.backbone(state, images, shape_vec=shape_vec, primitive_vec=primitive_vec)
        backbone_result = backbone_result.reshape((backbone_result.shape[0], -1, 13))
        mean, log_std, gripper = backbone_result[..., :6], backbone_result[..., 6:12], backbone_result[..., -1]
        mean = mean.reshape((mean.shape[0], -1))
        log_std = log_std.reshape((log_std.shape[0], -1))
        gripper = gripper.reshape((gripper.shape[0], -1))
        # mean, log_std, gripper = jnp.split(self.backbone(state, images), [self.output_dim-1, 2*(self.output_dim - 1)], axis=-1)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        action = action.reshape((action.shape[0], -1, 7))[..., :-1].reshape(mean.shape)
        log_probs = action_distribution.log_prob(action)

        if return_mean:
            return log_probs, mean, gripper
        return log_probs, gripper

    def __call__(self, state, images, shape_vec=None, primitive_vec=None, deterministic=False):
        backbone_result = self.backbone(state, images, shape_vec=shape_vec, primitive_vec=primitive_vec)
        backbone_result = backbone_result.reshape((backbone_result.shape[0], -1, 13))
        mean, log_std, gripper = backbone_result[..., :6], backbone_result[..., 6:12], backbone_result[..., -1]
        mean = mean.reshape((mean.shape[0], -1))
        log_std = log_std.reshape((log_std.shape[0], -1))
        gripper = gripper.reshape((gripper.shape[0], -1))
        # z = self.backbone(state, images, shape_vec=shape_vec)
        # mean, log_std, gripper = jnp.split(z, [self.output_dim-1, 2*(self.output_dim - 1)], axis=-1)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        gripper_act = jax.nn.sigmoid(gripper) > 0.5

        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(
                seed=self.make_rng('noise')
            )
        samples = jnp.concatenate((samples.reshape(samples.shape[0], -1, 6), gripper_act.reshape(gripper_act.shape[0], -1, 1)), axis=-1)
        samples = samples.reshape(samples.shape[0], -1)
        # samples = jnp.append(samples, gripper)
        return samples, log_prob
