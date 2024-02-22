from typing import Callable, Optional
import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat


def mask_union(mask1, mask2):
    return jnp.logical_or(mask1 > 0, mask2 > 0).astype(jnp.float32)


def mask_intersection(mask1, mask2):
    return jnp.logical_and(mask1 > 0, mask2 > 0).astype(jnp.float32)


def mask_not(mask):
    return 1.0 - mask


def mask_select(mask, this, other=None):
    if other is None:
        other = jnp.array(0, dtype=this.dtype)
    if len(this.shape) == 3:
        mask = jnp.expand_dims(mask, axis=-1)
    return jnp.where(mask == 0.0, this, other)


def no_mask(x):
    return jnp.zeros(x.shape[:2])


def all_mask(x):
    return jnp.ones(x.shape[:2])


def patch_mse_loss(patch_output, patch_target, valid=None):
    if valid is None:
        valid = all_mask(patch_target)
    valid_ratio = jnp.sum(valid, axis=-1) / valid.shape[-1]
    return jnp.mean(
        jnp.mean(
            jnp.where(
                valid > 0.0,
                jnp.mean(jnp.square(patch_target - patch_output), axis=-1),
                jnp.array(0.0),
            ),
            axis=-1,
        )
        / valid_ratio
    )


def extract_patches(inputs, patch_size):
    batch, height, width, channels = inputs.shape
    height, width = height // patch_size, width // patch_size
    x = jnp.reshape(inputs, (batch, height, patch_size,
                    width, patch_size, channels))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * width, patch_size**2 * channels))
    return x


def merge_patches(inputs, patch_size):
    batch, length, _ = inputs.shape
    height = width = int(length**0.5)
    x = jnp.reshape(inputs, (batch, height, width, patch_size, patch_size, -1))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * patch_size, width * patch_size, -1))
    return x


def extract_patches_video(
    inputs: jnp.ndarray, patch_size: int, time_size: int
) -> jnp.ndarray:
    batch, time, height, width, channels = inputs.shape
    time = time // time_size
    height = height // patch_size
    width = width // patch_size

    x = jnp.reshape(
        inputs,
        (batch, time, time_size, height, patch_size, width, patch_size,
            channels),
    )  # B(0), T(1), T_S(2), H(3), H_S(4), W(5), W_S(6), C(7)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    x = jnp.reshape(
        x, (batch, time, height * width, time_size * patch_size**2 * channels)
    )
    return x


def merge_patches_video(
    inputs: jnp.ndarray, patch_size: int, time_size: int
) -> jnp.ndarray:
    batch, time, length, _ = inputs.shape
    height = width = int(length**0.5)
    x = jnp.reshape(
        inputs, (batch, time, height, width,
                 time_size, patch_size, patch_size, -1)
    )
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    x = jnp.reshape(
        x, (batch, time * time_size, height *
            patch_size, width * patch_size, -1)
    )
    return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0,
    )


def interpolate_positional_embedding(pos_embed, orig_length, new_length):
    assert pos_embed.shape[1] == orig_length
    D = pos_embed.shape[2]
    orig_grid = jnp.arange(orig_length, dtype=jnp.float32)
    new_grid = jnp.linspace(0, orig_length - 1, new_length)
    new_pos_embed = []
    for i in range(D):
        new_pos_embed.append(jnp.interp(
            new_grid, orig_grid, pos_embed[0, :, i]))

    new_pos_embed = jnp.stack(new_pos_embed, axis=-1)
    print('interpolate positional embedding', new_pos_embed.shape)
    new_pos_embed = jnp.expand_dims(new_pos_embed, 0)
    return new_pos_embed


def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length**0.5)
    assert grid_size * \
        grid_size == length, f"grid_size: {grid_size}, length: {length}"

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)


def index_sequence(x, ids):
    return x[:, ids, ...]


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    depth: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i in range(self.depth):
            y = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.xavier_uniform()
            )(x)
            y = nn.gelu(y)
            y = nn.LayerNorm()(y)
            if i > 0:
                x = x + y
            else:
                x = y

        x = nn.Dense(self.output_dim,
                     kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.0
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init,
                     name="fc1")(inputs)

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init, name="fc2")(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x


class Attention(nn.Module):
    """Modified from flax_models to support mask"""

    dim: int
    num_heads: int = 8
    use_bias: bool = False
    att_drop: float = 0
    proj_drop: float = 0
    kernel_init: Callable = nn.initializers.xavier_uniform()
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self,
                 inputs,
                 deterministic=None,
                 attn_mask=None,
                 padding_mask=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        batch, n, channels = inputs.shape
        scale = (self.dim // self.num_heads) ** -0.5
        qkv = nn.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
        )(inputs)
        qkv = jnp.reshape(
            qkv, (batch, n, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        # shape: (3, batch, num_heads, n, channels // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale
        # shape of attention: (batch, num_heads, n, n)
        if attn_mask is not None:
            attention = jnp.where(attn_mask > 0, attention, jnp.array(-1e7))

        if padding_mask is not None:
            padding_mask = jnp.expand_dims(jnp.expand_dims(padding_mask, 1), 1)
            padding_mask = jnp.broadcast_to(padding_mask, attention.shape)
            attention = jnp.where(padding_mask > 0, jnp.array(-1e7), attention)

        attention = nn.softmax(attention, axis=-1)
        self.sow("intermediates", "attention", attention)
        attention = nn.Dropout(self.att_drop)(attention, deterministic)

        x = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels)
        x = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic)

        return x


class Block(nn.Module):
    emb_dim: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    att_drop: float = 0.0
    drop: float = 0.0

    @nn.compact
    def __call__(self,
                 inputs,
                 deterministic=False,
                 attn_mask=None,
                 padding_mask=None):
        x = nn.LayerNorm()(inputs)
        x = Attention(
            self.emb_dim, self.num_heads, True, self.att_drop, self.drop
        )(x, deterministic, attn_mask, padding_mask)
        inputs = inputs + x

        x = nn.LayerNorm()(inputs)
        x = TransformerMLP(
            self.emb_dim * self.mlp_ratio,
            self.emb_dim,
            self.drop,
        )(x, deterministic)
        return inputs + x


class Transformer(nn.Module):
    emb_dim: int = 1024
    depth: int = 24
    att_drop: float = 0
    drop: float = 0
    num_heads: int = 16
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, deterministic=False, padding_mask=None):
        for _ in range(self.depth):
            x = Block(
                self.emb_dim,
                self.num_heads,
                self.mlp_ratio,
                self.att_drop,
                self.drop,
                self.drop_path,
            )(x, deterministic, padding_mask)

        x = nn.LayerNorm()(x)
        return x


