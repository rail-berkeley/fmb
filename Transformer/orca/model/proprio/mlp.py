import functools as ft
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp

import numpy as np


ModuleDef = Any



class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = '512-512'

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split('-')]
        for h in hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)

        return nn.Dense(self.output_dim)(x)


mlp_configs = {
    "mlp-3": FullyConnectedNetwork, 

}
