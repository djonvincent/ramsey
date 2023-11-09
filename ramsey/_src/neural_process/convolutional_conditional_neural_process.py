from typing import Optional, Tuple
from flax import linen as nn
from jax import numpy as jnp

from ramsey._src.family import Family, Gaussian
from ramsey._src.neural_process.neural_process import NP
from ramsey._src.experimental.gaussian_process.kernel.base import Kernel

class CCNP(NP):
    decoder: Tuple[nn.Module, Kernel]
    deterministic_encoder: Kernel
    family: Family = Gaussian()
    density: int = 10

    def setup(self):
        """Construct the networks of the class."""
        self._decoder = self.decoder
        [self._decoder_cnn, self._decoder_kernel] = (
            self.decoder[0],
            self.decoder[1],
        )
        self._deterministic_encoder = self.deterministic_encoder
        self._family = self.family

    def _encode_deterministic(
            self,
            x_context: Array,
            y_context: Array,
            x_target: Array,  # pylint: disable=unused-argument
    ):
        x = jnp.concatenate((x_context, x_target), axis=-2)
        # ensure x is 3-dimensional to match batched shape
        if x.ndim < 3:
            x = jnp.expand_dims(x, 0)
        grid_axes = jnp.meshgrid(*[jnp.linspace(0, 1, self.density)]*x.shape[-1])
        unit_grid = jnp.column_stack([ax.flatten() for ax in grid_axes])
        uniform_grid = jnp.expand_dims(unit_grid,0) * \
                       jnp.expand_dims(x.max(axis=-2) - x.min(axis=-2), 1) + \
                       x.min(axis=-2)
        K = self.deterministic_encoder(uniform_grid, x_context)
        h0 = jnp.expand_dims(K.sum(axis=-1), -1)
        h1 = K @ y_target
        h1 = h1 / h0 + 1e-8)
        h = jnp.concatenate((h0, h1), axis=-1)
        h = h.reshape(-1, *(self.density for i in range(x.shape[-1])), 2)
        return uniform_grid, h

    @staticmethod
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        return z_deterministic

    def _decode(self, representation: Array, x_target: Array, y: Array):
        uniform_grid, h = representation
        f = self._decoder_cnn(h)
        K = self._decoder_kernel(uniform_grid, x_context)
        target = jnp.concatenate([representation, x_target], axis=-1)
        family = self._family(target)
        self._check_posterior_predictive_axis(family, x_target, y)
        return family