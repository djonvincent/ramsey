from typing import Optional, Tuple
from flax import linen as nn
from jax import Array
from jax import numpy as jnp
import numpyro.distributions as dist

from ramsey._src.family import Family, Gaussian
from ramsey._src.neural_process.neural_process import MaskedNP
from ramsey._src.experimental.gaussian_process.kernel.base import Kernel

__all__ = ["CCNP"]

class CCNP(MaskedNP):
    decoder: Tuple[nn.Module, Kernel, Kernel]
    deterministic_encoder: Kernel
    family: Family = Gaussian()
    grid_shape: Tuple[int, ...] = None
    uniform_grid: Array = None
    density: int = 10

    def setup(self):
        """Construct the networks of the class."""
        self._decoder = self.decoder
        [self._decoder_cnn, self._mean_kernel, self._sigma_kernel] = (
            self.decoder[0],
            self.decoder[1],
            self.decoder[2],
        )
        self._deterministic_encoder = self.deterministic_encoder
        self._family = self.family


    @staticmethod
    # pylint: disable=duplicate-code
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        return z_deterministic

    @staticmethod
    def construct_grid(x, density):
        x_range = jnp.stack((x.min(axis=(0,1)), x.max(axis=(0,1))), axis=1)
        dx = 1 / density
        grid_ticks = [jnp.arange(r[0], r[1] + dx, dx) for r in x_range]
        grid_axes = jnp.meshgrid(*grid_ticks)
        uniform_grid = jnp.stack(grid_axes, axis=-1)
        return uniform_grid

    def _encode_deterministic(
            self,
            x_context: Array,
            y_context: Array,
            x_target: Array,
            context_mask: Array,
            target_mask: Array,
    ):
        x_start = x_context.min(axis=1)[:, jnp.newaxis, :]
        K = self.deterministic_encoder(self.uniform_grid[jnp.newaxis, ...] + x_start, x_context)
        K = K * context_mask[:, jnp.newaxis, :]
        h0 = jnp.expand_dims(K.sum(axis=-1), -1)
        h1 = K @ y_context
        h1 = h1 / (h0 + 1e-8)
        h = jnp.concatenate((h0, h1), axis=-1)
        h = h.reshape((-1, *self.grid_shape, 2))
        return self.uniform_grid[jnp.newaxis, ...] + x_start, h

    def _decode(self, representation: Tuple[Array, Array], x_target: Array, y: Array):
        uniform_grid, h = representation
        f = self._decoder_cnn(h)
        # why is uniform grid being used here? mistake?
        K_mean = self._mean_kernel(x_target, uniform_grid)
        K_sigma = self._sigma_kernel(x_target, uniform_grid)
        f0, f1 = jnp.split(f, 2, axis=-1)
        mu = K_mean @ f0.reshape(f0.shape[0], -1, f0.shape[-1])
        sigma = 0.01 + 0.9*K_sigma @ nn.softplus(f1).reshape(f1.shape[0], -1, f1.shape[-1])
        #family = self._family(jnp.concatenate((mu, sigma), axis=-1))
        #self._check_posterior_predictive_axis(family, x_target, y)
        return dist.Normal(loc=mu, scale=sigma)
