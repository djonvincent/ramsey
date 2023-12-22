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
    #grid_shape: Tuple[int, ...] = None
    #uniform_grid: Array = None
    grid_size: Tuple[float] = None
    density: int = None

    def setup(self):
        """Construct the networks of the class."""
        self._decoder = self.decoder
        [self._decoder_cnn, self._mean_kernel, self._sigma_kernel] = (
            self.decoder[0],
            self.decoder[1],
            self.decoder[2],
        )
        if self.latent_encoder is not None:
            [self._latent_encoder, self._latent_variable_encoder] = (
                self.latent_encoder[0],
                self.latent_encoder[1],
            )
        self._deterministic_encoder = self.deterministic_encoder
        self._family = self.family
        self.uniform_grid, self.grid_shape = self.construct_grid(self.grid_size, self.density)

    @staticmethod
    # pylint: disable=duplicate-code
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        if z_latent is None:
            return z_deterministic
        uniform_grid, h = z_deterministic
        z_latent = jnp.expand_dims(z_latent, tuple(range(1, h.ndim - 2)))
        z_latent = jnp.broadcast_to(z_latent, (*h.shape[:-1], z_latent.shape[-1]))
        h = jnp.concatenate((h, z_latent), axis=-1)
        return uniform_grid, h

    @staticmethod
    def construct_grid(lengths, density):
        dx = 1 / density
        grid_ticks = [jnp.arange(0, l + dx, dx) for l in lengths]
        grid_axes = jnp.meshgrid(*grid_ticks)
        uniform_grid = jnp.stack(grid_axes, axis=-1)
        return uniform_grid, tuple(len(ticks) for ticks in grid_ticks)

    def _encode_latent(
            self,
            x_context: Array,
            y_context: Array,
            context_mask: Array
    ):
        x_start = x_context.min(axis=1)[:, jnp.newaxis, :]
        xy_context = jnp.concatenate([x_context - x_start, y_context], axis=-1)
        z_latent = self._latent_encoder(xy_context)
        return self._encode_latent_gaussian(z_latent, context_mask)

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
        K_mean = self._mean_kernel(x_target, uniform_grid)
        K_sigma = self._sigma_kernel(x_target, uniform_grid)
        f0, f1 = jnp.split(f, 2, axis=-1)
        #alpha = 0.01 + 0.9*K_mean @ nn.softplus(f0).reshape(f0.shape[0], -1, f0.shape[-1])
        #beta = 0.01 + 0.9*K_sigma @ nn.softplus(f1).reshape(f1.shape[0], -1, f1.shape[-1])
        mu = K_mean @ f0.reshape(f0.shape[0], -1, f0.shape[-1])
        sigma = 0.01 + 0.9*K_sigma @ nn.softplus(f1).reshape(f1.shape[0], -1, f1.shape[-1])
        #family = self._family(jnp.concatenate((mu, sigma), axis=-1))
        #self._check_posterior_predictive_axis(family, x_target, y)
        return dist.Normal(loc=mu, scale=sigma)
        #return dist.Beta(alpha, beta)
