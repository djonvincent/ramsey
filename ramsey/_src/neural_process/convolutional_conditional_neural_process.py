from typing import Optional, Tuple, Iterable
from flax import linen as nn
from jax import Array
from jax import numpy as jnp
import numpyro.distributions as dist
from chex import assert_axis_dimension
from jax import scipy as jsp
import numpy as np

from ramsey._src.family import Family, Gaussian
from ramsey._src.neural_process.neural_process import MaskedNP
from ramsey._src.experimental.gaussian_process.kernel.base import Kernel
from ramsey._src.experimental.gaussian_process.kernel.stationary import exponentiated_quadratic

__all__ = ["CCNP", "ConvGNP", "ConvGNPGP"]

class CCNP(MaskedNP):
    decoder: Tuple[nn.Module, Kernel, Kernel]
    deterministic_encoder: Kernel
    family: Family = Gaussian()
    grid_size: Iterable[float] = None
    density: float = None
    extend_grid_to_multiple_of: Iterable[int] = None

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
        self.uniform_grid, self.grid_shape = self.construct_grid(
            self.grid_size,
            self.density,
            self.extend_grid_to_multiple_of
        )

    @staticmethod
    # pylint: disable=duplicate-code
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        return z_deterministic

    @staticmethod
    def construct_grid(
            lengths: Iterable[float],
            density: float,
            round_to: Iterable[int] = None
    ):
        dx = 1 / density
        grid_ticks = [jnp.arange(0, l + dx, dx) for l in lengths]
        if round_to is not None:
            padding = [
                int(np.ceil(len(ticks)/r)*r) - len(ticks)
                for ticks, r in zip(grid_ticks, round_to)
            ]
            grid_ticks = [
                jnp.arange(0, l + dx + dx*p, dx)
                for l, p in zip(lengths, padding)
            ]
        grid_axes = jnp.meshgrid(*grid_ticks)
        uniform_grid = jnp.stack(grid_axes, axis=-1)
        return uniform_grid, tuple(len(ticks) for ticks in grid_ticks)

    def _encode_deterministic(
            self,
            x_context: Array,
            y_context: Array,
            x_target: Array,
            context_mask: Array,
            target_mask: Array,
    ):
        batch_size = x_context.shape[0]
        x_start = x_context.min(axis=1)[:, jnp.newaxis, :]
        K = self.deterministic_encoder(self.uniform_grid[jnp.newaxis, ...] + x_start, x_context)
        K = K * context_mask[:, jnp.newaxis, :]
        h0 = jnp.expand_dims(K.sum(axis=-1), -1)
        h1 = K @ y_context
        h1 = h1 / (h0 + 1e-8)
        h = jnp.concatenate((h0, h1), axis=-1)
        h = h.reshape((batch_size, *self.grid_shape, -1))
        return self.uniform_grid[jnp.newaxis, ...] + x_start, h

    def _decode(
            self,
            representation: Tuple[Array, Array],
            x_target: Array,
            y: Array,
            target_mask: Array
    ):
        uniform_grid, h = representation
        f = self._decoder_cnn(h)
        K_mean = self._mean_kernel(x_target, uniform_grid)
        K_sigma = self._sigma_kernel(x_target, uniform_grid)
        f0, f1 = jnp.split(f, 2, axis=-1)
        mu = K_mean @ f0.reshape(f0.shape[0], -1, f0.shape[-1])
        sigma = 0.01 + 0.9*K_sigma @ nn.softplus(f1).reshape(f1.shape[0], -1, f1.shape[-1])
        return dist.Normal(loc=mu, scale=sigma)

class ConvGNP(CCNP):
    decoder: Tuple[nn.Module, Kernel, Kernel, Kernel]
    copula_marginal: dist.Distribution = None

    def setup(self):
        """Construct the networks of the class."""
        self._decoder = self.decoder
        [self._decoder_cnn, self._mean_kernel, self._cov_kernel, self._cov_scale_kernel] = (
            self.decoder[0],
            self.decoder[1],
            self.decoder[2],
            self.decoder[3],
        )
        self._deterministic_encoder = self.deterministic_encoder
        self._family = self.family
        self.uniform_grid, self.grid_shape = self.construct_grid(
            self.grid_size,
            self.density,
            self.extend_grid_to_multiple_of
        )

    def _decode(
            self,
            representation: Tuple[Array, Array],
            x_target: Array,
            y: Array,
            target_mask: Array
    ):
        # Only implemented for 1 dimensional targets
        assert_axis_dimension(y, -1, 1)

        uniform_grid, h = representation
        f = self._decoder_cnn(h)
        batch_size = f.shape[0]
        K_mean = self._mean_kernel(x_target, uniform_grid)
        K_mean *= target_mask[..., jnp.newaxis]
        K_cov = self._cov_kernel(x_target, uniform_grid)
        K_cov *= target_mask[..., jnp.newaxis]
        K_cov_scale = self._cov_scale_kernel(x_target, uniform_grid)
        K_cov_scale *= target_mask[..., jnp.newaxis]
        m_, v_, g_ = f[..., 0:1], f[..., 1:2], f[..., 2:]
        m = K_mean @ m_.reshape(batch_size, -1, m_.shape[-1])
        v = K_cov_scale @ v_.reshape(batch_size, -1, v_.shape[-1])
        g = K_cov @ g_.reshape(batch_size, -1, g_.shape[-1])
        cov = exponentiated_quadratic(g, g, 1, 1) * (v @ v.transpose(0, 2, 1))
        cov = cov + jnp.eye(cov.shape[-1])[jnp.newaxis, ...] * (0.99/0.9)*(1-target_mask)[:, jnp.newaxis, :]
        cov += 0.01 * jnp.eye(x_target.shape[1])
        if self.copula_marginal is not None:
            return dist.GaussianCopula(self.copula_marginal, correlation_matrix=cov)
        return dist.MultivariateNormal(loc=m[:, :, 0], covariance_matrix=cov)

    def _negative_elbo(  # pylint: disable=too-many-locals
        self,
        x_context: Array,
        y_context: Array,
        x_target: Array,
        y_target: Array,
        context_mask: Array,
        target_mask: Array,
        **kwargs
    ):
        _, num_observations, _ = x_target.shape

        if context_mask is None:
            context_mask = jnp.ones(x_context.shape[:-1], dtype=int)
        if target_mask is None:
            target_mask = jnp.ones(x_target.shape[:-1], dtype=int)

        assert_axis_dimension(context_mask, 0, x_context.shape[0])
        assert_axis_dimension(context_mask, 1, x_context.shape[1])
        assert_axis_dimension(target_mask, 0, x_target.shape[0])
        assert_axis_dimension(target_mask, 1, x_target.shape[1])

        z_latent = None

        z_deterministic = self._encode_deterministic(
            x_context, y_context, x_target, context_mask, target_mask
        )
        representation = self._concat_and_tile(
            z_deterministic, z_latent, num_observations
        )
        pred_fn = self._decode(representation, x_target, x_context, y_context, target_mask)
        loglik = pred_fn.log_prob(y_target[..., 0] * target_mask)
        # Correction for the different dimensionality of masked target
        loglik += 0.5*jnp.log(2*jnp.pi) * (1-target_mask).sum(axis=-1)
        elbo = jnp.mean(loglik)

        return pred_fn, -elbo

