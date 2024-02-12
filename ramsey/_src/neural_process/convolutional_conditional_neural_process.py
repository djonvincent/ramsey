from dataclasses import KW_ONLY, dataclass
from typing import Optional, Tuple

import numpy as np
import numpyro.distributions as dist
from chex import assert_axis_dimension
from flax import linen as nn
from jax import Array
from jax import numpy as jnp

# pylint: disable=line-too-long
from ramsey._src.experimental.gaussian_process.kernel.stationary import (
    ExponentiatedQuadratic,
    exponentiated_quadratic,
)
from ramsey._src.neural_process.neural_process import MaskedNP

__all__ = ["CCNP", "ConvGNP"]


@dataclass
class CCNP(MaskedNP):
    """A convolutional conditional neural process.

    Implements the model from [1].

    Attributes
    ----------
    deterministic_encoder: flax.linen.Module
        a CNN to be used as the encoder
    family: Family
        distributional family of the response variable
    grid_size: Tuple[float]
        a tuple of floats the same size as the dimensionality of x. Specifies
        the lengths of the dimensions of the grid used to discretise the input
        points
    density: float
        the density of points on the grid. I.e. a density of 1 will result in 1
        grid point per unit
    extend_grid_to_multiple_of: Optional[Tuple[int]]
        a tuple of ints the same size as the dimensionality of x. If specified,
        the number of points in the corresponding axis of the grid will be
        rounded up to that number. Useful when using a U-Net as the encoder

    References
    ----------
    .. [1] Gordon, Jonathan, et al. "Convolutional conditional neural
           processes". ICLR. 2020.
    """

    # pylint: disable=too-many-instance-attributes
    _: KW_ONLY
    grid_size: Tuple[float]
    density: float
    extend_grid_to_multiple_of: Optional[Tuple[int]] = None

    def setup(self):
        """Construct the networks of the class."""
        rho = 2 / self.density
        self._decoder_cnn = self.decoder
        self._deterministic_encoder, self._mean_kernel, self._sigma_kernel = [
            ExponentiatedQuadratic(
                rho_init=nn.initializers.constant(jnp.log(rho))
            )
            for _ in range(3)
        ]
        self._family = self.family
        self.uniform_grid, self.grid_shape = self.construct_grid(
            self.grid_size, self.density, self.extend_grid_to_multiple_of
        )

    @staticmethod
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        return z_deterministic

    @staticmethod
    def construct_grid(
        lengths: Tuple[float],
        density: float,
        round_to: Optional[Tuple[int]] = None,
    ):
        dx = 1 / density
        grid_ticks = [jnp.arange(0, ln + dx, dx) for ln in lengths]
        if round_to is not None:
            padding = [
                int(np.ceil(len(ticks) / r) * r) - len(ticks)
                for ticks, r in zip(grid_ticks, round_to)
            ]
            grid_ticks = [
                jnp.arange(0, ln + dx + dx * p, dx)
                for ln, p in zip(lengths, padding)
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
        K = self._deterministic_encoder(
            self.uniform_grid[jnp.newaxis, ...] + x_start, x_context
        )
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
        target_mask: Array,
    ):
        uniform_grid, h = representation
        f = self._decoder_cnn(h)
        K_mean = self._mean_kernel(x_target, uniform_grid)
        K_sigma = self._sigma_kernel(x_target, uniform_grid)
        f0, f1 = jnp.split(f, 2, axis=-1)
        f1 = nn.softplus(f1)
        mu = K_mean @ f0.reshape(f0.shape[0], -1, f0.shape[-1])
        sigma = K_sigma @ f1.reshape(f1.shape[0], -1, f1.shape[-1])
        sigma += 0.01
        return dist.Normal(loc=mu, scale=sigma)


class ConvGNP(CCNP):
    """A convolutional conditional neural process with output correlations.

    Implements the model from [1]. Currently only supports 1-D y values.

    Attributes
    ----------
    deterministic_encoder: flax.linen.Module
        a CNN to be used as the encoder
    grid_size: Iterable[float]
        a tuple of floats the same size as the dimensionality of x. Specifies
        the lengths of the dimensions of the grid used to discretise the input
        points
    density: float
        the density of points on the grid. I.e. a density of 1 will result in 1
        grid point per unit
    extend_grid_to_multiple_of: Optional[Iterable[int]]
        a tuple of ints the same size as the dimensionality of x. If specified,
        the number of points in the corresponding axis of the grid will be
        rounded up to that number. Useful when using a U-Net as the encoder
    copula_marginal: Optional[numpyro.distributions.Distribution]
        a univariate numpyro distribution. If specified, the model will return
        a Gaussian copula with this as the marginal. Otherwise, a multivariate
        Gaussian is returned

    References
    ----------
    .. [1] Markou, Stratis, et al. "Practical conditional neural processes via
           tractable dependent predictions." ICLR. 2022.
    """

    # pylint: disable=too-many-instance-attributes
    copula_marginal: Optional[dist.Distribution] = None

    def setup(self):
        """Construct the networks of the class."""
        self._decoder_cnn = self.decoder
        rho = 2 / self.density
        [
            self._deterministic_encoder,
            self._mean_kernel,
            self._cov_kernel,
            self._cov_scale_kernel,
        ] = [
            ExponentiatedQuadratic(
                rho_init=nn.initializers.constant(jnp.log(rho))
            )
            for _ in range(4)
        ]
        self._family = self.family
        self.uniform_grid, self.grid_shape = self.construct_grid(
            self.grid_size, self.density, self.extend_grid_to_multiple_of
        )

    def _decode(
        self,
        representation: Tuple[Array, Array],
        x_target: Array,
        y: Array,
        target_mask: Array,
    ):
        # Only implemented for 1 dimensional targets
        assert_axis_dimension(y, -1, 1)

        uniform_grid, h = representation

        K_mean = self._mean_kernel(x_target, uniform_grid)
        K_mean *= target_mask[..., jnp.newaxis]
        K_cov = self._cov_kernel(x_target, uniform_grid)
        K_cov *= target_mask[..., jnp.newaxis]
        K_cov_scale = self._cov_scale_kernel(x_target, uniform_grid)
        K_cov_scale *= target_mask[..., jnp.newaxis]

        f = self._decoder_cnn(h)
        mu = K_mean @ f[..., 0:1].reshape(f.shape[0], -1)
        v = K_cov_scale @ f[..., 1:2].reshape(f.shape[0], -1, 1)
        g = K_cov @ f[..., 2:].reshape(f.shape[0], -1, f[..., 2:].shape[-1])

        # Use the 'kvv' formulation for covariance matrix
        cov = exponentiated_quadratic(g, g, 1, 1) * (v @ v.transpose(0, 2, 1))
        cov += 0.01 * jnp.eye(cov.shape[-1])

        # Set diagonal entries for masked out positions to 1 so that the log
        # determinant is unchanged
        cov += (
            jnp.eye(cov.shape[-1])[jnp.newaxis, ...]
            * 0.99
            * (1 - target_mask)[..., jnp.newaxis]
        )

        if self.copula_marginal is not None:
            return dist.GaussianCopula(
                self.copula_marginal, correlation_matrix=cov
            )
        return dist.MultivariateNormal(loc=mu, covariance_matrix=cov)

    def _negative_elbo(  # pylint: disable=too-many-locals
        self,
        x_context: Array,
        y_context: Array,
        x_target: Array,
        y_target: Array,
        context_mask: Array,
        target_mask: Array,
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
        pred_fn = self._decode(representation, x_target, y_context, target_mask)
        loglik = pred_fn.log_prob(y_target[..., 0] * target_mask)
        # Correction for the different dimensionality of masked target
        loglik += 0.5 * jnp.log(2 * jnp.pi) * (1 - target_mask).sum(axis=-1)
        elbo = jnp.mean(loglik)

        return pred_fn, -elbo
