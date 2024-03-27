from dataclasses import KW_ONLY, dataclass
from typing import Optional, Tuple, Callable, Dict

import numpy as np
import numpyro.distributions as dist
from chex import assert_axis_dimension
from flax import linen as nn
from jax import Array, vmap
from jax import numpy as jnp
from jax import lax
from jax.scipy.special import ndtri
from numpyro.distributions.util import clamp_probs

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
        self._encoder_kernel, self._decoder_kernel = [
            ExponentiatedQuadratic(
                rho_init=nn.initializers.constant(jnp.log(rho)),
                sigma=1
            )
            for _ in range(2)
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
        **kwargs,
    ):
        batch_size = x_context.shape[0]
        x_start = x_context.min(axis=1)[:, jnp.newaxis, :]
        shifted_grid = self.uniform_grid[jnp.newaxis, ...] + x_start
        K = self._encoder_kernel(shifted_grid, x_context)
        K *= context_mask[:, jnp.newaxis, :]
        h0 = K.sum(axis=-1, keepdims=True)
        h1 = K @ y_context
        h1 = h1 / (h0 + 1e-8)
        h = jnp.concatenate((h0, h1), axis=-1)
        h = h.reshape((batch_size, *self.grid_shape, -1))
        return shifted_grid, h

    def _decode(
        self,
        representation: Tuple[Array, Array],
        x_target: Array,
        y: Array,
        target_mask: Array,
        **kwargs,
    ):
        uniform_grid, h = representation
        f = self._decoder_cnn(h)
        K = self._decoder_kernel(x_target, uniform_grid)
        K /= K.sum(axis=-1, keepdims=True)
        f0, f1 = jnp.split(f, 2, axis=-1)
        f1 = nn.softplus(f1)
        mu = K @ f0.reshape(f0.shape[0], -1, f0.shape[-1])
        sigma = K @ f1.reshape(f1.shape[0], -1, f1.shape[-1])
        sigma += 0.01
        return dist.Normal(loc=mu, scale=sigma)



class CCNPWithScale(CCNP):
    _: KW_ONLY
    logit_transformation: Callable[[Array], Array] = None
    likelihood_fn: Callable[[Array, Array, Array], dist.Distribution] = None
    def _encode_deterministic(
            self,
            x_context: Array,
            y_context: Array,
            x_target: Array,
            context_mask: Array,
            target_mask: Array,
            loc: Array,
            scale: Array,
            **kwargs,
    ):
        return super()._encode_deterministic(
            x_context,
            (y_context - loc) / scale,
            x_target,
            context_mask,
            target_mask,
            **kwargs,
        )

    def _decode(
            self,
            representation: Tuple[Array, Array],
            x_target: Array,
            y: Array,
            target_mask: Array,
            loc: Array,
            scale: Array,
            train: bool,
            **kwargs,
    ):
        uniform_grid, h = representation
        f = self._decoder_cnn(h, train=train)
        f = f.reshape(f.shape[0], -1, f.shape[-1])
        f = self.logit_transformation(f, loc, scale)
        K = self._decoder_kernel(x_target, uniform_grid)
        K /= K.sum(axis=-1, keepdims=True)
        f = K @ f
        return self.likelihood_fn(f, loc, scale)


class CCNPWithFeatures(CCNPWithScale):
    _: KW_ONLY
    num_embeddings: int
    embed_dim: int
    def setup(self):
        super().setup()
        self.embedding = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embed_dim
        )

    def _decode(
            self,
            representation: Tuple[Array, Array],
            x_target: Array,
            y: Array,
            target_mask: Array,
            loc: Array,
            scale: Array,
            features: Array,
            **kwargs,
    ):
        uniform_grid, h = representation
        embeddings = self.embedding(features)
        embeddings = jnp.broadcast_to(
            embeddings[:, *[jnp.newaxis]*len(self.grid_shape), :],
            (embeddings.shape[0], *self.grid_shape, self.embed_dim)
        )
        h = jnp.concatenate((h, embeddings), axis=-1)
        return super()._decode(
            (uniform_grid, h),
            x_target,
            y,
            target_mask,
            scale,
            loc,
            **kwargs
        )


class ConvGNP(CCNPWithScale):
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

    def _decode(
            self,
            representation: Tuple[Array, Array],
            x_target: Array,
            y: Array,
            target_mask: Array,
            loc: Array,
            scale: Array,
            train: bool,
            **kwargs,
    ):
        # Only implemented for 1 dimensional targets
        assert_axis_dimension(y, -1, 1)

        uniform_grid, h = representation

        K = self._decoder_kernel(x_target, uniform_grid)
        K /= K.sum(axis=-1, keepdims=True)
        K *= target_mask[..., jnp.newaxis]

        f = self._decoder_cnn(h, train=train)
        mu_v = f[..., :2].reshape(f.shape[0], -1, 2)
        if self.likelihood_fn is not None:
            mu_v = self.logit_transformation(mu_v, loc, scale)
        mu_v = K @ mu_v
        g = K @ f[..., 2:].reshape(f.shape[0], -1, f[..., 2:].shape[-1])
        mu = (mu_v[..., 0] * target_mask).sum(axis=1) / target_mask.sum(axis=1)
        mu = jnp.broadcast_to(mu[:, jnp.newaxis], target_mask.shape) * target_mask
        v = mu_v[..., 1:2]

        if self.likelihood_fn is not None:
            std = v[..., 0]
            cov = exponentiated_quadratic(g, g, 1, 1)
            cov += 1e-3 * jnp.eye(cov.shape[-1])
            copula_marginal = self.likelihood_fn(mu, std)
            return dist.GaussianCopula(
                copula_marginal, correlation_matrix=cov
            )

        # Use the 'kvv' formulation for covariance matrix
        cov = exponentiated_quadratic(g, g, 1, 1) * (v @ v.transpose(0, 2, 1))

        cov += 1e-3 * jnp.eye(cov.shape[-1])

        # Set diagonal entries for masked out positions to 1 so that the log
        # determinant is unchanged
        cov += (
            jnp.eye(cov.shape[-1])[jnp.newaxis, ...]
            * (1 - 1e-3)
            * (1 - target_mask)[..., jnp.newaxis]
        )

        return dist.TransformedDistribution(
            dist.MultivariateNormal(loc=mu, covariance_matrix=cov),
            dist.transforms.AffineTransform(loc=loc[..., 0], scale=scale[..., 0])
        )

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

        z_deterministic = self._encode_deterministic(
            x_context, y_context, x_target, context_mask, target_mask, **kwargs
        )
        representation = self._concat_and_tile(
            z_deterministic, None, num_observations
        )
        pred_fn = self._decode(
            representation, x_target, y_context, target_mask, **kwargs
        )
        loglik = pred_fn.log_prob(y_target[..., 0] * target_mask)
        # Correction for the different dimensionality of masked target
        loglik += 0.5 * jnp.log(2 * jnp.pi) * (1 - target_mask).sum(axis=-1)
        elbo = jnp.mean(loglik)

        return pred_fn, -elbo


class ConditionalGaussianCopula(dist.GaussianCopula):
    def __init__(
        self,
        marginal_dist,
        loc,
        correlation_matrix=None,
        correlation_cholesky=None,
        *,
        validate_args=None,
    ):
        if len(marginal_dist.event_shape) > 0:
            raise ValueError("`marginal_dist` needs to be a univariate distribution.")

        self.marginal_dist = marginal_dist
        self.base_dist = dist.MultivariateNormal(
            loc=loc,
            covariance_matrix=correlation_matrix,
            scale_tril=correlation_cholesky,
        )

        event_shape = self.base_dist.event_shape
        batch_shape = lax.broadcast_shapes(
            self.marginal_dist.batch_shape[:-1],
            self.base_dist.batch_shape,
        )

        super(dist.GaussianCopula, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )
