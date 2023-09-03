from collections import namedtuple

import pandas as pd
from jax import numpy as jnp
from jax import random as jr
from numpyro import distributions as dist

from ramsey._src.datasets import M4Dataset
from ramsey._src.experimental.gaussian_process.kernel.stationary import (
    exponentiated_quadratic,
)


# pylint: disable=too-many-locals,invalid-name
def m4_data(interval: str = "hourly", drop_na: bool = True):
    """
    Load a data set from the M4 competition

    Parameters
    ----------
    interval: str
        either of "hourly", "daily", "weekly", "monthly", "yearly"
    drop_na: bool
        drop rows that contain NA values

    Returns
    -------
        Returns a named tuple.
    """

    train, test = M4Dataset().load(interval)
    df = pd.concat([train, test.reindex(train.index)], axis=1)
    if drop_na:
        df = df.dropna()
    y = df.values
    y = y.reshape((*y.shape, 1))
    x = jnp.arange(y.shape[1]) / train.shape[1]
    x = jnp.tile(x, [y.shape[0], 1]).reshape((y.shape[0], y.shape[1], 1))
    train_idxs = jnp.arange(train.shape[1])
    test_idxs = jnp.arange(test.shape[1]) + train.shape[1]

    return namedtuple(
        "data", ["y", "x", "train_idxs", "test_idxs"]
    )(  # type: ignore
        y, x, train_idxs, test_idxs
    )


# pylint: disable=too-many-locals,invalid-name
def sample_from_polynomial_function(
    seed, batch_size=10, order=1, num_observations=100, sigma=0.1
):
    x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
        (num_observations, 1)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        y_rng_key, coeff_rng_key, seed = jr.split(seed, 3)
        coeffs = list(jr.uniform(coeff_rng_key, shape=(order + 1, 1)) - 1)
        f = []
        for i in range(order + 1):
            f += coeffs[i] * x**i

        y = f + jr.normal(y_rng_key, shape=(num_observations, 1)) * sigma
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return namedtuple("data", "y x f")(y, x, f)


# pylint: disable=too-many-locals,invalid-name
def sample_from_sine_function(seed, batch_size=10, num_observations=100):
    x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
        (num_observations, 1)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        sample_key1, sample_key2, sample_key3, seed = jr.split(seed, 4)
        a = 2 * jr.uniform(sample_key1) - 1
        b = jr.uniform(sample_key2) - 0.5
        f = a * jnp.sin(x - b)
        y = f + jr.normal(sample_key3, shape=(num_observations, 1)) * 0.10
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return namedtuple("data", "y x f")(y, x, f)


# pylint: disable=too-many-locals,invalid-name
def sample_from_gaussian_process(
    seed, batch_size=10, num_observations=100, rho=None, sigma=None
):
    """
    Load a data set from the M4 competition

    Parameters
    ----------
    seed: jax.random.PRNGKey
        a random key for seeding
    batch_size: int
        size of batch
    num_observations: int
        number of observations per batch
    rho: Optional[float]
        the lengthscale of the kernel function
    sigma: Optional[float]
        the standard deviation of the kernel function

    Returns
    -------
    NamedTuple
        a tuple consisting of outputs (y), inputs (x) and latent GP
        realization (f)
    """

    x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
        (num_observations, 1)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        sample_key1, sample_key2, sample_key3, sample_key4, seed = jr.split(
            seed, 5
        )
        if rho is None:
            rho = dist.InverseGamma(1, 1).sample(sample_key1)
        if sigma is None:
            sigma = dist.InverseGamma(5, 5).sample(sample_key2)
        K = exponentiated_quadratic(x, x, sigma, rho)

        f = jr.multivariate_normal(
            sample_key3,
            mean=jnp.zeros(num_observations),
            cov=K + jnp.diag(jnp.ones(num_observations)) * 1e-5,
        )
        y = jr.multivariate_normal(
            sample_key4, mean=f, cov=jnp.eye(num_observations) * 0.05
        )
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return namedtuple("data", "y x f")(y, x, f)
