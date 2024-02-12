import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array
from jax import random as jr
from tqdm import tqdm

from ramsey._src.neural_process.neural_process import NP, MaskedNP

__all__ = ["train_neural_process", "train_masked_np"]


@jax.jit
def _step(rngs, state, **batch):
    current_step = state.step
    rngs = {name: jr.fold_in(rng, current_step) for name, rng in rngs.items()}

    def obj_fn(params):
        _, obj = state.apply_fn(variables=params, rngs=rngs, **batch)
        return obj

    obj, grads = jax.value_and_grad(obj_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, obj


# pylint: disable=too-many-locals
def train_neural_process(
    rng_key: jr.PRNGKey,
    neural_process: NP,  # pylint: disable=invalid-name
    x: Array,  # pylint: disable=invalid-name
    y: Array,  # pylint: disable=invalid-name
    n_context: int,
    n_target: int,
    batch_size: int,
    optimizer=optax.adam(3e-4),
    n_iter=20000,
    verbose=False,
):
    r"""Train a neural process.

    Utility function to train a latent or conditional neural process, i.e.,
    a process belonging to the `NP` class.

    Parameters
    ----------
    rng_key: jax.random.PRNGKey
        a key for seeding random number generators
    neural_process: Union[NP, ANP, DANP]
        an object that inherits from NP
    x: jax.Array
        array of inputs. Should be a tensor of dimension
        :math:`b \times n \times p`
        where :math:`b` indexes a sequence of batches, e.g., different time
        series, :math:`n` indexes the number of observations per batch, e.g.,
        time points, and :math:`p` indexes the number of feats
    y: jax.Array
        array of outputs. Should be a tensor of dimension
        :math:`b \times n \times q`
        where :math:`b` and :math:`n` are the same as for :math:`x` and
        :math:`q` is the number of outputs
    n_context: int
        number of context points
    n_target: int
        number of target points
    batch_size: int
        number of elements that are samples for each gradient step, i.e.,
        number of elements in first axis of :math:`x` and :math:`y`
    optimizer: optax.GradientTransformation
        an optax optimizer object
    n_iter: int
        number of training iterations
    verbose: bool
        true if print training progress

    Returns
    -------
    Tuple[dict, jnp.Array]
        returns a tuple of trained parameters and training loss profile
    """
    train_state_rng, rng_key = jr.split(rng_key)
    state = _create_train_state(
        train_state_rng,
        neural_process,
        optimizer,
        x_context=x,
        y_context=y,
        x_target=x,
    )

    objectives = []
    for i in tqdm(range(n_iter)):
        split_rng_key, sample_rng_key, rng_key = jr.split(rng_key, 3)
        batch = _split_data(
            split_rng_key,
            x,
            y,
            n_context=n_context,
            n_target=n_target,
            batch_size=batch_size,
        )
        state, obj = _step({"sample": sample_rng_key}, state, **batch)
        objectives.append(obj)
        if (i % 100 == 0 or i == n_iter - 1) and verbose:
            elbo = -float(obj)
            print(f"ELBO at itr {i}: {elbo:.2f}")

    return state.params, np.array(objectives)


def train_masked_np(
    rng_key: jr.PRNGKey,
    neural_process: MaskedNP,  # pylint: disable=invalid-name
    data_func,
    batch_size: int,
    shuffle=False,
    optimizer=optax.adam(3e-4),
    n_iter=20000,
    n_context_max: int = 50,
    n_context_min: int = 3,
    n_target_max: int = 50,
    n_target_min: int = 3,
    verbose=False,
    split_fn=None,
):

    train_state_rng, rng_key = jr.split(rng_key)
    init_kwargs = {
        "x_context": jnp.empty((batch_size, n_context_max, 1)),
        "y_context": jnp.empty((batch_size, n_context_max, 1)),
        "x_target": jnp.empty((batch_size, n_target_max, 1)),
        "context_mask": jnp.empty((batch_size, n_context_max), dtype=int),
        "target_mask": jnp.empty((batch_size, n_target_max), dtype=int),
    }

    state = _create_train_state(
        train_state_rng, neural_process, optimizer, **init_kwargs
    )

    def get_context_target_ranges(rng_key):
        ctx_rng_key, tgt_rng_key = jr.split(rng_key, 2)
        chunks = 2
        context_chunks = jnp.linspace(
            n_context_min, n_context_max, chunks + 1
        ).astype(int)
        target_chunks = jnp.linspace(
            n_target_min, n_target_max, chunks + 1
        ).astype(int)
        context_chunk_idx = jr.randint(ctx_rng_key, (), 0, chunks)
        target_chunk_idx = jr.randint(tgt_rng_key, (), 0, chunks)

        return {
            "n_context_min": context_chunks[context_chunk_idx].item(),
            "n_context_max": context_chunks[context_chunk_idx + 1].item(),
            "n_target_min": target_chunks[target_chunk_idx].item(),
            "n_target_max": target_chunks[target_chunk_idx + 1].item(),
        }

    def split_data(
        rng_key, x, y, n_context_min, n_context_max, n_target_min, n_target_max
    ):
        context_rng_key, target_rng_key, perm_rng_key = jr.split(rng_key, 3)
        n_context = jr.randint(
            context_rng_key, (1,), n_context_min, n_context_max + 1
        )
        n_target = jr.randint(
            target_rng_key, (1,), n_target_min, n_target_max + 1
        )

        if shuffle:
            idxs = jr.permutation(
                perm_rng_key,
                jnp.repeat(
                    jnp.arange(x.shape[1])[jnp.newaxis, :], batch_size, axis=0
                ),
                axis=1,
                independent=True,
            )
        else:
            idxs = jnp.repeat(
                jnp.arange(x.shape[1])[jnp.newaxis, :], batch_size, axis=0
            )

        tiled_arange = jnp.repeat(
            jnp.arange(n_context_max)[jnp.newaxis, :], batch_size, axis=0
        )
        context_mask = (tiled_arange < n_context[:, jnp.newaxis]).astype(int)
        tiled_arange = jnp.repeat(
            jnp.arange(n_target_max)[jnp.newaxis, :], batch_size, axis=0
        )
        target_mask = (tiled_arange < n_target[:, jnp.newaxis]).astype(int)

        x_samples = jnp.take_along_axis(x, idxs[..., jnp.newaxis], axis=1)
        y_samples = jnp.take_along_axis(y, idxs[..., jnp.newaxis], axis=1)
        x_context = x_samples[:, :n_context_max, :]
        x_target = x_samples[
            :, n_context_max : (n_context_max + n_target_max), :
        ]
        y_context = y_samples[:, :n_context_max, :]
        y_target = y_samples[
            :, n_context_max : (n_context_max + n_target_max), :
        ]
        return {
            "x_context": x_context,
            "y_context": y_context,
            "context_mask": context_mask,
            "x_target": x_target,
            "y_target": y_target,
            "target_mask": target_mask,
        }

    if split_fn is None:
        _split_data = jax.jit(split_data, static_argnums=[3, 4, 5, 6])
    else:
        _split_data = jax.jit(split_fn, static_argnums=[3, 4, 5, 6])
    _data_func = jax.jit(data_func, static_argnums=[1, 2])
    objectives = []
    for i in tqdm(range(n_iter)):
        [
            data_rng_key,
            range_rng_key,
            split_rng_key,
            sample_rng_key,
            rng_key,
        ] = jr.split(rng_key, 5)
        x, y = _data_func(
            data_rng_key, batch_size, n_context_max + n_target_max
        )
        ranges = get_context_target_ranges(range_rng_key)
        batch = _split_data(split_rng_key, x, y, **ranges)
        state, obj = _step({"sample": sample_rng_key}, state, **batch)
        objectives.append(obj)
        if (i % 100 == 0 or i == n_iter - 1) and verbose:
            elbo = -float(obj)
            print(f"ELBO at itr {i}: {elbo:.2f}")

    return state.params, np.array(objectives)


# pylint: disable=too-many-locals
def _split_data(
    rng_key: jr.PRNGKey,
    x: Array,  # pylint: disable=invalid-name
    y: Array,  # pylint: disable=invalid-name
    batch_size: int,
    n_context: int,
    n_target: int,
):
    batch_rng_key, idx_rng_key, rng_key = jr.split(rng_key, 3)
    ibatch = jr.choice(
        batch_rng_key, x.shape[0], shape=(batch_size,), replace=False
    )
    idxs = jr.choice(
        idx_rng_key, x.shape[1], shape=(n_context + n_target,), replace=False
    )
    x_context = x[ibatch][:, idxs[:n_context], :]
    y_context = y[ibatch][:, idxs[:n_context], :]
    x_target = x[ibatch][:, idxs, :]
    y_target = y[ibatch][:, idxs, :]

    return {
        "x_context": x_context,
        "y_context": y_context,
        "x_target": x_target,
        "y_target": y_target,
    }


def _create_train_state(rng, model, optimizer, **init_data):
    init_key, sample_key = jr.split(rng)
    params = model.init({"sample": sample_key, "params": init_key}, **init_data)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state
