"""Matched-init CD training for the MNIST RBM (Week 6 Part 2).

Design (per review): one JIT-compiled epoch function using lax.scan over
minibatches; a plain Python loop over the 20 epochs calls it so per-epoch
metrics (CD surrogate mean, held-out one-step reconstruction Hamming
fraction) can be logged without repeated compilation or stacking the whole
epoch axis of MNIST in memory. Shared parameters/permutations are built once
with recorded SHA-256 hashes so CD-1 and CD-10 are exactly matched.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from ebm.rbm import (cd_surrogate, gibbs_chain, init_params,
                     one_step_reconstruction_hamming)


def sha256_array(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def build_shared_init(master_key, *, n_epochs: int, n_items: int = 60000,
                      n_visible: int = 784, n_hidden: int = 128,
                      pixel_mean=None):
    """Exactly-shared init for CD-1/CD-10: parameters + one permutation per
    epoch, all deterministic in the master key, with SHA-256 hashes recorded.
    """
    key_init, key_perm = jax.random.split(master_key)
    params = init_params(key_init, n_visible=n_visible, n_hidden=n_hidden,
                         pixel_mean=pixel_mean)
    keys = jax.random.split(key_perm, n_epochs)
    perms = jax.vmap(lambda k: jax.random.permutation(k, n_items))(keys)
    hashes = {
        "a": sha256_array(params["a"]),
        "b": sha256_array(params["b"]),
        "W": sha256_array(params["W"]),
        "perms": [sha256_array(p) for p in perms],
        "master_key_bits": sha256_array(np.asarray(master_key)),
    }
    return {"params": params, "perms": perms, "hashes": hashes,
            "master_key": master_key, "perm_keys": keys}


def make_epoch_fn(k: int, lr: float, batch_size: int, n_batches: int):
    """Build the JIT-compiled epoch function (one compilation total).

    Epoch inputs: params, batched images (n_batches, batch_size, n_visible),
    one epoch key (split internally into per-batch step keys).
    Returns (params, per-batch CD surrogate values).
    """

    def batch_body(p, inputs):
        v0, step_key = inputs
        vk = gibbs_chain(v0, p, step_key, k)
        loss, grads = jax.value_and_grad(cd_surrogate)(p, v0, vk)
        p_new = jax.tree_util.tree_map(lambda pp, g: pp - lr * g, p, grads)
        return p_new, loss

    @jax.jit
    def epoch(params, batched, epoch_key):
        keys = jax.random.split(epoch_key, n_batches)
        params_new, losses = jax.lax.scan(batch_body, params,
                                          (batched, keys))
        return params_new, losses

    return epoch


def epoch_batches(images: np.ndarray, perm: np.ndarray, batch_size: int):
    """One epoch's minibatches in stored-permutation order (Python-side)."""
    n_batches = len(images) // batch_size
    ordered = images[np.asarray(perm)]
    return ordered[: n_batches * batch_size].reshape(
        n_batches, batch_size, images.shape[-1])


def recon_hamming(v_test, params, key):
    """Mean one-step sampled reconstruction Hamming fraction on a set."""
    return float(jnp.mean(one_step_reconstruction_hamming(v_test, params, key)))
