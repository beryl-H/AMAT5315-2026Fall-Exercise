"""Part 3: MLP next-token model (Bengio-style one-token context).

input = previous BPE id; embedding 64; one hidden layer of 512 tanh units;
softmax over 512 tokens; all linear maps biased. 328704 parameters.
Frozen hyperparameters in MLP_HPARAMS.

Exact-minibatch optimization: the compressed path computes logits for ALL 512
previous ids once per step and gathers the rows of the ACTUAL 16384 sampled
previous ids (no dropout anywhere, so this is exactly the minibatch objective,
not weighted full-data training)."""
import numpy as np
import jax
import jax.numpy as jnp

from src.nn_common import stable_ce

MLP_HPARAMS = {"steps": 4000, "batch": 16384, "peak_lr": 0.006, "warmup": 200,
               "end_frac": 0.1, "betas": (0.9, 0.999), "wd": 0.01, "seed": 0,
               "d_embed": 64, "hidden": 512}


def init_mlp(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "E": jax.random.normal(k1, (512, 64), dtype=jnp.float32) * 0.02,
        "W1": jax.random.normal(k2, (64, 512), dtype=jnp.float32) * 0.02,
        "b1": jnp.zeros(512, dtype=jnp.float32),
        "W2": jax.random.normal(k3, (512, 512), dtype=jnp.float32) * 0.02,
        "b2": jnp.zeros(512, dtype=jnp.float32),
    }


def num_params(params):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def forward_ids(params, prev_ids):
    """Logits [B, 512] for the given previous-token ids (direct path)."""
    h0 = params["E"][prev_ids]                      # B x 64
    h1 = jnp.tanh(h0 @ params["W1"] + params["b1"])  # B x 512
    return h1 @ params["W2"] + params["b2"]


def forward_all512(params):
    """Logits [512, 512] for every possible previous id, computed once."""
    h1 = jnp.tanh(params["E"] @ params["W1"] + params["b1"])
    return h1 @ params["W2"] + params["b2"]


def loss_batch(params, prev_ids, targets):
    """Direct-path CE [B] for the actual examples."""
    return stable_ce(forward_ids(params, prev_ids), targets)


def loss_from_all512(params, prev_ids, targets):
    """Compressed path: gather the rows of the all-512 logits at the ACTUAL
    sampled previous ids; identical objective to loss_batch (no dropout)."""
    all_logits = forward_all512(params)
    return stable_ce(all_logits[prev_ids], targets)


def sample_pairs(stream, n, key):
    """Draw n actual training pairs: indices 1..len(stream)-1 (prev, target).
    Returns (indices, prev_ids, target_ids)."""
    idx = jax.random.randint(key, (n,), 1, len(stream))
    arr = np.asarray(stream)
    prev = jnp.asarray(arr[np.asarray(idx) - 1], dtype=jnp.int32)
    tgt = jnp.asarray(arr[np.asarray(idx)], dtype=jnp.int32)
    return idx, prev, tgt
