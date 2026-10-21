"""Reusable checkpoint loader (production, roundtrip-tested).

Checkpoint npz leaves follow JAX tree-leaf order of the list-of-dicts param
tree: for layer i, arr_{2*i} = bias, arr_{2*i + 1} = weight.
"""

import jax
import jax.numpy as jnp
import numpy as np

LAYER_SHAPES = [((128,), (34, 128)), ((128,), (128, 128)),
                ((128,), (128, 128)), ((2,), (128, 2))]


def load_params(path):
    """Load a saved parameter checkpoint, verifying shapes and finiteness."""
    ck = np.load(path)
    assert len(ck.files) == 8
    params = []
    for i, (b_shape, w_shape) in enumerate(LAYER_SHAPES):
        b = ck[f"arr_{2 * i}"]
        w = ck[f"arr_{2 * i + 1}"]
        assert b.shape == b_shape, (i, b.shape, b_shape)
        assert w.shape == w_shape, (i, w.shape, w_shape)
        assert np.all(np.isfinite(b)) and np.all(np.isfinite(w))
        params.append({"b": jnp.asarray(b), "w": jnp.asarray(w)})
    return params


def save_params(path, params):
    """Save params in the same tree-leaf order load_params expects."""
    leaves = []
    for p in params:
        leaves.append(np.asarray(p["b"]))
        leaves.append(np.asarray(p["w"]))
    np.savez(path, *leaves)
