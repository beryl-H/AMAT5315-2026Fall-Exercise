"""Checkpoint loader roundtrip test against the real run directory."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import network
from src.checkpoints import load_params, save_params

RUN = "artifacts/run1"


def test_loader_roundtrip_against_init_params():
    """init_params(2026) must reproduce checkpoint 00000 exactly."""
    params = network.init_params(jax.random.PRNGKey(2026))
    path = "/tmp/ckpt_roundtrip_test.npz"
    save_params(path, params)
    loaded = load_params(path)
    for p, l in zip(params, loaded):
        np.testing.assert_array_equal(np.asarray(p["w"]), np.asarray(l["w"]))
        np.testing.assert_array_equal(np.asarray(p["b"]), np.asarray(l["b"]))


@pytest.mark.skipif(not os.path.exists(f"{RUN}/checkpoints/params_00000.npz"),
                    reason="real training run not present")
def test_loader_reads_real_run_checkpoints():
    for step in (0, 25000):
        params = load_params(f"{RUN}/checkpoints/params_{step:05d}.npz")
        assert len(params) == 4
        for p in params:
            assert bool(jax.tree_util.tree_all(
                jax.tree_util.tree_map(lambda x: jnp.isfinite(x).all(), p)))


def test_loader_rejects_wrong_shapes(tmp_path):
    bad = "/tmp/ckpt_bad_test.npz"
    np.savez(bad, np.zeros(3), np.zeros((4, 4)), np.zeros(3), np.zeros((4, 4)),
             np.zeros(3), np.zeros((4, 4)), np.zeros(2), np.zeros((4, 2)))
    with pytest.raises(AssertionError):
        load_params(bad)
