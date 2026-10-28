"""Part 3 MLP next-token model tests: exact 16384-example minibatch objective,
compressed all-512-forward equivalence (loss AND gradient), batch index
freshness, frozen hyperparameters."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.mlp_model import (
    MLP_HPARAMS,
    forward_all512,
    forward_ids,
    init_mlp,
    loss_batch,
    loss_from_all512,
    num_params,
    sample_pairs,
)


def test_param_count_328704_and_f32():
    params = init_mlp(jax.random.PRNGKey(0))
    assert num_params(params) == 328704
    assert params["E"].shape == (512, 64)
    assert params["W1"].shape == (64, 512) and params["b1"].shape == (512,)
    assert params["W2"].shape == (512, 512) and params["b2"].shape == (512,)
    for x in jax.tree_util.tree_leaves(params):
        assert x.dtype == jnp.float32


def test_frozen_hparams():
    assert MLP_HPARAMS == {"steps": 4000, "batch": 16384, "peak_lr": 0.006,
                           "warmup": 200, "end_frac": 0.1, "betas": (0.9, 0.999),
                           "wd": 0.01, "seed": 0, "d_embed": 64, "hidden": 512}


def test_forward_shapes_and_finite():
    params = init_mlp(jax.random.PRNGKey(1))
    ids = jnp.asarray([5, 100, 511])
    logits = forward_ids(params, ids)
    assert logits.shape == (3, 512)
    assert np.isfinite(np.asarray(logits)).all()
    all512 = forward_all512(params)
    assert all512.shape == (512, 512)
    # gather consistency: same id -> same row
    assert np.allclose(np.asarray(logits[0]), np.asarray(all512[5]), atol=1e-6)


def test_exact_minibatch_loss_and_gradient_equivalence():
    params = init_mlp(jax.random.PRNGKey(2))
    rng = np.random.default_rng(0)
    prev = jnp.asarray(rng.integers(0, 512, size=97))
    tgt = jnp.asarray(rng.integers(0, 512, size=97))

    def loss_direct(p):
        return loss_batch(p, prev, tgt).sum()

    def loss_compressed(p):
        return loss_from_all512(p, prev, tgt).sum()

    ld, gd = loss_direct(params), jax.grad(loss_direct)(params)
    lc, gc = loss_compressed(params), jax.grad(loss_compressed)(params)
    assert np.isclose(float(ld), float(lc), rtol=1e-5)
    for a, b in zip(jax.tree_util.tree_leaves(gd), jax.tree_util.tree_leaves(gc)):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=1e-6)


def test_repeated_input_matches_lookup_row():
    params = init_mlp(jax.random.PRNGKey(3))
    prev = jnp.full((16,), 123, dtype=jnp.int32)
    tgt = jnp.asarray(np.random.default_rng(1).integers(0, 512, size=16))
    # repeated-input direct forward equals the all-512 row for id 123
    l_direct = loss_batch(params, prev, tgt)
    l_lookup = loss_from_all512(params, prev, tgt)
    assert np.allclose(np.asarray(l_direct), np.asarray(l_lookup), rtol=1e-5)


def test_batch_indices_fresh_each_step():
    stream = list(range(1000))  # toy stream (values as ids)
    i1, pv1, tg1 = sample_pairs(stream, 64, jax.random.PRNGKey(10))
    i2, pv2, tg2 = sample_pairs(stream, 64, jax.random.PRNGKey(11))
    assert not np.array_equal(np.asarray(i1), np.asarray(i2))
    assert i1.min() >= 1 and i1.max() <= len(stream) - 1  # actual pairs 1..N-1
    assert np.array_equal(np.asarray(pv1), np.asarray(stream)[np.asarray(i1) - 1])
    assert np.array_equal(np.asarray(tg1), np.asarray(stream)[np.asarray(i1)])
