"""Part 5 transformer tests: parameter count 328832, causal invariance with a
masked-prefix paired-input control (and its unmasked negative control), strict
upper-triangular attention zeros, shapes/dtypes, frozen hyperparameters."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.transformer_model import (
    TR_HPARAMS,
    forward,
    init_transformer,
    num_params,
)


def test_param_count_328832_and_f32():
    params = init_transformer(jax.random.PRNGKey(0))
    assert num_params(params) == 328832
    assert params["tok_emb"].shape == (512, 96)
    assert params["pos_emb"].shape == (64, 96)
    assert len(params["blocks"]) == 2
    blk = params["blocks"][0]
    for k in ["ln1_g", "ln1_b", "Wq", "bq", "Wk", "bk", "Wv", "bv",
              "Wo", "bo", "ln2_g", "ln2_b", "W1", "b1", "W2", "b2"]:
        assert k in blk, k
    for x in jax.tree_util.tree_leaves(params):
        assert x.dtype == jnp.float32


def test_frozen_hparams():
    assert TR_HPARAMS == {"steps": 6000, "batch": 32, "ctx": 64, "peak_lr": 0.003,
                          "warmup": 300, "end_frac": 0.1, "betas": (0.9, 0.999),
                          "wd": 0.01, "seed": 0, "width": 96, "heads": 4,
                          "mlp_hidden": 384, "clip": 1.0, "layers": 2}


def test_forward_shapes_finite_and_attn_mask():
    params = init_transformer(jax.random.PRNGKey(1))
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 512, size=(2, 64)))
    logits, attns = forward(params, ids, return_attn=True)
    assert logits.shape == (2, 64, 512)
    assert np.isfinite(np.asarray(logits)).all()
    assert len(attns) == 2
    for a in attns:
        assert a.shape == (2, 4, 64, 64)
        upper = a[..., 0, :]  # placeholder to keep shape ops honest
        # strict causal zeros above the diagonal
        iu = np.triu_indices(64, k=1)
        assert np.abs(np.asarray(a)[..., iu[0], iu[1]]).max() == 0.0
        # rows sum to one
        assert np.allclose(np.asarray(a).sum(-1), 1.0, atol=1e-5)


def test_causal_prefix_invariance_paired_inputs():
    params = init_transformer(jax.random.PRNGKey(2))
    rng = np.random.default_rng(3)
    base = rng.integers(0, 512, size=(8, 64))
    alter = base.copy()
    alter[:, 32:] = rng.integers(0, 512, size=(8, 32))  # differ only after 32
    base_j, alter_j = jnp.asarray(base), jnp.asarray(alter)
    l_base, attn_b = forward(params, base_j, return_attn=True)
    l_alt, attn_a = forward(params, alter_j, return_attn=True)
    # masked prefix logits unchanged (positions 0..31)
    assert np.allclose(np.asarray(l_base[:, :32]), np.asarray(l_alt[:, :32]),
                       atol=1e-4), "prefix logits changed under causal mask"
    # suffix logits generally differ
    assert not np.allclose(np.asarray(l_base[:, 32:]), np.asarray(l_alt[:, 32:]),
                           atol=1e-4)


def test_unmasked_negative_control_breaks_prefix_independence():
    params = init_transformer(jax.random.PRNGKey(4))
    rng = np.random.default_rng(5)
    base = rng.integers(0, 512, size=(4, 64))
    alter = base.copy()
    alter[:, 32:] = rng.integers(0, 512, size=(4, 32))
    l_base, _ = forward(params, jnp.asarray(base), causal=False)
    l_alt, _ = forward(params, jnp.asarray(alter), causal=False)
    assert not np.allclose(np.asarray(l_base[:, :32]), np.asarray(l_alt[:, :32]),
                           atol=1e-4), "unmasked control should leak future info"


def test_loss_axis_correctness():
    params = init_transformer(jax.random.PRNGKey(6))
    ids = jnp.full((2, 64), 7, dtype=jnp.int32)
    logits, _ = forward(params, ids)
    targets = jnp.full((2, 64), 7, dtype=jnp.int32)
    from src.nn_common import stable_ce

    ce = stable_ce(logits, targets)
    assert ce.shape == (2, 64)
    # repeated token 7 -> model can in principle learn it; here just finite
    assert np.isfinite(np.asarray(ce)).all()
