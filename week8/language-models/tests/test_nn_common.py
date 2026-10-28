"""Shared NN kernel tests: algebraically stable cross-entropy, AdamW state
mechanics, LR schedule. f64 equivalence tests isolated to this module."""
import jax
jax.config.update("jax_enable_x64", True)  # isolated unit-check config
import jax.numpy as jnp
import numpy as np
import pytest

from src.nn_common import LR_SCHEDULES, adamw_step, schedule_lr, stable_ce


def test_stable_ce_equals_naive_and_high_confidence():
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(7, 16)) * 3.0)
    targets = jnp.asarray(rng.integers(0, 16, size=7))
    got = stable_ce(logits, targets)
    naive = jnp.log(jnp.sum(jnp.exp(logits), axis=1)) - logits[np.arange(7), targets]
    assert np.allclose(got, naive, atol=1e-10)
    # equal logits -> ln V
    logits_eq = jnp.zeros((1, 512))
    assert np.isclose(stable_ce(logits_eq, jnp.array([0]))[0].item(), np.log(512))
    # high-confidence case (+20 on target) -> near zero, still finite
    big = jnp.zeros((1, 8)).at[0, 3].set(20.0)
    ce = stable_ce(big, jnp.array([3]))[0].item()
    assert 0.0 <= ce < 1e-6


def test_stable_ce_gradient_sum_zero_and_target_complement_nonzero():
    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.normal(size=(1, 8)))
    targets = jnp.array([2])
    g = jax.grad(lambda z: stable_ce(z, targets).sum())(logits)
    assert np.isclose(g.sum().item(), 0.0, atol=1e-12)  # softmax - onehot
    assert g[0, 2].item() < 0.0  # target gets negative gradient
    assert (g[0] - g[0, 2]).max().item() > 0  # complement nonzero
    # +20 logits: gradient stays finite and correct
    big = jnp.zeros((1, 8)).at[0, 3].set(20.0)
    g2 = jax.grad(lambda z: stable_ce(z, jnp.array([3])).sum())(big)
    p = jax.nn.softmax(big)[0]
    assert np.allclose(np.asarray(g2[0]), np.asarray(p - jax.nn.one_hot(3, 8)), atol=1e-9)


def test_stable_ce_batch_time_axes():
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(size=(2, 5, 9)))
    targets = jnp.asarray(rng.integers(0, 9, size=(2, 5)))
    got = stable_ce(logits, targets)
    assert got.shape == (2, 5)
    for b in range(2):
        for t in range(5):
            row = logits[b, t]
            naive = np.log(np.exp(row).sum()) - row[targets[b, t]]
            assert np.isclose(float(got[b, t]), naive, atol=1e-10)


def test_adamw_single_step_matches_direct_numpy():
    rng = np.random.default_rng(3)
    params = {"w": jnp.asarray(rng.normal(size=(4, 3))), "b": jnp.asarray(rng.normal(size=(4,)))}
    grads = {"w": jnp.asarray(rng.normal(size=(4, 3))), "b": jnp.asarray(rng.normal(size=(4,)))}
    state = {"m": {k: jnp.zeros_like(v) for k, v in params.items()},
             "v": {k: jnp.zeros_like(v) for k, v in params.items()}, "step": 0}
    lr, b1, b2, wd, eps = 0.01, 0.9, 0.999, 0.01, 1e-8
    new_p, new_s = adamw_step(params, state, grads, lr=lr, betas=(b1, b2),
                              wd=wd, eps=eps)
    # direct numpy reference, step count starts at 1 (bias correction)
    m_ref = {k: (1 - b1) * np.asarray(g) for k, g in grads.items()}
    v_ref = {k: (1 - b2) * np.asarray(g) ** 2 for k, g in grads.items()}
    t = 1
    out = {}
    for k in params:
        mh = m_ref[k] / (1 - b1 ** t)
        vh = v_ref[k] / (1 - b2 ** t)
        upd = mh / (np.sqrt(vh) + eps)
        if np.asarray(params[k]).ndim == 2:  # wd on 2D weights only
            upd = upd + wd * np.asarray(params[k])
        out[k] = np.asarray(params[k]) - lr * upd
    assert np.allclose(np.asarray(new_p["w"]), out["w"], atol=1e-12)
    assert np.allclose(np.asarray(new_p["b"]), out["b"], atol=1e-12)
    assert new_s["step"] == 1
    assert np.allclose(np.asarray(new_s["m"]["w"]), m_ref["w"])
    # no hidden reset: a second step continues from state
    p2, s2 = adamw_step(new_p, new_s, grads, lr=lr, betas=(b1, b2), wd=wd, eps=eps)
    assert s2["step"] == 2


def test_schedule_warmup_cosine_endpoints():
    f = schedule_lr(0, 100, 2000, peak=6e-3, warmup=200, end_frac=0.1)
    assert f == pytest.approx(0.0)
    assert schedule_lr(200, 100, 2000, peak=6e-3, warmup=200, end_frac=0.1) \
        == pytest.approx(6e-3)
    assert schedule_lr(2000, 100, 2000, peak=6e-3, warmup=200, end_frac=0.1) \
        == pytest.approx(0.6e-3)
    # monotone rise in warmup, monotone fall after
    a = schedule_lr(100, 100, 2000, peak=1.0, warmup=200, end_frac=0.1)
    b = schedule_lr(150, 100, 2000, peak=1.0, warmup=200, end_frac=0.1)
    c = schedule_lr(1000, 100, 2000, peak=1.0, warmup=200, end_frac=0.1)
    d = schedule_lr(1500, 100, 2000, peak=1.0, warmup=200, end_frac=0.1)
    assert a < b and c > d


def test_frozen_schedule_params():
    assert LR_SCHEDULES["mlp"] == {"steps": 4000, "peak": 0.006, "warmup": 200, "end_frac": 0.1}
    assert LR_SCHEDULES["transformer"] == {"steps": 6000, "peak": 0.003, "warmup": 300, "end_frac": 0.1}
