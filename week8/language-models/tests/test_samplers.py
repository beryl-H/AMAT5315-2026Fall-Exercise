"""Sampler tests (pure NumPy, f64 target distributions on stored f32 logits).

Samplers: temperature, top-k (k largest), top-p (smallest set with cumulative
>= p INCLUDING the crossing token), min-p (p_i >= minp * pmax). The chi2
helper pools rare bins (expected >= 5, merge rule) and reports support
violations instead of nan-away. Negative control: a top-p sampler that drops
the crossing token must be rejected against the correct target distribution."""
import numpy as np
import pytest

from src.sampling import (
    chi2_pooled,
    min_p_dist,
    sample_dist,
    temperature_dist,
    top_k_dist,
    top_p_dist,
    top_p_dist_wrong_negative_control,
)


def _logits_fixture():
    # distinctive ordering with a near-threshold crossing token
    return np.array([2.0, 1.5, 1.0, 0.9, 0.8, 0.0, -1.0, -2.0, -3.0, -10.0])


def test_temperature_dist_sums_and_sharpens():
    z = _logits_fixture()
    d1 = temperature_dist(z, 1.0)
    d05 = temperature_dist(z, 0.5)
    assert np.isclose(d1.sum(), 1.0) and (d1 >= 0).all()
    assert d05.max() > d1.max()  # sharper
    assert np.isclose(d05.sum(), 1.0)


def test_top_k_keeps_exactly_k_largest():
    z = _logits_fixture()
    d = top_k_dist(z, 3, T=0.8)
    keep = np.argsort(-z)[:3]
    assert (d[keep] > 0).all() and d.sum() > 0
    d_rest = d.copy(); d_rest[keep] = 0
    assert d_rest.sum() == 0.0  # exact support


def test_top_p_includes_crossing_token():
    z = _logits_fixture()
    p = 0.7
    d = top_p_dist(z, p, T=1.0)
    probs = np.exp(z - z.max()); probs /= probs.sum()
    order = np.argsort(-z)
    cum = np.cumsum(probs[order])
    # the kept set is the smallest prefix reaching p; verify against cum
    n_kept = int((d > 0).sum())
    assert cum[n_kept - 1] >= p - 1e-12
    if n_kept >= 2:
        assert cum[n_kept - 2] < p - 1e-12  # smallest such set
    assert np.isclose(d.sum(), 1.0)


def test_top_p_wrong_dropper_differs_and_is_rejected_by_chi2():
    z = _logits_fixture()
    correct = top_p_dist(z, 0.7, T=1.0)
    wrong = top_p_dist_wrong_negative_control(z, 0.7, T=1.0)
    assert not np.allclose(correct, wrong)
    # 100k draws from the WRONG sampler vs the CORRECT target -> rejected
    rng = np.random.default_rng(531505)
    draws = rng.choice(len(z), size=100_000, p=wrong)
    stat, df, p, meta = chi2_pooled(correct, draws, emin=5.0)
    assert meta.get("support_violation") is False
    assert p < 1e-6  # rejected
    # draws from the CORRECT sampler pass
    draws_ok = rng.choice(len(z), size=100_000, p=correct)
    stat2, df2, p2, _ = chi2_pooled(correct, draws_ok, emin=5.0)
    assert p2 > 0.001


def test_min_p_relative_to_pmax_at_T07():
    z = _logits_fixture()
    d = min_p_dist(z, 0.1, T=0.7)
    probs = temperature_dist(z, 0.7)
    keep = probs >= 0.1 * probs.max()
    assert ((d > 0) == keep).all()  # exact support rule
    assert np.isclose(d.sum(), 1.0)


def test_temperature_1e4_returns_argmax_always():
    z = _logits_fixture()
    rng = np.random.default_rng(0)
    d = temperature_dist(z, 1e-4)
    u = rng.random(1000)
    draws = sample_dist(d, u)
    assert (draws == int(np.argmax(z))).all()


def test_support_violation_explicit_zero_p_rejection():
    d = np.array([0.5, 0.5, 0.0])  # token 2 impossible
    rng = np.random.default_rng(1)
    draws = rng.choice(2, size=100, p=np.array([0.5, 0.5]))
    draws = np.concatenate([draws, np.full(10, 2)])  # observed mass on impossible
    stat, df, p, meta = chi2_pooled(d, draws, emin=5.0)
    assert meta["support_violation"] is True
    assert p == 0.0  # EXPLICIT rejection, never nan (a nan must not pass gates)
