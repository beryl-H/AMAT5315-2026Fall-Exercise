"""Self-consistency eval formula tests (oracle fixtures, not sample outcomes).

z = mean_i(D_i) / (std_i(D_i, ddof=1) / sqrt(n)) where
D_i = mean_t(surprise_it - entropy_it) per sequence; SE is over SEQUENCES."""
import numpy as np
import pytest

from src.selfconsistency import entropy_zscore


def test_zscore_closed_form_oracle_fixture():
    n = 128
    rng = np.random.default_rng(0)
    D = rng.normal(0.0, 0.3, size=n)  # per-seq mean differences
    entropies = np.ones((n, 1))
    surprises = D[:, None] + 1.0  # surprise - entropy = D per sequence
    z = entropy_zscore(surprises, entropies)
    expected = D.mean() / (D.std(ddof=1) / np.sqrt(n))
    assert z == pytest.approx(expected, rel=1e-12)


def test_uniform_draws_z_near_zero_and_gate_bounds():
    # oracle: token uniform over 64, model knows it -> surprise = entropy = ln 64
    n, T, V = 128, 32, 64
    rng = np.random.default_rng(1)
    surprises = np.full((n, T), np.log(V)) + rng.normal(0, 1e-9, (n, T))
    entropies = np.full((n, T), np.log(V))
    z = entropy_zscore(surprises, entropies)
    assert abs(z) <= 4  # gate at T=1
    # sharpened sampler scored by the same model: surprise systematically LOWER
    surprises_cool = np.full((n, T), np.log(V) - 2.0)
    z2 = entropy_zscore(surprises_cool, entropies)
    assert z2 < -40  # gate at T=0.7 (magnitude > 40)
