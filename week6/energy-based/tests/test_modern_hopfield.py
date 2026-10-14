"""Part 4 tests: modern Hopfield update vs JAX dot-product attention.

All tests call the production functions; the NumPy reference arithmetic is
independent of the implementation under test.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ebm.modern_hopfield import (
    energy_modern,
    modern_update,
    nearest_pattern,
    sign_overlap,
    spinize,
)
from ebm.hopfield import random_patterns

# independent reference implementation
def ref_softmax(u):
    u = np.asarray(u, dtype=np.float64)
    e = np.exp(u - u.max())
    return e / e.sum()


def ref_modern_update(X, x, beta):
    X = np.asarray(X, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    w = ref_softmax(beta * (X.T @ x))
    return X @ w


def ref_energy(X, x, beta):
    X = np.asarray(X, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    u = beta * (X.T @ x)
    lse = np.log(np.exp(u - u.max()).sum()) + u.max()
    return -lse / beta + 0.5 * float(x @ x)


# ---------------------------------------------------------------- basics
def test_spinize_orientation():
    """X columns are stored patterns; binarize + spin map is correct."""
    imgs = np.zeros((2, 28, 28), dtype=np.uint8)
    imgs[0, 0, 0] = 128  # -> v=1 -> s=+1
    imgs[1, 27, 27] = 127  # -> v=0 -> s=-1
    X = spinize(imgs)  # (784, 2)
    assert X.shape == (784, 2)
    assert X[0, 0] == 1 and X[-1, 1] == -1


def test_modern_update_matches_numpy_reference():
    key = jax.random.PRNGKey(0)
    X = random_patterns(key, P=7, N=30).T.astype(jnp.float32)  # (N, P)
    x = random_patterns(key, P=1, N=30)[0].astype(jnp.float32)
    for beta in (0.01, 0.3, 1.0):
        out = np.asarray(modern_update(X, x, beta))
        np.testing.assert_allclose(out, ref_modern_update(X, x, beta),
                                   rtol=1e-5, atol=1e-6)


def test_attention_equality_with_explicit_scale():
    """jax.nn.dot_product_attention with scale=beta equals the modern
    update; the default 1/sqrt(D) scale demonstrably fails."""
    key = jax.random.PRNGKey(1)
    X = random_patterns(key, P=11, N=24).T.astype(jnp.float32)
    x = random_patterns(key, P=1, N=24)[0].astype(jnp.float32)
    for beta in (0.003, 0.03, 0.3):
        out = modern_update(X, x, beta)
        att = jax.nn.dot_product_attention(
            x[None, None, :], X.T[:, None, :], X.T[:, None, :], scale=beta)[0, 0]
        assert float(jnp.max(jnp.abs(out - att))) < 1e-5
    att_default = jax.nn.dot_product_attention(
        x[None, None, :], X.T[:, None, :], X.T[:, None, :])[0, 0]
    # negative control: with the default 1/sqrt(D) scale instead of beta=1
    # the retrieval differs substantially (near-uniform vs concentrated)
    diff_wrong = float(jnp.max(jnp.abs(modern_update(X, x, 1.0)
                                       - att_default)))
    assert diff_wrong > 0.1


def test_energy_nonincreasing_under_update():
    """One modern update never raises E(x) (Ramsauer property)."""
    key = jax.random.PRNGKey(2)
    X = random_patterns(key, P=9, N=40).T.astype(jnp.float32)
    x = random_patterns(key, P=1, N=40)[0].astype(jnp.float32)
    for beta in (0.01, 0.1, 0.3, 1.0):
        E0 = ref_energy(X, x, beta)
        x1 = np.asarray(modern_update(X, x, beta))
        E1 = ref_energy(X, x1, beta)
        assert E1 <= E0 + 1e-5
        assert float(energy_modern(X, x, beta)) == pytest.approx(E0, abs=1e-4)
        x2 = np.asarray(modern_update(X, x, beta))
        assert float(energy_modern(X, x2, beta)) <= \
            float(energy_modern(X, x1, beta)) + 1e-4


# ------------------------------------------------------------ criterion
def test_sign_overlap_and_tie_policy():
    """sign(0)=0 contributes zero to the overlap (fixed policy)."""
    xi = np.array([1, -1, 1, 1], dtype=np.int32)
    s = np.array([1, 1, 0, -1], dtype=np.int32)  # match, mismatch, tie, mismatch
    # (matches - mismatches + 0*ties)/N = (1 - 2 + 0)/4 = -0.25
    assert float(sign_overlap(xi, s)) == pytest.approx(-0.25)
    assert float(sign_overlap(xi, xi)) == pytest.approx(1.0)


def test_nearest_pattern_first_index_tie():
    # columns are patterns: col0 = col1 = (1,-1,1), col2 different
    X = np.array([[1, 1, -1], [-1, -1, 1], [1, 1, -1]], dtype=np.int32)
    s = np.array([1, -1, 1], dtype=np.int32)
    # patterns 0 and 1 tie at overlap 1.0 -> first index wins
    assert nearest_pattern(X, s) == 0


def test_high_overlap_but_wrong_nearest_fails():
    """Success requires the target to be the NEAREST pattern, not just
    overlap >= 0.95 (a near-duplicate can be nearer)."""
    N = 100
    target = np.ones(N, dtype=np.int32)
    near_dup = target.copy(); near_dup[-1] = -1   # 1 bit differs from target
    other = -target
    X = np.stack([target, near_dup, other], axis=1).astype(np.float32)
    s = target.copy(); s[-2:] = -1                # retrieved sign vector
    assert float(sign_overlap(target, s)) >= 0.95  # overlap criterion met
    assert nearest_pattern(X, s) == 1              # but nearest is the dup
    # so success must be False under the full criterion
    from ebm.modern_hopfield import retrieval_success
    assert not retrieval_success(X, target_idx=0, s=s)


def test_retrieval_success_true_case():
    from ebm.modern_hopfield import retrieval_success
    rng = np.random.default_rng(0)
    X = rng.choice([-1, 1], size=(10, 3)).astype(np.float32)
    assert retrieval_success(X, target_idx=1, s=X[:, 1])
    assert not retrieval_success(X, target_idx=1, s=X[:, 2])
