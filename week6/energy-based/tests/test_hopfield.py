"""Part 3 tests: classical Hopfield network (production functions only).

Independent NumPy reference implements the same deterministic async rule;
tests compare the JAX implementation against it.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ebm.hopfield import (
    energy,
    hebb_k,
    overlap,
    random_patterns,
    recall,
    spin_step,
    unstable_fraction,
)
from ebm.hopfield import numpy_recall_reference


def ref_hebb(Xi):
    K = np.asarray(Xi, dtype=np.int64).T @ np.asarray(Xi, dtype=np.int64)
    np.fill_diagonal(K, 0)
    return K


def ref_energy(s, K, N):
    return -0.5 * float(s @ K @ s) / N


# ------------------------------------------------------------- Hebb / energy
def test_hebb_symmetric_zero_diagonal():
    key = jax.random.PRNGKey(0)
    Xi = random_patterns(key, P=7, N=11)
    K = np.asarray(hebb_k(Xi))
    assert K.shape == (11, 11)
    np.testing.assert_array_equal(K, K.T)
    assert np.all(np.diag(K) == 0)
    np.testing.assert_array_equal(K, ref_hebb(Xi))


def test_energy_overlap_identity():
    """E(s) = -N/2 sum_mu m_mu^2 + P/2 (sheet eq. 13)."""
    key = jax.random.PRNGKey(1)
    Xi = random_patterns(key, P=5, N=12)
    K = hebb_k(Xi)
    s = np.asarray(random_patterns(key, P=1, N=12)[0])
    E = float(energy(jnp.asarray(s), K))
    m = np.asarray(overlap(Xi, jnp.asarray(s)))
    ref = -12 / 2 * float(np.sum(m ** 2)) + 5 / 2
    assert E == pytest.approx(ref, abs=1e-5)
    assert E == pytest.approx(ref_energy(s, ref_hebb(Xi), 12), abs=1e-5)


# ------------------------------------------------------- update dynamics
def test_spin_step_energy_strictly_decreasing_and_tie_holds():
    key = jax.random.PRNGKey(2)
    Xi = random_patterns(key, P=10, N=30)
    K = hebb_k(Xi)
    s = np.asarray(Xi[0]).copy()
    s[:5] *= -1  # corrupt 5 bits so flips are guaranteed
    h = np.asarray(K @ jnp.asarray(s))  # integer fields
    prev_E = ref_energy(s, ref_hebb(Xi), 30)
    order = np.random.default_rng(0).permutation(30)
    flipped = 0
    for i in order:
        s_new, h_new = spin_step(
            jnp.asarray(s), jnp.asarray(h), K, int(i))
        s_new, h_new = np.asarray(s_new), np.asarray(h_new)
        if h[i] == 0:
            assert np.array_equal(s_new, s)          # tie holds
        E_new = ref_energy(s_new, ref_hebb(Xi), 30)
        if not np.array_equal(s_new, s):
            flipped += 1
            assert E_new < prev_E                    # strictly decreasing
        np.testing.assert_array_equal(h_new, ref_hebb(Xi) @ s_new)  # recompute
        prev_E = E_new
        s, h = s_new, h_new
    assert flipped > 0


def test_incremental_fields_match_recomputed():
    key = jax.random.PRNGKey(3)
    Xi = random_patterns(key, P=8, N=20)
    K = hebb_k(Xi)
    s = np.asarray(random_patterns(key, P=1, N=20)[0])
    h = np.asarray(K @ jnp.asarray(s))
    for i in range(20):
        s, h = spin_step(jnp.asarray(s), jnp.asarray(h), K, i)
        s, h = np.asarray(s), np.asarray(h)
        np.testing.assert_array_equal(h, np.asarray(K @ jnp.asarray(s)))


def test_numpy_reference_agrees_on_same_order():
    """JAX recall and the independent NumPy reference agree when both use
    the same deterministic update order (seeded permutations)."""
    key = jax.random.PRNGKey(4)
    Xi = random_patterns(key, P=6, N=24)
    K = hebb_k(Xi)
    starts = Xi[:3]
    rng = np.random.default_rng(0)
    perms = np.stack([rng.permuted(np.tile(np.arange(24), (3, 1)), axis=1)
                      for _ in range(200)])  # (200, 3, 24)
    out = recall(K, starts, key, max_sweeps=200, perms=perms)
    ref_states = numpy_recall_reference(np.asarray(K), np.asarray(starts), perms)
    np.testing.assert_array_equal(np.asarray(out["states"]), ref_states)


def test_converged_state_has_stable_fields():
    key = jax.random.PRNGKey(5)
    Xi = random_patterns(key, P=5, N=40)
    K = hebb_k(Xi)
    starts = Xi[:5]
    out = recall(K, starts, key, max_sweeps=100)
    assert bool(out["converged"].all())
    states = np.asarray(out["states"])
    for c in range(5):
        h = np.asarray(K @ jnp.asarray(states[c]))
        stable = (states[c] * h >= 0)  # h==0 leaves spin (also >=0)
        # strict: sign(h_i) == s_i whenever h_i != 0
        assert np.all(states[c][h > 0] == 1)
        assert np.all(states[c][h < 0] == -1)
        assert stable.all()


def test_seeded_permutation_reproducibility():
    key = jax.random.PRNGKey(6)
    Xi = random_patterns(key, P=10, N=50)
    K = hebb_k(Xi)
    starts = Xi[:4]
    a = recall(K, starts, key, max_sweeps=50)
    b = recall(K, starts, key, max_sweeps=50)
    np.testing.assert_array_equal(np.asarray(a["states"]), np.asarray(b["states"]))
    np.testing.assert_array_equal(np.asarray(a["sweeps_used"]),
                                  np.asarray(b["sweeps_used"]))


# -------------------------------------------------- unstable-bit fraction
def test_unstable_fraction_range_and_counts_strict_negatives():
    key = jax.random.PRNGKey(7)
    Xi = random_patterns(key, P=9, N=60)
    K = hebb_k(Xi)
    f = float(unstable_fraction(K, Xi))
    assert 0.0 <= f <= 1.0
    # manual count: xi_i * (K xi)_i < 0 over all bits, zeros excluded
    H = np.asarray(K @ Xi.T.astype(jnp.float32))  # (N, P)
    Xi_np = np.asarray(Xi).T  # (N, P)
    count = int(np.sum(Xi_np * H < 0))
    total = Xi_np.size
    assert f == pytest.approx(count / total)


def test_negative_control_wrong_patterns_near_half():
    """Couplings from OTHER random patterns: unstable fraction ~ 0.5."""
    key = jax.random.PRNGKey(8)
    Xi = random_patterns(key, P=40, N=400)
    Xi_wrong = random_patterns(jax.random.PRNGKey(999), P=40, N=400)
    f = float(unstable_fraction(hebb_k(Xi_wrong), Xi))
    assert 0.4 < f < 0.6


def test_negative_control_nonzero_diagonal_suppresses():
    """Keeping the diagonal self-coupling pushes bits away from instability."""
    key = jax.random.PRNGKey(9)
    Xi = random_patterns(key, P=40, N=400)
    K0 = np.asarray(hebb_k(Xi))
    Kdiag = K0 + np.eye(400, dtype=np.int64) * 40  # add P to the diagonal
    f0 = float(unstable_fraction(jnp.asarray(K0), Xi))
    fd = float(unstable_fraction(jnp.asarray(Kdiag), Xi))
    assert fd < f0
    assert fd < 0.1 * max(f0, 1e-12) or fd < f0 / 2


# --------------------------------------------- single-pattern retrieval
def test_single_pattern_single_bit_retrieval():
    """N=4, P=1: one corrupted bit of the stored pattern is restored by
    async zero-temperature updates (deterministic known case)."""
    key = jax.random.PRNGKey(10)
    Xi = random_patterns(key, P=1, N=4)
    K = hebb_k(Xi)
    corrupted = np.asarray(Xi[0]).copy()
    corrupted[2] *= -1
    out = recall(K, jnp.asarray(corrupted)[None], key, max_sweeps=10)
    assert float(overlap(Xi, out["states"][0]).sum()) == pytest.approx(1.0)
    assert bool(out["converged"][0])
