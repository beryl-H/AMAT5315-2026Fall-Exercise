"""Memory-store ISR tests — mandatory regression cases from the corrected SPEC.

LS store: H = K pinv(K), own coefficient h_jj, R = ||recall - h_jj v_j|| /
(|h_jj| ||v_j||). The naive ||recall - v_j||/||v_j|| metric is forbidden.
Part 4: R = ||recall - w_own * picture|| / (|w_own| * ||picture||) with the
1024-pixel +-(1) picture norm exactly 32.
"""
import numpy as np
import pytest

from src.memory import (
    ls_recall,
    ls_R,
    naive_ls_R_forbidden_reference,
    outer_recall,
    picture_R,
    relu_recall,
)


def test_ls_regression_d1_m2_R_equals_one():
    K = np.array([[1.0], [1.0]])
    V = np.array([[1.0], [1.0]])
    recall, h_diag = ls_recall(K, V)
    assert np.allclose(recall, [[1.0], [1.0]])
    assert np.allclose(h_diag, 0.5)
    R = ls_R(K, V, recall, h_diag)
    # own = 0.5 * 1, interference = 0.5 -> R = 0.5 / (0.5 * 1) = 1
    assert np.allclose(R, 1.0)
    # the forbidden naive metric gives 0 here (wrong capacity)
    assert np.allclose(naive_ls_R_forbidden_reference(K, V), 0.0)


def test_ls_perfect_recall_below_capacity():
    rng = np.random.default_rng(101)
    d, M = 8, 4
    K = rng.normal(size=(M, d))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.normal(size=(M, d))
    recall, h_diag = ls_recall(K, V)
    R = ls_R(K, V, recall, h_diag)
    assert np.max(np.abs(R)) < 1e-10  # sheet: R < 1e-10 up to M = d


def test_ls_hdiag_matches_explicit_H_beyond_capacity():
    rng = np.random.default_rng(102)
    d, M = 4, 10
    K = rng.normal(size=(M, d))
    V = rng.normal(size=(M, d))
    recall, h_diag = ls_recall(K, V)
    H = K @ np.linalg.pinv(K)  # explicit M x M, allowed only in the test
    assert np.allclose(h_diag, np.diag(H))
    assert np.allclose(recall, H @ V)


def test_outer_and_relu_recall_own_coefficient_is_one():
    rng = np.random.default_rng(103)
    d, M = 8, 6
    K = rng.normal(size=(M, d))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.normal(size=(M, d))
    x = K[0]
    f_outer = outer_recall(K, V, x)
    f_relu = relu_recall(K, V, x, 1)
    assert np.allclose(f_outer, V.T @ (K @ x))
    assert np.allclose(f_relu, V.T @ np.maximum(K @ x, 0.0))
    # with orthonormal keys both stores recall v_0 exactly (R = 0)
    Q, _ = np.linalg.qr(rng.normal(size=(d, M)))
    Kort = Q.T
    assert np.allclose(outer_recall(Kort, V, Kort[0]), V[0])
    assert np.allclose(relu_recall(Kort, V, Kort[0], 3), V[0])


def test_picture_R_uses_norm_32_denominator():
    p = np.ones(1024)
    p[::2] = -1.0
    assert np.linalg.norm(p) == pytest.approx(32.0)
    w_own = 0.7
    leak = np.zeros(1024)
    leak[0] = abs(w_own) * 32.0  # ||leak|| = |w_own| * ||p|| -> R = 1
    recall = w_own * p + leak
    R = picture_R(recall, w_own, p)
    assert R == pytest.approx(1.0)  # factor-32 error would break this
