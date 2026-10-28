"""Key-value memory stores (SPEC Parts 3 and 4 conventions).

ISR convention: R_j = ||f(k_j) - c_j v_j|| / (|c_j| * ||v_j||) with own
coefficient c_j (LS store: c_j = h_jj, diagonal of H = K pinv(K); outer/ReLU
stores: c_j = sigma(k_j . k_j) = 1).
"""
import numpy as np


def ls_recall(K: np.ndarray, V: np.ndarray):
    """Least-squares store. Returns (recall Mxd, h_diag M,).

    Uses the SVD-based pseudoinverse K^+ (d x M) so no M x M matrix is ever
    formed: recall = K @ (K^+ @ V), h_jj = sum_a K[j,a] * K^+[a,j].
    For M <= d with full row rank, H = I and R = 0 exactly.
    """
    Kpinv = np.linalg.pinv(K)  # d x M
    recall = K @ (Kpinv @ V)  # M x d, never M x M
    h_diag = np.einsum("ja,aj->j", K, Kpinv)
    return recall, h_diag


def ls_R(K, V, recall, h_diag) -> np.ndarray:
    own = h_diag[:, None] * V
    return np.linalg.norm(recall - own, axis=1) / (
        np.abs(h_diag) * np.linalg.norm(V, axis=1)
    )


def naive_ls_R_forbidden_reference(K, V) -> np.ndarray:
    """The WRONG metric (||recall - v_j||/||v_j||), kept only so tests can
    demonstrate it fails the mandatory d=1, M=2 regression case."""
    recall, _ = ls_recall(K, V)
    return np.linalg.norm(recall - V, axis=1) / np.linalg.norm(V, axis=1)


def _chunked_scores(K, x):
    return K @ x


def outer_recall(K, V, x) -> np.ndarray:
    return V.T @ (K @ x)


def relu_recall(K, V, x, p: int) -> np.ndarray:
    return V.T @ np.maximum(K @ x, 0.0) ** p


def picture_R(recall, w_own, picture) -> float:
    """Part 4 ISR: ||recall - w_own*picture|| / (|w_own| * ||picture||).
    The 1024-pixel +-(1) picture has norm exactly 32."""
    return float(
        np.linalg.norm(recall - w_own * picture)
        / (abs(w_own) * np.linalg.norm(picture))
    )
