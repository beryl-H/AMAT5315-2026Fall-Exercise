"""Unbiased energy-distance estimator (Székely–Rizzo form, distinct pairs).

D = 2*mean_cross - mean_within_X - mean_within_Y, with within-means over
ordered pairs i != j (zero diagonal excluded):
    2*sum_cross/(n m) - sum_withinX/(n (n-1)) - sum_withinY/(m (m-1)).
The unbiased estimate may be negative; raw values are kept.
"""

import numpy as np


def _within_sum(D):
    return float(D.sum() - np.trace(D))


def energy_distance_unbiased(X, Y):
    """X: (n, d), Y: (m, d) float arrays. Unbiased energy distance."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        raise ValueError("need at least 2 samples per set")
    from scipy.spatial.distance import cdist
    Dxy = cdist(X, Y)
    Dxx = cdist(X, X)
    Dyy = cdist(Y, Y)
    cross = float(Dxy.sum())
    wx = _within_sum(Dxx)
    wy = _within_sum(Dyy)
    return 2.0 * cross / (n * m) - wx / (n * (n - 1)) - wy / (m * (m - 1))
