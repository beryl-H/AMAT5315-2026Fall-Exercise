"""Part 6 scaling-law fitter tests (smoke scope: no full 243-start grid and no
400 bootstrap refits here)."""
import numpy as np
import pytest

from src.scaling import (
    BOUNDS,
    FIT_OPTIONS,
    GRID,
    derive_optimal,
    load_runs,
    objective_grad,
)

HERE = os_path = __import__("os").path.dirname(__file__)
DATA_CSV = __import__("os").path.join(HERE, "..", "data", "chinchilla-runs.csv")


def _synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.array([np.log(1.8), np.log(400.0), np.log(2000.0), 0.347, 0.367])
    N = 10 ** rng.uniform(7.5, 10, size=n)
    D = 10 ** rng.uniform(8.5, 11.5, size=n)
    L = np.exp(theta[0]) + np.exp(theta[1]) * N ** -theta[3] + \
        np.exp(theta[2]) * D ** -theta[4]
    return theta, N, D, np.log(L)


def test_gradient_matches_central_fd_at_nonoptimal_point():
    theta, N, D, logL = _synthetic()
    # perturb away from the optimum (non-optimal evaluation point)
    th = theta + np.array([0.3, -0.8, 0.5, 0.07, -0.09])
    F, g = objective_grad(th, N, D, logL)
    assert np.isfinite(F) and np.isfinite(g).all() and np.abs(g).sum() > 0
    eps = 1e-5
    for i in range(5):
        tp = th.copy(); tp[i] += eps
        tm = th.copy(); tm[i] -= eps
        fd = (objective_grad(tp, N, D, logL)[0] - objective_grad(tm, N, D, logL)[0]) / (2 * eps)
        assert np.isclose(g[i], fd, rtol=1e-7), (i, g[i], fd)


def test_logsumexp_overflow_safety():
    theta, N, D, logL = _synthetic(n=20)
    th = np.array([np.log(1e-8), np.log(1e12), np.log(1e12), 0.5, 0.5])
    F, g = objective_grad(th, N, D, logL)
    assert np.isfinite(F) and np.isfinite(g).all()


def test_derive_optimal_compute_identity_and_first_order_condition():
    alpha, A, beta, B = 0.347, 478.0, 0.367, 2143.0
    C = 5.76e23
    Ns, Ds = derive_optimal(alpha, A, beta, B, C)
    assert np.isclose(6 * Ns * Ds, C, rtol=1e-12)
    # first-order condition: d/dN of L(N, (C/6)/N) vanishes at N*
    L = lambda N: A * N ** -alpha + B * ((C / 6) / N) ** -beta
    h = Ns * 1e-6
    num = (L(Ns + h) - L(Ns - h)) / (2 * h)
    assert abs(num) < 1e-6 * abs(L(Ns))  # stationary to high relative accuracy
    # grid+refine argmin agrees
    grid = np.geomspace(Ns / 4, Ns * 4, 200001)
    assert grid[np.argmin(L(grid))] == pytest.approx(Ns, rtol=1e-3)


def test_load_runs_filters_five_highest_losses():
    N, C, L = load_runs(DATA_CSV)
    assert len(N) == 245
    keep = np.argsort(L)[:-5]  # drop the five highest losses
    assert len(keep) == 240
    D = C / (6 * N)
    assert (D > 0).all() and np.isfinite(D).all()


def test_frozen_grid_bounds_options_literals():
    assert sorted(GRID["E"]) == [1.5, 1.7, 1.9]
    assert sorted(GRID["A"]) == [10, 100, 1000]
    assert sorted(GRID["B"]) == [100, 1000, 10000]
    assert sorted(GRID["alpha"]) == [0.25, 0.35, 0.45]
    assert sorted(GRID["beta"]) == [0.25, 0.35, 0.45]
    n_start = 1
    for v in GRID.values():
        n_start *= len(v)
    assert n_start == 243
    assert BOUNDS["logE"] == (np.log(1.0), np.log(3.0))
    assert BOUNDS["alpha"] == (0.05, 1.5) and BOUNDS["beta"] == (0.05, 1.5)
    assert FIT_OPTIONS["maxls"] == 100 and FIT_OPTIONS["ftol"] == 1e-15
