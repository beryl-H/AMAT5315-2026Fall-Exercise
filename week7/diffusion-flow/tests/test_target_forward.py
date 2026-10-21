"""Task 1 tests: target sampling and forward diffusion (no network, no reverse).

Derived from the lecture sheet, Part 1: mixture target, beta schedule,
alpha/abar coefficients and direct noising (Equation 7).
"""

import math

import jax
import jax.numpy as jnp
import pytest

from src.diffusion import (
    T_LEVELS,
    abar_from_betas,
    beta_schedule,
    forward_step,
    noise_to_level,
    sample_target,
)

CENTRES = [(3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0)]


# ---------- schedule coefficients, hand derived ----------

def test_beta_schedule_endpoints_and_length():
    betas = beta_schedule()
    assert betas.shape == (T_LEVELS,)
    # beta_1 = 1e-4, beta_1000 = 0.02, linear in between
    assert float(betas[0]) == pytest.approx(1e-4, abs=1e-8)
    assert float(betas[-1]) == pytest.approx(0.02, abs=1e-8)
    # strictly increasing
    assert bool(jnp.all(betas[1:] > betas[:-1]))


def test_beta_schedule_linear_midpoint():
    betas = beta_schedule()
    # Linear schedule: beta_500 (index 499) = 1e-4 + (0.02-1e-4) * 499/999
    expected = 1e-4 + (0.02 - 1e-4) * 499 / 999
    assert float(betas[499]) == pytest.approx(expected, abs=1e-8)


def test_alpha_and_abar_hand_derived():
    betas = beta_schedule()
    alphas = 1.0 - betas
    abar = abar_from_betas(betas)
    assert abar.shape == (T_LEVELS + 1,)  # abar_0 ... abar_1000
    assert float(abar[0]) == pytest.approx(1.0, abs=1e-12)
    assert float(abar[1]) == pytest.approx(1.0 - 1e-4, abs=1e-7)  # float32
    assert float(abar[2]) == pytest.approx((1.0 - 1e-4) * alphas[1], abs=1e-7)
    # abar_1000 is approximately 4e-5 per the lecture
    assert 3.0e-5 < float(abar[1000]) < 5.5e-5
    # abar strictly decreasing and abar_t <= alpha_t for t>=1
    assert bool(jnp.all(abar[1:] < abar[:-1]))
    assert bool(jnp.all(abar[1:] <= alphas + 1e-7))


# ---------- target sampling statistics ----------

def test_target_shape_and_dtype():
    x = sample_target(jax.random.PRNGKey(0), 1000)
    assert x.shape == (1000, 2)
    assert x.dtype == jnp.float32


def test_target_cluster_statistics():
    """Large sample: four equal clusters, exact centres, std 0.3."""
    n = 40000
    x = sample_target(jax.random.PRNGKey(1), n)
    x = jax.device_get(x)
    d = jnp.linalg.norm(x[:, None, :] - jnp.array(CENTRES)[None], axis=-1)
    assign = jnp.argmin(d, axis=1)
    for k, c in enumerate(CENTRES):
        pts = x[assign == k]
        frac = len(pts) / n
        assert abs(frac - 0.25) < 0.02, f"cluster {k} fraction {frac}"
        for i in range(2):
            assert abs(float(pts[:, i].mean()) - c[i]) < 0.05
            assert abs(float(pts[:, i].std()) - 0.3) < 0.02


def test_target_streams_independent():
    """Different keys give independent draws; same key reproduces exactly."""
    a1 = sample_target(jax.random.PRNGKey(7), 8)
    a2 = sample_target(jax.random.PRNGKey(7), 8)
    b = sample_target(jax.random.PRNGKey(8), 8)
    assert bool(jnp.all(a1 == a2))
    assert bool(jnp.all(a1 != b))


# ---------- forward diffusion ----------

def test_forward_step_moments():
    """One forward step: x1 = sqrt(a) x0 + sqrt(b) z has the right moments."""
    beta = 0.25
    a = 1.0 - beta
    n = 200_000
    key = jax.random.PRNGKey(3)
    x0 = jnp.full((n, 2), 3.0, dtype=jnp.float32)  # constant clean point
    x1 = forward_step(x0, a, beta, key)
    x1 = jax.device_get(x1)
    # mean should be sqrt(a) * 3
    assert abs(float(x1[:, 0].mean()) - math.sqrt(a) * 3.0) < 0.01
    # variance should be a*0 + beta = beta
    assert abs(float(x1[:, 0].var()) - beta) < 0.01
    assert x1.dtype == jnp.float32


def test_noise_to_level_hand_derived_mean_variance():
    """Equation 7: x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps."""
    betas = beta_schedule()
    abar = abar_from_betas(betas)
    t = 500  # mathematical level 500 -> array index 499
    idx = t - 1
    a = float(abar[idx])
    n = 200_000
    x0 = jnp.full((n, 2), 2.0, dtype=jnp.float32)
    xt = noise_to_level(x0, t, abar, jax.random.PRNGKey(4))
    xt = jax.device_get(xt)
    assert abs(float(xt[:, 0].mean()) - math.sqrt(a) * 2.0) < 0.01
    assert abs(float(xt[:, 0].var()) - (1.0 - a)) < 0.01


def test_noise_to_level_deterministic_and_level0_identity():
    betas = beta_schedule()
    abar = abar_from_betas(betas)
    x0 = jnp.array([[3.0, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    k = jax.random.PRNGKey(5)
    a = noise_to_level(x0, 10, abar, k)
    b = noise_to_level(x0, 10, abar, k)
    assert bool(jnp.all(a == b))
    # t = 0: no shrinkage, no noise
    c = noise_to_level(x0, 0, abar, k)
    assert bool(jnp.all(c == x0))


def test_noised_level_1000_is_nearly_standard_gaussian():
    """t=1000 with centres at (3,0): coordinate means < 0.1, variance ~ 1."""
    betas = beta_schedule()
    abar = abar_from_betas(betas)
    x0 = jnp.full((200_000, 2), 3.0, dtype=jnp.float32)
    xt = noise_to_level(x0, 1000, abar, jax.random.PRNGKey(6))
    xt = jax.device_get(xt)
    assert abs(float(xt[:, 0].mean())) < 0.1
    assert abs(float(xt[:, 0].var()) - 1.0) < 0.1
    assert abs(float(xt[:, 1].mean())) < 0.1
    assert abs(float(xt[:, 1].var()) - 1.0) < 0.1
