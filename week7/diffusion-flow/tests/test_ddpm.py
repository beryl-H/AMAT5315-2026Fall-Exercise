"""Task 3 tests: DDPM reverse sampler (Algorithm 2), production code only.

Convention: betas[t-1], alpha[t-1], abar[t]; time features use (t-1).
A scripted predictor (no network) lets us hand-compute means exactly.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.diffusion import abar_from_betas, beta_schedule
from src.sample import ddpm_sample


def tiny_schedule():
    betas = jnp.array([0.25, 0.5], dtype=jnp.float32)  # T = 2 levels
    alphas = 1.0 - betas
    abar = abar_from_betas(betas)  # [1, .75, .375]
    return betas, alphas, abar


def scripted_predictor(shift=0.0):
    """Predictor returning a known function of (x, t)."""
    def predict(params, x, t):
        # eps_theta(x, t) = t * ones (distinguishable per level)
        return shift + t[:, None].astype(jnp.float32) * jnp.ones_like(x)
    return predict, None


def test_ddpm_two_steps_hand_computed():
    betas, alphas, abar = tiny_schedule()
    predict, params = scripted_predictor()
    key = jax.random.PRNGKey(0)
    k_init, k_rev = jax.random.split(key)
    xT = jax.random.normal(k_init, (1, 2), dtype=jnp.float32)
    # Shared reverse-step noise array for t=2, then t=1
    z2 = jax.random.normal(k_rev, (1, 2), dtype=jnp.float32)
    z1 = jnp.zeros((1, 2), dtype=jnp.float32)  # t=1 adds no noise
    x = ddpm_sample(predict, params, xT, betas, abar,
                    reverse_noise={2: z2, 1: z1})
    # Hand computation, from the lecture formula
    # t=2: mu = (xT - beta2/sqrt(1-abar2) * eps_hat)/sqrt(alpha2)
    eps2 = 2.0  # scripted: t * ones
    mu2 = (xT - 0.5 / math.sqrt(1 - 0.375) * eps2) / math.sqrt(1 - 0.5)
    x1 = mu2 + math.sqrt(0.5) * z2
    # t=1: mu = (x1 - beta1/sqrt(1-abar1) * eps_hat)/sqrt(alpha1); no noise
    eps1 = 1.0
    mu1 = (x1 - 0.25 / math.sqrt(1 - 0.75) * eps1) / math.sqrt(1 - 0.25)
    np.testing.assert_allclose(np.asarray(x), np.asarray(mu1), rtol=1e-6,
                               atol=1e-6)


def test_ddpm_final_step_adds_no_noise():
    betas, alphas, abar = tiny_schedule()
    predict, params = scripted_predictor()
    key = jax.random.PRNGKey(1)
    k_init, k_rev = jax.random.split(key)
    xT = jax.random.normal(k_init, (1, 2), dtype=jnp.float32)
    # Even a nonzero noise array at t=1 must be ignored
    z1 = 100.0 * jnp.ones((1, 2), dtype=jnp.float32)
    z2 = jnp.zeros((1, 2), dtype=jnp.float32)
    x = ddpm_sample(predict, params, xT, betas, abar,
                    reverse_noise={2: z2, 1: z1})
    # Recompute the mean at the last step from the documented recursion:
    mu2 = (xT - 0.5 / math.sqrt(1 - 0.375) * 2.0) / math.sqrt(0.5)
    x1 = mu2  # z2 = 0
    mu1 = (x1 - 0.25 / math.sqrt(0.25) * 1.0) / math.sqrt(0.75)
    np.testing.assert_allclose(np.asarray(x), np.asarray(mu1), rtol=1e-5)


def test_ddpm_no_clipping():
    """States far outside [-1, 1] must pass through unclipped."""
    betas, alphas, abar = tiny_schedule()
    # Predictor that returns 0: means are just shrinkage of xT
    def predict(params, x, t):
        return jnp.zeros_like(x)
    xT = jnp.array([[50.0, -50.0]], dtype=jnp.float32)
    z = {2: jnp.zeros((1, 2), dtype=jnp.float32),
         1: jnp.zeros((1, 2), dtype=jnp.float32)}
    x = ddpm_sample(predict, None, xT, betas, abar, reverse_noise=z)
    # t=2: x1 = xT/sqrt(alpha2); t=1: x0 = x1/sqrt(alpha1)
    expected = xT / math.sqrt(0.5) / math.sqrt(0.75)
    np.testing.assert_allclose(np.asarray(x), np.asarray(expected), rtol=1e-6)
    assert float(jnp.abs(x).max()) > 10.0  # definitely not clipped


def test_ddpm_negative_control_wrong_variance_fails():
    """A wrong reverse variance (std used as variance) must disagree."""
    betas, alphas, abar = tiny_schedule()
    predict, params = scripted_predictor()
    key = jax.random.PRNGKey(2)
    k_init, k_rev = jax.random.split(key)
    xT = jax.random.normal(k_init, (1, 2), dtype=jnp.float32)
    z2 = jax.random.normal(k_rev, (1, 2), dtype=jnp.float32)
    z = {2: z2, 1: jnp.zeros((1, 2), dtype=jnp.float32)}
    correct = ddpm_sample(predict, params, xT, betas, abar, reverse_noise=z)
    # Buggy variant: noise std = beta (i.e. variance beta^2) at t=2
    mu2 = (xT - 0.5 / math.sqrt(1 - 0.375) * 2.0) / math.sqrt(0.5)
    x1_buggy = mu2 + 0.5 * z2  # WRONG: std 0.5 instead of sqrt(0.5)
    x0_buggy = (x1_buggy - 0.25 / math.sqrt(0.25) * 1.0) / math.sqrt(0.75)
    assert float(jnp.abs(correct - x0_buggy).max()) > 0.1


def test_ddpm_full_schedule_matches_moments():
    """With a zero predictor over the real schedule, the endpoint mean is
    the deterministic shrinkage chain; check T=1000 hand coefficient."""
    betas = beta_schedule()
    abar = abar_from_betas(betas)
    def predict(params, x, t):
        return jnp.zeros_like(x)
    xT = jnp.array([[1.0, 1.0]], dtype=jnp.float32)
    z = {t: jnp.zeros((1, 2), dtype=jnp.float32) for t in range(1, 1001)}
    x = ddpm_sample(predict, None, xT, betas, abar, reverse_noise=z)
    # x_{t-1} = x_t / sqrt(alpha_t) => x0 = xT / sqrt(abar_1000)
    expected = xT / math.sqrt(float(abar[1000]))
    np.testing.assert_allclose(np.asarray(x), np.asarray(expected), rtol=1e-4)


# ---------- regression: fresh-noise production branch (coordinator review) ----------

def test_ddpm_fresh_key_mode_independent_innovations():
    """Production fresh-key path: innovations (x_{t-1}-mu)/sqrt(beta_t) must
    be independent draws, not one repeated array."""
    betas, alphas, abar = tiny_schedule()
    def predict(params, x, t):
        return jnp.zeros_like(x)
    xT = jnp.array([[0.3, -0.7], [1.0, 0.5], [-0.2, 0.9]],
                   dtype=jnp.float32)
    x, states = ddpm_sample(predict, None, xT, betas, abar,
                            key=jax.random.PRNGKey(11), record_states=True)
    # states: [xT, x1, x0]; innovation at t=2 (first reverse step)
    mu2 = (xT - 0.5 / math.sqrt(1 - 0.375) * 0.0) / math.sqrt(0.5)
    innov2 = (states[1] - mu2) / math.sqrt(0.5)
    mu1 = states[1] / math.sqrt(0.75)  # zero predictor, t=1
    innov1 = (states[2] - mu1) / math.sqrt(0.25)  # must be zero (t=1 no noise)
    assert float(jnp.abs(innov1).max()) == 0.0
    # successive innovations differ (not one repeated array)
    assert float(jnp.abs(innov2).sum()) > 1e-6
    # a different sampler key changes the output; same key reproduces it
    xa = ddpm_sample(predict, None, xT, betas, abar, key=jax.random.PRNGKey(11))
    xb = ddpm_sample(predict, None, xT, betas, abar, key=jax.random.PRNGKey(12))
    xc = ddpm_sample(predict, None, xT, betas, abar, key=jax.random.PRNGKey(11))
    assert float(jnp.abs(xa - xb).max()) > 1e-6
    assert float(jnp.abs(xa - xc).max()) == 0.0


def test_ddpm_fresh_key_mode_large_sample_uncorrelated():
    """1000 shared-start paths: innovations across samples near zero mean."""
    betas = beta_schedule()
    abar = abar_from_betas(betas)
    def predict(params, x, t):
        return jnp.zeros_like(x)
    n = 4000
    xT = jnp.zeros((n, 2), dtype=jnp.float32)
    x, states = ddpm_sample(predict, None, xT, betas, abar,
                            key=jax.random.PRNGKey(3), record_states=True)
    states = [jnp.asarray(s) for s in states]
    t = 500
    a_t = float(abar[t]); beta_t = float(betas[t - 1]); alpha_t = 1 - beta_t
    mu = states[1000 - t] / math.sqrt(alpha_t)  # zero predictor
    innov = (states[1000 - t + 1] - mu) / math.sqrt(beta_t)
    assert abs(float(innov.mean())) < 0.05  # zero-mean fresh noise
    # empirical correlation between the two coordinates near zero
    c = float(jnp.mean((innov[:, 0] - innov[:, 0].mean())
                       * (innov[:, 1] - innov[:, 1].mean())))
    assert abs(c) < 0.05


def test_ddpm_missing_key_rejected_in_fresh_mode():
    betas, alphas, abar = tiny_schedule()
    def predict(params, x, t):
        return jnp.zeros_like(x)
    with pytest.raises(ValueError):
        ddpm_sample(predict, None, jnp.zeros((1, 2)), betas, abar)
