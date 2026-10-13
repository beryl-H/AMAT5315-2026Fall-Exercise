"""Mathematical tests for the Week 6 Part 1 diffusion building blocks.

These tests are derived from the SPEC / learning-sheet identities, not from any
implementation. They define what "correct" means for the schedule, the forward
noising process, the exact mixture score, and the DDPM reverse update.
"""

import jax
import jax.numpy as jnp
import numpy as np

import ddm

T = 1000
CENTERS = jnp.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])


def test_schedule_is_linear_with_cumulative_alphas():
    sched = ddm.make_schedule(num_levels=T, beta_min=1e-4, beta_max=0.02)
    betas = np.asarray(sched["betas"])
    abar = np.asarray(sched["abar"])

    assert betas.shape == (T + 1,)
    assert betas[0] == 0.0
    np.testing.assert_allclose(betas[1], 1e-4)
    np.testing.assert_allclose(betas[T], 0.02)

    # betas[1:] must be linearly spaced from 1e-4 to 0.02 inclusive.
    np.testing.assert_allclose(betas[1:], np.linspace(1e-4, 0.02, T))

    # abar_0 = 1 and abar_t is the cumulative product of (1 - beta_s).
    # The implementation is float32, so allow a few parts per million of
    # cumulative rounding against the float64 NumPy product.
    np.testing.assert_allclose(abar[0], 1.0)
    np.testing.assert_allclose(abar, np.cumprod(1.0 - betas), rtol=2e-6, atol=2e-6)


def test_noising_matches_closed_forward_form():
    sched = ddm.make_schedule()
    x0 = jnp.array([[0.5, -1.0]])
    t = jnp.array([250])
    key = jax.random.PRNGKey(0)

    x_t, eps = ddm.noise_forward(key, x0, t, sched)

    a = sched["sqrt_abar"][250]
    b = sched["sqrt_1m_abar"][250]
    expected = a * x0 + b * eps
    np.testing.assert_allclose(np.asarray(x_t), np.asarray(expected), rtol=1e-6, atol=1e-7)


def test_exact_score_reduces_to_single_gaussian_formula():
    # With one zero-mean component at t = 0 there is no noising, so the score
    # must equal the exact Gaussian score -x / variance.
    sched = ddm.make_schedule()
    centers = jnp.array([[0.0, 0.0]])
    x = jnp.array([[0.3, -0.2], [1.0, 2.0]])
    t = jnp.array([0, 0])

    score = ddm.exact_score(x, t, sched, centers, data_var=0.09)

    expected = -x / 0.09
    np.testing.assert_allclose(np.asarray(score), np.asarray(expected), rtol=1e-6, atol=1e-7)


def test_exact_score_is_gradient_of_log_density():
    # The score must be the gradient of the log of the mixture density for the
    # four-component target at a noised level.
    sched = ddm.make_schedule()
    x = jnp.array([[0.7, -1.3]])
    t = jnp.array([500])

    def logp_scalar(x):
        density = ddm.log_mixture_density(x, t, sched, CENTERS, 0.09)
        return jnp.sum(density)

    score_ad = jax.grad(logp_scalar)(x)
    score_analytic = ddm.exact_score(x, t, sched, CENTERS, 0.09)

    np.testing.assert_allclose(
        np.asarray(score_ad), np.asarray(score_analytic), rtol=2e-6, atol=2e-6
    )


def test_reverse_mean_matches_ddpm_equation():
    sched = ddm.make_schedule()
    x_t = jnp.array([[1.0, -2.0]])
    eps_pred = jnp.array([[0.3, 0.1]])
    t = jnp.array([500])

    mean = ddm.reverse_mean(x_t, t, sched, eps_pred)

    alpha_t = sched["alphas"][500]
    abar_t = sched["abar"][500]
    beta_t = sched["betas"][500]
    expected = (x_t - (beta_t / jnp.sqrt(1.0 - abar_t)) * eps_pred) / jnp.sqrt(alpha_t)

    np.testing.assert_allclose(np.asarray(mean), np.asarray(expected), rtol=1e-6, atol=1e-7)


def test_reverse_step_has_no_noise_on_final_step():
    sched = ddm.make_schedule()
    x_t = jnp.array([[0.4, -0.8]])
    eps_pred = jnp.array([[0.1, -0.2]])
    t = jnp.array([1])

    out1 = ddm.reverse_step(x_t, t, sched, eps_pred, jax.random.PRNGKey(1))
    out2 = ddm.reverse_step(x_t, t, sched, eps_pred, jax.random.PRNGKey(2))

    assert out1.shape == x_t.shape == out2.shape
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), rtol=0.0, atol=0.0)


def test_reverse_step_adds_variance_beta_t_off_final_step():
    sched = ddm.make_schedule()
    x_t = jnp.array([[0.4, -0.8]])
    eps_pred = jnp.zeros_like(x_t)
    t = jnp.array([500])
    n_trials = 20000

    keys = jax.random.split(jax.random.PRNGKey(0), n_trials)
    outs = jax.vmap(lambda k: ddm.reverse_step(x_t, t, sched, eps_pred, k))(keys)

    mean = jnp.mean(outs, axis=0)
    var = jnp.var(outs, axis=0)

    np.testing.assert_allclose(
        np.asarray(mean), np.asarray(ddm.reverse_mean(x_t, t, sched, eps_pred)), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(var),
        np.full((1, 2), float(sched["betas"][500])),
        rtol=2e-2,
        atol=2e-2,
    )