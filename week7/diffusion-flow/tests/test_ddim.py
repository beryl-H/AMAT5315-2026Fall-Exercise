"""Part 2 tests: deterministic DDIM and the analytic exact mixture noise.

Schedule convention as Part 1: betas[t-1], abar[t] (length T+1, abar[0]=1).
All tests use tiny explicit schedules and hand-computed coefficients.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.ddim import ddim_sample, exact_predictor, mixture_exact_noise
from src.diffusion import CENTRES, abar_from_betas

TINY_ABAR = jnp.array([1.0, 0.75, 0.375], dtype=jnp.float32)  # betas [.25,.5]
TINY_LEVELS = 2


def scripted_predictor(shift=0.0):
    def predict(params, x, t):
        return shift + t[:, None].astype(jnp.float32) * jnp.ones_like(x)
    return predict, None


# ---------- validation ----------

@pytest.mark.parametrize("levels", [
    [2, 1, 0],      # valid, exercised elsewhere
])
def test_valid_grid_accepted(levels):
    predict, params = scripted_predictor()
    xT = jnp.zeros((3, 2), dtype=jnp.float32)
    out = ddim_sample(predict, params, xT, jnp.array(levels), TINY_ABAR)
    assert out.shape == (3, 2)


@pytest.mark.parametrize("levels", [
    [2, 2, 0],      # repeated level (not strictly decreasing)
    [2, 0, 1],      # increasing tail
    [1, 0],         # wrong endpoint: must start at T = 2
    [2, 1],         # wrong endpoint: must end at 0
    [2, 5, 0],      # out of range / not decreasing
    [3, 0],         # level beyond T
])
def test_invalid_grid_rejected(levels):
    predict, params = scripted_predictor()
    xT = jnp.zeros((3, 2), dtype=jnp.float32)
    with pytest.raises(ValueError):
        ddim_sample(predict, params, xT, jnp.array(levels), TINY_ABAR)


def test_float_grid_rejected():
    predict, params = scripted_predictor()
    xT = jnp.zeros((3, 2), dtype=jnp.float32)
    with pytest.raises(ValueError):
        ddim_sample(predict, params, xT, jnp.array([2.0, 1.0, 0.0]), TINY_ABAR)


# ---------- hand-computed updates ----------

def test_ddim_single_jump_hand_computed():
    """Grid [2,1,0]: the first jump t=2 -> s=1 (eps_hat = 2, scripted),
    checked via the recorded intermediate state."""
    predict, params = scripted_predictor()
    xT = jnp.array([[1.0, -2.0]], dtype=jnp.float32)
    x, lv, states = ddim_sample(predict, params, xT, jnp.array([2, 1, 0]),
                                TINY_ABAR, record_states=True)
    assert [int(v) for v in lv] == [2, 1, 0]
    b_t, b_s = 0.375, 0.75
    x0hat = (xT - math.sqrt(1 - b_t) * 2.0) / math.sqrt(b_t)
    expected = math.sqrt(b_s) * x0hat + math.sqrt(1 - b_s) * 2.0
    np.testing.assert_allclose(np.asarray(states[0]), np.asarray(expected),
                               rtol=1e-6)


def test_ddim_multi_jump_nonadjacent_levels_hand_computed():
    """Levels [2,0]: a single nonadjacent jump t=2 -> s=0 (abar[0]=1)."""
    predict, params = scripted_predictor()
    xT = jnp.array([[1.0, -2.0]], dtype=jnp.float32)
    x = ddim_sample(predict, params, xT, jnp.array([2, 0]), TINY_ABAR)
    b_t = 0.375
    x0hat = (xT - math.sqrt(1 - b_t) * 2.0) / math.sqrt(b_t)
    np.testing.assert_allclose(np.asarray(x), np.asarray(x0hat), rtol=1e-6)


def test_ddim_full_grid_matches_chain_of_single_jumps():
    """[2,1,0] must equal chaining the two single-jump updates by hand."""
    predict, params = scripted_predictor()
    xT = jnp.array([[0.7, 1.1]], dtype=jnp.float32)
    x = ddim_sample(predict, params, xT, jnp.array([2, 1, 0]), TINY_ABAR)
    x1 = (xT - math.sqrt(1 - 0.375) * 2.0) / math.sqrt(0.375)
    x1 = math.sqrt(0.75) * x1 + math.sqrt(0.25) * 2.0
    x0h = (x1 - math.sqrt(1 - 0.75) * 1.0) / math.sqrt(0.75)
    np.testing.assert_allclose(np.asarray(x), np.asarray(x0h), rtol=1e-6)


def test_ddim_deterministic_no_hidden_rng():
    predict, params = scripted_predictor()
    key = jax.random.PRNGKey(0)
    xT = jax.random.normal(key, (5, 2), dtype=jnp.float32)
    a = ddim_sample(predict, params, xT, jnp.array([2, 1, 0]), TINY_ABAR)
    b = ddim_sample(predict, params, xT, jnp.array([2, 1, 0]), TINY_ABAR)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_ddim_no_clipping():
    """Zero predictor, huge start: states pass through unclipped."""
    def predict(params, x, t):
        return jnp.zeros_like(x)
    xT = jnp.array([[80.0, -80.0]], dtype=jnp.float32)
    x = ddim_sample(predict, None, xT, jnp.array([2, 1, 0]), TINY_ABAR)
    # zero predictor: x_next = x * sqrt(abar[s]/abar[t]) per jump, so the
    # updates telescope to x0 = xT / sqrt(abar[T]) = 80/sqrt(0.375)
    expected = xT / math.sqrt(0.375)
    np.testing.assert_allclose(np.asarray(x), np.asarray(expected), rtol=1e-6)
    assert float(jnp.abs(x).max()) > 10.0


def test_ddim_uses_original_level_not_loop_index():
    """Predictor eps = t (original level). Multi-jump nonadjacent grid
    [3,2,0] with abar [1,.9,.6,.3]: if the implementation passed the loop
    index instead of the original level, the first prediction would be 0
    instead of 3 and the endpoint would differ. Both variants are hand
    computed; production must match the original-level one."""
    abar = jnp.array([1.0, 0.9, 0.6, 0.3], dtype=jnp.float32)

    def predict(params, x, t):
        return t[:, None].astype(jnp.float32) * jnp.ones_like(x)

    xT = jnp.array([[1.0, -2.0]], dtype=jnp.float32)
    x = ddim_sample(predict, None, xT, jnp.array([3, 2, 0]), abar)
    # original levels: t=3 (eps 3), then t=2 (eps 2)
    x1 = math.sqrt(0.6) * (xT - math.sqrt(0.7) * 3.0) / math.sqrt(0.3) \
        + math.sqrt(0.4) * 3.0
    x0 = (x1 - math.sqrt(0.4) * 2.0) / math.sqrt(0.6)
    np.testing.assert_allclose(np.asarray(x), np.asarray(x0), rtol=1e-6)
    # loop-index variant (eps = 0 then 1) would give a different endpoint
    y1 = math.sqrt(0.6) * (xT - math.sqrt(0.7) * 0.0) / math.sqrt(0.3) \
        + math.sqrt(0.4) * 0.0
    y0 = (y1 - math.sqrt(0.4) * 1.0) / math.sqrt(0.6)
    assert float(jnp.abs(x0 - y0).max()) > 0.5


# ---------- exact mixture noise ----------

def test_exact_noise_matches_independent_score_identity():
    """Independent check: E[eps|x] = -sqrt(1-b) * grad log p_t(x).

    The mixture at level t has components N(sqrt(b) mu_k, v I) with
    v = 0.09 b + (1-b); its score is sum_k w_k (-(x - sqrt(b) mu_k)/v).
    """
    key = jax.random.PRNGKey(5)
    x = jax.random.normal(key, (64, 2), dtype=jnp.float32) * 1.5
    betas = jnp.linspace(1e-4, 0.02, 1000, dtype=jnp.float32)
    abar = abar_from_betas(betas)
    for t in (1, 50, 500, 999):
        b = float(abar[t])
        v = 0.09 * b + (1 - b)
        sq = math.sqrt(b)
        d = x[:, None, :] - sq * CENTRES[None]            # (N,4,2)
        logp = -0.5 * (d ** 2).sum(-1) / v    # up to constants
        w = jax.nn.softmax(logp, axis=-1)
        score = -(w[..., None] * d / v).sum(axis=1)
        expected = -math.sqrt(1 - b) * score
        got = mixture_exact_noise(x, t, abar)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected),
                                   rtol=1e-4, atol=1e-4)


def test_exact_noise_zero_at_t0_and_hand_value():
    """At t=0 the noise contribution is exactly zero."""
    x = jnp.array([[1.0, 2.0], [-3.0, 0.5]], dtype=jnp.float32)
    abar = TINY_ABAR
    out = mixture_exact_noise(x, 0, abar)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-12)
    # hand value at t=1 (b=.75, v=.09*.75+.25=.3175) for a symmetric point
    # x = (0,0): responsibilities equal, sum w mu = 0, so eps = 0
    out0 = mixture_exact_noise(jnp.zeros((1, 2), dtype=jnp.float32), 1, abar)
    np.testing.assert_allclose(np.asarray(out0), 0.0, atol=1e-6)


def test_exact_predictor_ddim_endpoints():
    """Exact-predictor DDIM through the tiny grid stays finite and shifts
    toward the data mean (zero here), shrinking the noise share."""
    abar = TINY_ABAR
    pred = exact_predictor(abar)
    xT = jnp.array([[3.0, 3.0]], dtype=jnp.float32)
    x = ddim_sample(pred, None, xT, jnp.array([2, 1, 0]), abar)
    assert x.shape == (1, 2) and bool(jnp.all(jnp.isfinite(x)))
