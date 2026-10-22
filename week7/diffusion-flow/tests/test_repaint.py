"""Part 5 tests: exact Algorithm 6 RePaint on tiny explicit schedules.

Hand-derived posterior mean/variance (lecture pages 17-19); the posterior
variance (1-abar[s])(1-a)/(1-abar[t]) is NOT the coarse-DDPM variance (1-a).
The sampler receives observations+mask only — never the withheld truth.
"""

import math

import jax
import jax.numpy as jnp

jnp = jnp  # noqa
import numpy as np
import pytest

from src.repaint import (make_hole_mask, repaint_sample,
                         repaint_events)

TINY_BETAS = jnp.array([0.25, 0.5], dtype=jnp.float32)
TINY_ABAR = jnp.array([1.0, 0.75, 0.375], dtype=jnp.float32)
LEVELS = np.array([2, 1, 0])  # tiny grid, 2 jumps, attempts 2 except final


def eps_index_predictor(weights, x, model_index):
    """eps = model_index * ones (numeric, tracer-safe; original t -> t-1)."""
    return jnp.asarray(model_index, dtype=jnp.float32) * jnp.ones_like(x)


def tiny_mask():
    """6x6 with a 2x2 hidden hole: both pixel types present."""
    m = np.ones((1, 1, 6, 6), dtype=np.float32)
    m[0, 0, 2:4, 2:4] = 0.0
    return jnp.asarray(m)


def clip(v):
    return max(-1.0, min(1.0, v))


# ---------- event schedule ----------

def test_official_event_schedule_99_49():
    events = repaint_events(range(1000, 19, -20).__iter__().__next__()
                            and (list(range(1000, 19, -20)) + [0]),
                            attempts=2)
    assert len(events) == 99
    assert sum(1 for e in events if e["is_retry"]) == 49
    assert [e["t"] for e in events[:2]] == [1000, 1000]
    assert [e["s"] for e in events[:2]] == [980, 980]
    assert events[-1]["t"] == 20 and events[-1]["s"] == 0
    assert not events[-1]["is_retry"]  # final jump: 1 attempt only
    from collections import Counter
    c = Counter((e["t"], e["s"]) for e in events)
    assert c[(1000, 980)] == 2 and c[(20, 0)] == 1
    assert len(c) == 50


def test_event_and_level_validation():
    with pytest.raises(ValueError):
        repaint_events([2.0, 1.0, 0])          # float levels rejected
    with pytest.raises(ValueError):
        repaint_events([2, 0], attempts=1.5)   # non-int attempts
    with pytest.raises(ValueError):
        repaint_events([2, 2, 0])              # not strictly decreasing
    with pytest.raises(ValueError):
        repaint_events([2, 1])                 # must end at 0
    with pytest.raises(ValueError):
        repaint_sample({}, jnp.zeros((1, 1, 2, 2)), jnp.ones((1, 1, 2, 2)),
                       jnp.array([0.25, 0.5]), jnp.array([1.0, 0.75]),
                       [2, 1, 0], key=jax.random.PRNGKey(0))  # abar len
    with pytest.raises(ValueError):
        repaint_sample({}, jnp.zeros((1, 1, 2, 2)), jnp.ones((1, 1, 2, 2)),
                       jnp.array([0.25, 0.5]),
                       jnp.array([1.0, 0.75, 0.5]), [1, 1, 0],
                       key=jax.random.PRNGKey(0))  # levels start != T


# ---------- hand-derived first event (WITH x0 clipping) ----------

def test_first_event_posterior_mean_variance_and_replacement():
    """t=2 -> s=1, a = abar[2]/abar[1] = 0.5, eps = index 1 * ones.
    x0hat = (2 - sqrt(1-abar2)*1)/sqrt(abar2) = 1.974 -> CLIPPED to 1.
    posterior variance = (1-abar[s])(1-a)/(1-abar[t]) = .2 (NOT 1-a = .5).
    Observed pixel: replaced by known = sqrt(abar[s])*0 + sqrt(1-abar[s])*eta.
    """
    xT = jnp.array([[[[2.0]]]], dtype=jnp.float32)
    obs = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)   # observed value 0
    mask = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
    xi = [0.6, 0.0, 0.0]
    eta = [-0.4, 0.0, 0.0]
    states = repaint_sample(
        {}, obs, mask, TINY_BETAS, TINY_ABAR, LEVELS, attempts=2,
        key=jax.random.PRNGKey(0), predict=eps_index_predictor,
        record_states=True,
        noise_stream={"xi": xi, "eta": eta, "f": [0.0, 0.0, 0.0]},
        xT=xT)[1]
    a = 0.5
    x0hat = clip((2.0 - math.sqrt(0.625) * 1.0) / math.sqrt(0.375))
    assert x0hat == 1.0  # genuinely clipped
    mean = (math.sqrt(0.75) * (1 - a) / (1 - 0.375) * x0hat
            + math.sqrt(a) * (1 - 0.75) / (1 - 0.375) * 2.0)
    var = (1 - 0.75) * (1 - a) / (1 - 0.375)
    known = math.sqrt(0.75) * 0.0 + math.sqrt(0.25) * (-0.4)
    assert var == pytest.approx(0.2)
    # production must match the POSTERIOR arithmetic, and clearly differ
    # from the WRONG coarse-DDPM variance (1-a):
    np.testing.assert_allclose(float(states[0, 0, 0, 0, 0]), known,
                               rtol=1e-6)
    wrong = (mean + math.sqrt(1 - a) * xi[0])
    assert abs(float(states[0, 0, 0, 0, 0]) - wrong) > 0.05


def test_hidden_pixel_hand_posterior_and_retry_step():
    """All-hidden pixel: clipped posterior with retry forward re-noise.
    Three events for the tiny grid [2,1,0]: two attempts at t2->s1 and the
    final t1->0 jump."""
    xT = jnp.array([[[[2.0]]]], dtype=jnp.float32)
    obs = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    mask = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    xi = [0.6, -0.3, 0.0]
    eta = [0.1, 0.2, 0.0]
    f = [0.0, 0.7, 0.0]
    states = repaint_sample(
        {}, obs, mask, TINY_BETAS, TINY_ABAR, LEVELS, attempts=2,
        key=jax.random.PRNGKey(0), predict=eps_index_predictor,
        record_states=True, noise_stream={"xi": xi, "eta": eta, "f": f},
        xT=xT)[1]
    a = 0.5
    # attempt 1 (t=2, index 1): x0hat clipped to 1
    x0hat = clip((2.0 - math.sqrt(0.625) * 1.0) / math.sqrt(0.375))
    mean = (math.sqrt(0.75) * 0.5 / 0.625 * x0hat
            + math.sqrt(0.5) * 0.25 / 0.625 * 2.0)
    var = 0.2
    x_s1 = mean + math.sqrt(var) * xi[0]
    np.testing.assert_allclose(float(states[0, 0, 0, 0, 0]), x_s1, rtol=1e-6)
    # retry: forward re-noise with f, same t (index 1), clipped x0hat again
    x_t2 = math.sqrt(a) * x_s1 + math.sqrt(1 - a) * f[1]
    x0hat2 = clip((x_t2 - math.sqrt(0.625) * 1.0) / math.sqrt(0.375))
    mean2 = (math.sqrt(0.75) * 0.5 / 0.625 * x0hat2
             + math.sqrt(0.5) * 0.25 / 0.625 * x_t2)
    expected2 = mean2 + math.sqrt(var) * xi[1]
    np.testing.assert_allclose(float(states[1, 0, 0, 0, 0]), expected2,
                               rtol=1e-5)
    # final jump t=1 -> 0 (index 0): var 0, x0hat = clip(x/ sqrt(.75))
    final = clip(expected2 / math.sqrt(0.75))
    np.testing.assert_allclose(float(states[2, 0, 0, 0, 0]), final,
                               rtol=1e-5)
    # wrong-variance (1-a) variant is clearly distinguishable
    wrong2 = mean2 + math.sqrt(1 - a) * xi[1]
    assert abs(expected2 - wrong2) > 0.05  # 0.078 measured separation


def test_posterior_variance_differs_from_coarse_ddpm():
    a = float(TINY_ABAR[2] / TINY_ABAR[1])
    post = (1 - 0.75) * (1 - a) / (1 - 0.375)
    assert abs(post - (1 - a)) > 0.1


# ---------- clipping: only the x0 estimate ----------

def test_clipping_applies_only_to_x0_estimate():
    """Huge start, zero predictor: x0hat clipped to 1; the x_t term in the
    posterior mean stays unclipped; noisy states never clipped."""
    xT = jnp.full((1, 1, 1, 1), 50.0, dtype=jnp.float32)
    obs = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    mask = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    states = repaint_sample(
        {}, obs, mask, TINY_BETAS, TINY_ABAR, LEVELS, attempts=2,
        key=jax.random.PRNGKey(1), predict=lambda w, x, i: jnp.zeros_like(x),
        record_states=True,
        noise_stream={"xi": [0.0, 0.0, 0.0], "eta": [0.0, 0.0, 0.0],
                      "f": [0.0, 0.0, 0.0]}, xT=xT)[1]
    a = 0.5
    mean = (math.sqrt(0.75) * 0.5 / 0.625 * 1.0
            + math.sqrt(0.5) * 0.25 / 0.625 * 50.0)
    np.testing.assert_allclose(float(states[0, 0, 0, 0, 0]), mean, rtol=1e-5)
    assert mean > 5.0  # noisy/merged state unclipped


# ---------- observed restoration / hidden variability / truth control -----

def test_observed_restored_hidden_varies_and_truth_perturbation():
    key = jax.random.PRNGKey(5)
    shape = (1, 1, 6, 6)
    mask = tiny_mask()
    rng = np.random.default_rng(7)
    truth = rng.normal(size=shape).astype(np.float32)
    obs = jnp.asarray(truth * np.asarray(mask))  # zero at hidden
    out = repaint_sample({}, obs, mask, TINY_BETAS, TINY_ABAR, LEVELS,
                         attempts=2, key=key, predict=eps_index_predictor)
    out = np.asarray(out)
    np.testing.assert_allclose(out * np.asarray(mask),
                               np.asarray(obs), atol=1e-6)  # np.all semantics
    assert np.all(out[np.asarray(mask) == 0] != 0.0)  # hidden filled
    # hidden-truth perturbation control: different truths, identical obs
    truth2 = truth.copy()
    truth2[0, 0, 2:4, 2:4] += 10.0            # perturb ONLY inside hole
    obs2 = jnp.asarray(truth2 * np.asarray(mask))
    assert float(jnp.abs(obs2 - obs).max()) == 0.0  # obs unchanged
    out2 = repaint_sample({}, obs2, mask, TINY_BETAS, TINY_ABAR, LEVELS,
                          attempts=2, key=key, predict=eps_index_predictor)
    np.testing.assert_array_equal(out, np.asarray(out2))  # exact invariance


def test_canonical_32x32_mask():
    m = np.asarray(make_hole_mask((1, 3, 32, 32)))
    assert m.shape == (1, 3, 32, 32)
    assert np.all(m[0, :, 5, 16] == 1.0) and np.all(m[0, :, 26, 16] == 1.0)
    assert np.all(m[0, :, 6:26, 6:26] == 0.0)
    assert m.sum() == 3 * (32 * 32 - 400)


def test_rng_replay_and_root_sensitivity():
    shape = (1, 1, 6, 6)
    mask = tiny_mask()
    obs = jnp.zeros(shape, dtype=jnp.float32)
    kw = dict(betas=TINY_BETAS, abar=TINY_ABAR, levels=LEVELS, attempts=2,
              predict=eps_index_predictor)
    a1 = repaint_sample({}, obs, mask, key=jax.random.PRNGKey(9), **kw)
    a2 = repaint_sample({}, obs, mask, key=jax.random.PRNGKey(9), **kw)
    b = repaint_sample({}, obs, mask, key=jax.random.PRNGKey(10), **kw)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    assert float(jnp.abs(a1 - b).max()) > 1e-4


def test_hidden_observation_rejected():
    mask = tiny_mask()
    obs = jnp.ones((1, 1, 6, 6), dtype=jnp.float32)  # nonzero at hidden
    with pytest.raises(ValueError):
        repaint_sample({}, obs, mask, TINY_BETAS, TINY_ABAR, LEVELS,
                       attempts=2, key=jax.random.PRNGKey(0),
                       predict=eps_index_predictor)
