"""Part 4 sampler tests with a mock image predictor (no real weights)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.diffusion import abar_from_betas
from scripts.part4_generate import chunked_image_ddpm


def tiny_schedule():
    betas = jnp.array([0.25, 0.5], dtype=jnp.float32)
    abar = jnp.array([1.0, 0.75, 0.375], dtype=jnp.float32)
    return betas, abar


def mock_predictor(record):
    """Predictor recording the model index it sees; output t*ones."""
    def predict(weights, x, model_index):
        record.append(int(np.asarray(jax.device_get(model_index))[0])
                      if np.ndim(model_index) > 0 else None)
        return jnp.asarray(model_index, dtype=jnp.float32)[..., None,
                                                          None, None] \
            * jnp.ones_like(x)
    return predict


def test_chunked_sampler_hand_two_steps():
    """Tiny T=2 schedule betas [.25,.5], abar [1,.75,.375]; predictor
    returns eps = MODEL INDEX ones. xT=1, innovation at t=2 is 0.5:
      t=2 (index 1): mu2 = (1 - .5/sqrt(.625)*1)/sqrt(.5)
                     x1 = mu2 + sqrt(.5)*0.5
      t=1 (index 0): x0 = x1/sqrt(.75), final noise ignored.
    A loop-index confusion (eps 0 then 1) gives a different endpoint."""
    betas, abar = tiny_schedule()
    xT = jnp.array([[[[1.0]]]], dtype=jnp.float32)       # (B,C,H,W)
    z = jnp.array([[[[[0.5]]]], [[[[0.0]]]]], dtype=jnp.float32)

    def predict(weights, x, model_index):
        # tracer-safe numeric function of the model index
        return jnp.asarray(model_index, dtype=jnp.float32) * jnp.ones_like(x)

    states = chunked_image_ddpm({}, xT, z, betas, abar, chunk=1,
                                predict=predict)
    assert states.shape == (3, *xT.shape)
    mu2 = (1.0 - 0.5 / math.sqrt(0.625) * 1.0) / math.sqrt(0.5)
    x1 = mu2 + math.sqrt(0.5) * 0.5
    x0 = x1 / math.sqrt(0.75)
    np.testing.assert_allclose(states[1, 0, 0, 0, 0], x1, rtol=1e-5)
    np.testing.assert_allclose(states[2, 0, 0, 0, 0], x0, rtol=1e-5)
    # explicit defect emulation: predictions = loop index (0 then 1)
    counter = {"n": 0}

    def defect(weights, x, model_index):
        counter["n"] += 1
        return jnp.ones_like(x) * (counter["n"] - 1)

    states_defect = chunked_image_ddpm({}, xT, z, betas, abar, chunk=1,
                                       predict=defect)
    assert float(jnp.abs(states - states_defect).max()) > 1e-3


def test_chunked_validation():
    betas, abar = tiny_schedule()
    xT = jnp.zeros((1, 1, 1, 1))
    with pytest.raises(ValueError):
        chunked_image_ddpm({}, xT, jnp.zeros((2, 1, 1, 1)), betas,
                           jnp.array([1.0, 0.75]), chunk=1,
                           predict=lambda w, x, i: x)  # abar wrong length
    with pytest.raises(ValueError):
        chunked_image_ddpm({}, xT, jnp.zeros((2, 1, 1)), betas, abar,
                           chunk=1, predict=lambda w, x, i: x)  # z shape


def test_chunked_matches_part1_sampler_on_image_shapes():
    """Same schedule/noise: chunked image sampler == Part 1 ddpm_sample.
    Part 1's predictor receives the ORIGINAL t, so it returns .01*(t-1);
    the image mock receives the model index and returns .01*index."""
    from src.sample import ddpm_sample
    T = 50
    betas = jnp.linspace(1e-4, 0.02, T, dtype=jnp.float32)
    abar = abar_from_betas(betas)
    k_xT, k_z = jax.random.split(jax.random.PRNGKey(11))
    xT = jax.random.normal(k_xT, (1, 3, 8, 8), dtype=jnp.float32)
    z = jax.random.normal(k_z, (T, 1, 3, 8, 8), dtype=jnp.float32)

    def image_mock(weights, x, model_index):
        return 0.01 * jnp.asarray(model_index, dtype=jnp.float32) \
            * jnp.ones_like(x)

    states = chunked_image_ddpm({}, xT, z, betas, abar, chunk=10,
                                predict=image_mock)
    fine = ddpm_sample(
        lambda p, x, t: 0.01 * (t - 1)[:, None, None, None]
        .astype(jnp.float32) * jnp.ones_like(x),
        None, xT, betas, abar,
        reverse_noise={t: z[T - t] for t in range(1, T + 1)})
    np.testing.assert_allclose(states[-1], np.asarray(fine), rtol=1e-4,
                               atol=1e-6)
    np.testing.assert_allclose(states[0], np.asarray(xT), rtol=1e-6)


def test_chunked_final_noise_omitted_and_no_clipping():
    betas, abar = tiny_schedule()
    xT = jnp.full((1, 1, 1, 1), 50.0, dtype=jnp.float32)

    def predict(weights, x, model_index):
        return jnp.zeros_like(x)

    z = jnp.zeros((2, *xT.shape), dtype=jnp.float32)
    states = chunked_image_ddpm({}, xT, z, betas, abar, chunk=1,
                                predict=predict)
    # final-slot innovation set to 100 must be ignored exactly
    z2 = jnp.zeros((2, *xT.shape), dtype=jnp.float32).at[-1].set(100.0)
    states2 = chunked_image_ddpm({}, xT, z2, betas, abar, chunk=1,
                                 predict=predict)
    np.testing.assert_array_equal(states, states2)
    # zero predictor telescopes x_{t-1} = x_t / sqrt(alpha_t), unclipped
    expected = 50.0 / math.sqrt(0.75) / math.sqrt(0.5)
    np.testing.assert_allclose(float(states2[-1, 0, 0, 0, 0]), expected,
                               rtol=1e-5)
    assert expected > 10.0
