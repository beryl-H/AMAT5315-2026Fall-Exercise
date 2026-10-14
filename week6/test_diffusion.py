"""Part 2 tests for the `diffusion` package, derived from the learning sheet.

Areas covered (per the Part 2 brief): preprocessing and split isolation,
forward process, EMA update and checkpoint save/load, reverse grid endpoints
and network-evaluation counts, effective alpha/beta on jumps, DDPM step
variance and final-step noise omission, serialization dimensions/ranges/
rounding, and trajectory shapes/levels.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from diffusion import checkpoint, data, ema, io, model, process, sampler, schedule, trajectory


# ---- Data: preprocessing and split isolation -------------------------------

def test_preprocess_pads_two_zero_pixels_then_scales_to_pm1():
    img = np.full((28, 28), 128, np.uint8)
    img[0, 0] = 0
    img[27, 27] = 255

    out = data.preprocess(img[None])

    assert out.shape == (1, 32, 32)
    assert out.dtype == np.float32
    # the padded border is exactly -1
    assert np.all(out[0, :2, :] == -1.0)
    assert np.all(out[0, -2:, :] == -1.0)
    assert np.all(out[0, :, :2] == -1.0)
    assert np.all(out[0, :, -2:] == -1.0)
    # original pixels shifted by 2 and scaled 0..255 -> -1..1
    np.testing.assert_allclose(float(out[0, 2, 2]), -1.0, atol=0)          # was 0
    np.testing.assert_allclose(float(out[0, 29, 29]), 1.0, atol=0)         # was 255
    np.testing.assert_allclose(float(out[0, 16, 16]), 128 / 127.5 - 1.0, atol=1e-7)


def test_training_data_uses_only_train_split(tmp_path):
    # Train pixels carry the value 10 everywhere; reserved test pixels carry
    # 200. If any test image leaked into training data, the max would exceed
    # the train scaling of 10.
    np.savez(
        tmp_path / "mnist.npz",
        x_train=np.full((6, 28, 28), 10, np.uint8),
        y_train=np.zeros(6, np.uint8),
        x_test=np.full((4, 28, 28), 200, np.uint8),
        y_test=np.ones(4, np.uint8),
    )
    path = tmp_path / "mnist.npz"

    x_train, y_train, x_test, y_test = data.load_splits(path)
    assert x_train.shape == (6, 28, 28)
    assert x_test.shape == (4, 28, 28)
    assert len(y_train) == 6 and len(y_test) == 4

    train_images = data.load_training_images(path)
    assert train_images.shape == (6, 32, 32)
    expected = 10 / 127.5 - 1.0
    # check the interior only: the zero-padded border is exactly -1
    interior = train_images[:, 2:-2, 2:-2]
    np.testing.assert_allclose(interior.min(), expected, atol=1e-7)
    np.testing.assert_allclose(interior.max(), expected, atol=1e-7)


# ---- Schedule and forward process ------------------------------------------

def test_schedule_matches_part2_constants():
    sched = schedule.make_schedule()
    betas = np.asarray(sched["betas"])
    assert betas.shape == (1001,)
    np.testing.assert_allclose(betas[1:], np.linspace(1e-4, 0.02, 1000))
    assert float(sched["abar"][0]) == 1.0
    np.testing.assert_allclose(
        float(sched["abar"][1000]),
        float(np.prod(1.0 - np.linspace(1e-4, 0.02, 1000))),
        rtol=2e-6,
    )


def test_forward_process_closed_form():
    sched = schedule.make_schedule()
    x0 = jnp.asarray(data.preprocess(np.full((2, 28, 28), 100, np.uint8)))
    t = jnp.array([1, 1000])
    key = jax.random.PRNGKey(0)

    x_t, eps = process.forward_noise(key, x0, t, sched)

    assert eps.shape == x0.shape
    a = sched["sqrt_abar"][t][:, None, None]
    b = sched["sqrt_1m_abar"][t][:, None, None]
    np.testing.assert_allclose(
        np.asarray(x_t), np.asarray(a * x0 + b * eps), rtol=1e-6, atol=1e-6
    )


# ---- EMA and checkpoint -----------------------------------------------------

def test_ema_update_formula():
    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([[3.0]])}
    updated = {"a": jnp.array([3.0, 0.0]), "b": jnp.array([[6.0]])}

    e0 = ema.ema_init(params)
    e1 = ema.ema_update(e0, updated)

    np.testing.assert_allclose(np.asarray(e1["a"]), [0.999 * 1 + 0.001 * 3, 0.999 * 2])
    np.testing.assert_allclose(np.asarray(e1["b"]), [[0.999 * 3 + 0.001 * 6]])


def test_checkpoint_roundtrip(tmp_path):
    config = model.UNetConfig()
    params = model.init_unet(jax.random.PRNGKey(0), config)
    saved_ema = ema.ema_update(ema.ema_init(params), params)
    settings = {"case": "mnist", "size": 32, "t_steps": 1000, "beta": [0.0001, 0.02]}

    path = tmp_path / "checkpoint.npz"
    checkpoint.save_checkpoint(path, config, saved_ema, settings)
    config2, loaded_ema, settings2 = checkpoint.load_checkpoint(path)

    assert settings2 == settings
    assert config2.channels == config.channels
    assert config2.image_size == config.image_size
    for a, b in zip(jax.tree_util.tree_leaves(saved_ema), jax.tree_util.tree_leaves(loaded_ema)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)


# ---- Reverse grid ------------------------------------------------------------

@pytest.mark.parametrize("steps", [2, 50, 1000])
def test_reverse_grid_endpoints_and_network_evaluations(steps):
    grid = sampler.reverse_grid(steps)

    assert grid[0] == 1000
    assert grid[-1] == 0
    assert len(grid) == steps + 1
    assert (np.diff(grid) < 0).all()      # visited in reverse
    assert len(grid) - 1 == steps          # one network evaluation per step


def test_reverse_grid_1000_steps_visits_every_level_then_zero():
    grid = sampler.reverse_grid(1000)
    assert np.array_equal(grid, np.concatenate([np.arange(1000, 0, -1), [0]]))


def test_reverse_grid_2_steps_is_jump_to_level1_then_zero():
    assert np.array_equal(sampler.reverse_grid(2), np.array([1000, 1, 0]))


# ---- Effective alpha/beta on jumps ------------------------------------------

def test_effective_alpha_beta_matches_definition():
    sched = schedule.make_schedule()
    a, b = sampler.effective_alpha_beta(sched, 1000, 1)

    np.testing.assert_allclose(
        float(a), float(sched["abar"][1000] / sched["abar"][1]), rtol=1e-6
    )
    np.testing.assert_allclose(float(b), 1.0 - float(a), rtol=1e-6)


def test_effective_beta_adjacent_equals_beta_t():
    sched = schedule.make_schedule()
    for t in (1000, 500, 2):
        a, b = sampler.effective_alpha_beta(sched, t, t - 1)
        np.testing.assert_allclose(float(b), float(sched["betas"][t]), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(float(a), float(sched["alphas"][t]), rtol=1e-5)


def test_effective_alpha_to_zero_uses_abar_t():
    sched = schedule.make_schedule()
    a, b = sampler.effective_alpha_beta(sched, 700, 0)
    np.testing.assert_allclose(float(a), float(sched["abar"][700]), rtol=1e-6)
    np.testing.assert_allclose(float(b), 1.0 - float(sched["abar"][700]), rtol=1e-6)


# ---- DDPM step: variance and final-step omission ----------------------------

def test_ddpm_step_adds_variance_effective_beta_off_final_step():
    sched = schedule.make_schedule()
    x = jnp.zeros((1, 4, 4, 1))
    eps_pred = jnp.zeros_like(x)
    t, s = 1000, 1
    n_trials = 20000

    keys = jax.random.split(jax.random.PRNGKey(0), n_trials)
    outs = jax.vmap(lambda k: sampler.ddpm_step(x, t, s, sched, eps_pred, k))(keys)
    var = jnp.var(outs, axis=0)

    _, beta_eff = sampler.effective_alpha_beta(sched, t, s)
    np.testing.assert_allclose(np.asarray(var), float(beta_eff), rtol=0.05, atol=1e-4)


def test_ddpm_step_final_has_no_noise_and_equals_predicted_x0():
    sched = schedule.make_schedule()
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(2, 8, 8, 1)).astype(np.float32))
    eps_pred = jnp.asarray(rng.normal(size=(2, 8, 8, 1)).astype(np.float32))

    out1 = sampler.ddpm_step(x, 1, 0, sched, eps_pred, jax.random.PRNGKey(1))
    out2 = sampler.ddpm_step(x, 1, 0, sched, eps_pred, jax.random.PRNGKey(2))

    # no fresh noise on the final transition: both keys give the same output
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(out1),
        np.asarray(sampler.predicted_x0(x, 1, sched, eps_pred)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_predicted_x0_matches_sheet_formula():
    sched = schedule.make_schedule()
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(3, 8, 8, 1)).astype(np.float32))
    eps = jnp.asarray(rng.normal(size=(3, 8, 8, 1)).astype(np.float32))
    t_arr = jnp.full((3,), 500, dtype=jnp.int32)

    expected = (x - sched["sqrt_1m_abar"][500] * eps) / sched["sqrt_abar"][500]
    np.testing.assert_allclose(
        np.asarray(sampler.predicted_x0(x, 500, sched, eps)),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(sampler.predicted_x0(x, t_arr, sched, eps)),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )


# ---- Serialization -----------------------------------------------------------

def test_sample_rows_clip_range_and_dimensions():
    x = np.linspace(-2.0, 2.0, 2048, dtype=np.float32).reshape(2, 32, 32)

    rows = io.sample_rows(x)

    assert len(rows) == 2
    assert rows[0]["i"] == 0 and rows[1]["i"] == 1
    xs = np.asarray(rows[1]["x"], dtype=np.float64)
    assert xs.shape == (1024,)
    assert xs.min() >= -1.0 and xs.max() <= 1.0
    clipped = np.clip(x, -1.0, 1.0)
    np.testing.assert_allclose(xs, np.round(clipped[1].reshape(-1), 3), atol=0)


def test_sample_rows_rounds_to_three_decimals():
    x = np.zeros((1, 32, 32), np.float32)
    x[0, 0, 0] = 0.1234
    x[0, 0, 1] = 0.1236
    x[0, 0, 2] = -0.1234

    rows = io.sample_rows(x)

    assert rows[0]["x"][0] == 0.123
    assert rows[0]["x"][1] == 0.124
    assert rows[0]["x"][2] == -0.123


# ---- Trajectory ----------------------------------------------------------------

def test_trajectory_levels_match_sheet():
    levels = trajectory.trajectory_levels(1000)
    assert levels[0] == 1000
    assert levels[-1] == 40
    assert len(levels) == 25
    assert all(t % 40 == 0 for t in levels)


def test_recorder_frames_levels_shapes_and_final_image():
    n, size = 8, 32
    levels = trajectory.trajectory_levels(1000)
    rec = trajectory.TrajectoryRecorder(n, size, levels)
    rng = np.random.default_rng(0)

    for t in levels:
        x = rng.normal(size=(n, size, size, 1)).astype(np.float32)
        x0 = rng.normal(size=(n, size, size, 1)).astype(np.float32)
        rec.observe(t, x, x0)
    rec.final(np.full((n, size, size, 1), 0.5, np.float32))

    frames = rec.frames()
    assert [f["t"] for f in frames] == levels + [0]
    for f in frames:
        assert len(f["x"]) == n * size * size
        assert len(f["x0"]) == n * size * size
    last = frames[-1]
    np.testing.assert_allclose(last["x"], last["x0"], atol=0)
    np.testing.assert_allclose(last["x"], [0.5] * (n * size * size), atol=0)


def test_recorder_sample_major_row_major_ordering():
    n, size = 2, 32
    vals = np.arange(n * size * size, dtype=np.float32).reshape(n, size, size, 1)
    rec = trajectory.TrajectoryRecorder(n, size, [1000])

    rec.observe(1000, vals, vals + 0.25)
    frame = rec.frames()[0]

    # first 1024 entries are image 0 row-major, next 1024 are image 1
    assert frame["x"][0] == 0.0
    assert frame["x"][1023] == 1023.0
    assert frame["x"][1024] == 1024.0
    assert frame["x0"][1024] == pytest.approx(1024.25, abs=1e-3)


def test_recorder_rounds_to_three_decimals():
    n, size = 1, 32
    x = np.zeros((n, size, size, 1), np.float32)
    x[0, 0, 0, 0] = 0.1234
    x[0, 0, 1, 0] = 0.1236
    rec = trajectory.TrajectoryRecorder(n, size, [1000])

    rec.observe(1000, x, x)
    frame = rec.frames()[0]

    assert frame["x"][0] == 0.123
    assert frame["x"][1] == 0.124


# ---- UNet model -----------------------------------------------------------------

def test_trajectory_final_frame_is_clipped_saved_image():
    # The t = 0 frame must contain the final SAVED image (clipped to [-1, 1]),
    # matching samples.jsonl; intermediate frames may be unbounded.
    from diffusion import sample as sample_mod

    sched = schedule.make_schedule()
    size = 8

    def apply_fn(x, t):
        # a large constant prediction drives the run far outside [-1, 1]
        return jnp.full(x.shape, 10.0, dtype=x.dtype)

    rec = trajectory.TrajectoryRecorder(2, size, trajectory.trajectory_levels(1000))
    saved = sample_mod.run_trajectory(
        apply_fn, sched, jax.random.PRNGKey(0), size, 1000, rec
    )

    frames = rec.frames()
    assert [f["t"] for f in frames] == trajectory.trajectory_levels(1000) + [0]
    last = frames[-1]
    assert last["x"] == last["x0"]
    xs = np.asarray(last["x"])
    assert xs.shape == (2 * size * size,)
    assert xs.min() >= -1.0 and xs.max() <= 1.0
    np.testing.assert_allclose(xs, np.asarray(saved).reshape(-1), atol=1e-9)


def test_unet_forward_shape_and_finite():
    config = model.UNetConfig()
    params = model.init_unet(jax.random.PRNGKey(1), config)

    x = jnp.zeros((2, 32, 32, 1))
    t = jnp.array([5, 1000])
    y = model.unet_apply(params, config, x, t)

    assert y.shape == (2, 32, 32, 1)
    assert bool(np.isfinite(np.asarray(y)).all())