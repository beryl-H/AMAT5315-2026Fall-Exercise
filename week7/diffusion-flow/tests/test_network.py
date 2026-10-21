"""Task 2 tests: diffusion.toml conformance and the noise predictor network."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import network
from src.diffusion import beta_schedule

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@pytest.fixture(scope="module")
def cfg():
    with open("diffusion.toml", "rb") as f:
        return tomllib.load(f)


def test_diffusion_toml_matches_lecture(cfg):
    m, tr = cfg["model"], cfg["training"]
    assert m["hidden_widths"] == [128, 128, 128]
    assert m["activation"] == "SiLU"
    assert m["time_features"] == 32
    assert m["time_frequency_base"] == 10_000.0
    assert m["time_offset"] == -1
    assert tr["updates"] == 25_000
    assert tr["batch_size"] == 512
    assert tr["optimizer"] == "Adam"
    assert (tr["beta1"], tr["beta2"], tr["epsilon"]) == (0.9, 0.999, 1e-8)
    assert tr["learning_rate"] == 0.001
    assert tr["learning_rate_decay"] == "cosine"
    assert tr["final_learning_rate"] == 0.0
    assert tr["precision"] == "float32"
    assert tr["seed"] == 2026
    assert tr["checkpoint_updates"] == [0, 50, 100, 200, 500, 1000, 2000,
                                        5000, 10000, 25000]


def test_time_features_hand_values():
    # omega_j = 10000**(-j/16); phase (t-1)*omega_j
    t = 1
    feats = np.asarray(network.time_features(jnp.array([t])))[0]
    for j in range(16):
        omega = 10000.0 ** (-j / 16.0)
        assert feats[j] == pytest.approx(math.sin(0.0), abs=1e-6)  # phase 0
        assert feats[16 + j] == pytest.approx(math.cos(0.0), abs=1e-6)
        _ = omega
    t = 11  # phase (11-1)*omega_j
    feats = np.asarray(network.time_features(jnp.array([11])))[0]
    for j in range(16):
        omega = 10000.0 ** (-j / 16.0)
        assert feats[j] == pytest.approx(math.sin(10.0 * omega), abs=1e-5)
        assert feats[16 + j] == pytest.approx(math.cos(10.0 * omega), abs=1e-5)


def test_init_and_predict_shapes_and_finiteness():
    params = network.init_params(jax.random.PRNGKey(0))
    assert len(params) == 4
    widths = [p["w"].shape for p in params]
    assert widths == [(34, 128), (128, 128), (128, 128), (128, 2)]
    for p in params:
        assert float(jnp.abs(p["b"]).max()) == 0.0
        assert bool(jnp.all(jnp.isfinite(p["w"])))
    x = jnp.zeros((5, 2), dtype=jnp.float32)
    t = jnp.array([1, 2, 3, 4, 5])
    out = network.predict(params, x, t)
    assert out.shape == (5, 2) and out.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(out)))


def test_init_weight_std_is_1_over_sqrt_fan_in():
    key = jax.random.PRNGKey(7)
    params = network.init_params(key)
    # First layer: std 1/sqrt(34)
    w0 = np.asarray(params[0]["w"])
    assert w0.std() == pytest.approx(1.0 / math.sqrt(34), rel=0.05)
    # Hidden layer: std 1/sqrt(128)
    w1 = np.asarray(params[1]["w"])
    assert w1.std() == pytest.approx(1.0 / math.sqrt(128), rel=0.05)


def test_predict_deterministic_per_t():
    params = network.init_params(jax.random.PRNGKey(1))
    x = jnp.ones((3, 2), dtype=jnp.float32)
    t = jnp.array([500, 500, 500])
    a = network.predict(params, x, t)
    b = network.predict(params, x, t)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Same x at different levels: time conditioning must change the output
    t2 = jnp.array([1, 500, 1000])
    assert not np.allclose(np.asarray(network.predict(params, x, t2)),
                           np.asarray(a))
