"""Time-conditioned MLP noise predictor, exactly per diffusion.toml."""

import jax
import jax.numpy as jnp

HIDDEN_WIDTHS = (128, 128, 128)
TIME_FEATURES = 32
TIME_FREQUENCY_BASE = 10000.0
TIME_OFFSET = -1


def time_features(t):
    """r(t): 16 sine + 16 cosine features, phase (t-1)*omega_j.

    omega_j = 10000**(-j/16) for j = 0..15. t is an integer array of
    levels 1..1000 (or a scalar). Returns shape (..., 32).
    """
    t = jnp.asarray(t, dtype=jnp.float32)
    j = jnp.arange(16, dtype=jnp.float32)
    omega = TIME_FREQUENCY_BASE ** (-j / 16.0)
    phase = (t + TIME_OFFSET)[..., None] * omega  # (..., 16)
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def init_params(key):
    """Dense weights Normal(std 1/sqrt(input width)), zero biases.

    Layers: 34 -> 128 -> 128 -> 128 -> 2 (input is x[2] ++ features[32]).
    """
    widths = [2 + TIME_FEATURES, *HIDDEN_WIDTHS, 2]
    params = []
    for fan_in, fan_out in zip(widths[:-1], widths[1:]):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (fan_in, fan_out), dtype=jnp.float32)
        params.append({"w": w / jnp.sqrt(jnp.float32(fan_in)),
                       "b": jnp.zeros(fan_out, dtype=jnp.float32)})
    return params


def _silu(u):
    return u / (1.0 + jnp.exp(-u))


def predict(params, x, t):
    """eps_theta(x, t): input concat(x, r(t)), three SiLU layers, linear out."""
    h = jnp.concatenate([x, time_features(t)], axis=-1)
    for layer in params[:-1]:
        h = _silu(h @ layer["w"] + layer["b"])
    return h @ params[-1]["w"] + params[-1]["b"]
