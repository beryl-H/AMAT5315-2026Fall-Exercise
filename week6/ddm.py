"""Diffusion building blocks for Week 6 Part 1.

The module implements the forward noising process, the exact score of the
four-component Gaussian mixture, a neural noise predictor, the DDPM reverse
update, and DDPM sampling. It is deliberately small and formula-driven so each
piece can be checked against the mathematical identities in SPEC.md and the
learning sheet.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# ---- Part 1 fixed configuration -------------------------------------------

P1_NUM_LEVELS = 1000
P1_BETA_MIN = 1e-4
P1_BETA_MAX = 0.02
P1_CENTERS = jnp.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])
P1_DATA_VAR = 0.09
P1_DATA_STD = math.sqrt(P1_DATA_VAR)
P1_TIME_DIM = 64
P1_HIDDEN = (256, 256, 256, 256)


# ---- Schedule --------------------------------------------------------------

def make_schedule(num_levels: int = P1_NUM_LEVELS,
                  beta_min: float = P1_BETA_MIN,
                  beta_max: float = P1_BETA_MAX) -> dict[str, jax.Array]:
    """Return the noise schedule with arrays indexed by level 0..num_levels.

    betas[0] = 0 and betas[1:] = linspace(beta_min, beta_max, num_levels).
    alpha = 1 - beta; abar is the cumulative product of alpha; abar[0] = 1.
    """
    betas = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.linspace(beta_min, beta_max, num_levels, dtype=jnp.float32),
    ])
    alphas = 1.0 - betas
    abar = jnp.cumprod(alphas)
    sqrt_abar = jnp.sqrt(abar)
    sqrt_1m_abar = jnp.sqrt(jnp.maximum(1.0 - abar, 0.0))
    return {
        "betas": betas,
        "alphas": alphas,
        "abar": abar,
        "sqrt_abar": sqrt_abar,
        "sqrt_1m_abar": sqrt_1m_abar,
    }


# ---- Small broadcasting helpers --------------------------------------------

def _broadcast_level(t: Any, n: int) -> jax.Array:
    return jnp.broadcast_to(jnp.asarray(t), (n,))


def _as_broadcastable(v: Any, x: jax.Array) -> jax.Array:
    """Return a scalar or per-sample vector shaped to broadcast against x."""
    v = jnp.asarray(v, dtype=x.dtype)
    if v.ndim > 0:
        v = v.reshape(v.shape + (1,) * (x.ndim - v.ndim))
    return v


# ---- Target distribution and forward noising -------------------------------

def sample_mixture(key: jax.Array,
                   n: int,
                   centers: jax.Array = P1_CENTERS,
                   data_std: float = P1_DATA_STD) -> jax.Array:
    """Draw n clean samples from the equally weighted Gaussian mixture."""
    key_idx, key_noise = jax.random.split(key)
    idx = jax.random.randint(key_idx, (n,), 0, centers.shape[0])
    means = centers[idx]
    eps = jax.random.normal(key_noise, (n, centers.shape[1]))
    return means + data_std * eps


def noise_forward(key: jax.Array,
                  x0: jax.Array,
                  t: Any,
                  sched: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Return (x_t, eps) using x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps."""
    eps = jax.random.normal(key, x0.shape, dtype=x0.dtype)
    t_b = _broadcast_level(t, x0.shape[0])
    a = _as_broadcastable(sched["sqrt_abar"][t_b], x0)
    b = _as_broadcastable(sched["sqrt_1m_abar"][t_b], eps)
    x_t = a * x0 + b * eps
    return x_t, eps


# ---- Exact score of the noised mixture -------------------------------------

def _mixture_stats(x: jax.Array,
                   t: Any,
                   sched: dict[str, jax.Array],
                   centers: jax.Array,
                   data_var: float):
    """Return (dx, var, posterior_weights) for x against each component."""
    x = jnp.asarray(x, dtype=jnp.float32)
    centers = jnp.asarray(centers, dtype=jnp.float32)
    n = x.shape[0]
    k = centers.shape[0]
    t_b = _broadcast_level(t, n)

    abar = sched["abar"][t_b]
    sqrt_abar = sched["sqrt_abar"][t_b]
    var = abar * data_var + (1.0 - abar)                    # (n,)

    means = sqrt_abar[:, None, None] * centers[None, :, :]  # (n, k, 2)
    dx = x[:, None, :] - means                              # (n, k, 2)
    dx_sq = jnp.sum(dx * dx, axis=-1)                       # (n, k)

    log_w = jnp.log(jnp.full((k,), 1.0 / k, dtype=jnp.float32))
    log_phi = -jnp.log(2.0 * math.pi) - jnp.log(var[:, None]) - 0.5 * dx_sq / var[:, None]
    log_joint = log_phi + log_w[None, :]
    log_norm = jax.scipy.special.logsumexp(log_joint, axis=1, keepdims=True)
    weights = jnp.exp(log_joint - log_norm)                 # (n, k)

    return dx, var, weights


def exact_score(x: jax.Array,
                t: Any,
                sched: dict[str, jax.Array],
                centers: jax.Array = P1_CENTERS,
                data_var: float = P1_DATA_VAR) -> jax.Array:
    """Exact score of the noised Gaussian mixture at level(s) t."""
    dx, var, weights = _mixture_stats(x, t, sched, centers, data_var)
    comp_scores = -dx / var[:, None, None]                  # (n, k, 2)
    return jnp.sum(weights[:, :, None] * comp_scores, axis=1)


def log_mixture_density(x: jax.Array,
                        t: Any,
                        sched: dict[str, jax.Array],
                        centers: jax.Array = P1_CENTERS,
                        data_var: float = P1_DATA_VAR) -> jax.Array:
    """Log density of the noised Gaussian mixture at level(s) t."""
    x = jnp.asarray(x, dtype=jnp.float32)
    centers = jnp.asarray(centers, dtype=jnp.float32)
    n = x.shape[0]
    k = centers.shape[0]
    t_b = _broadcast_level(t, n)

    abar = sched["abar"][t_b]
    sqrt_abar = sched["sqrt_abar"][t_b]
    var = abar * data_var + (1.0 - abar)

    means = sqrt_abar[:, None, None] * centers[None, :, :]
    dx = x[:, None, :] - means
    dx_sq = jnp.sum(dx * dx, axis=-1)

    log_w = jnp.log(jnp.full((k,), 1.0 / k, dtype=jnp.float32))
    log_phi = -jnp.log(2.0 * math.pi) - jnp.log(var[:, None]) - 0.5 * dx_sq / var[:, None]
    log_joint = log_phi + log_w[None, :]
    return jax.scipy.special.logsumexp(log_joint, axis=1)


# ---- DDPM reverse process --------------------------------------------------

def reverse_mean(x_t: jax.Array,
                 t: Any,
                 sched: dict[str, jax.Array],
                 eps_pred: jax.Array) -> jax.Array:
    """Deterministic part of the DDPM reverse update (paper equation 5)."""
    t_b = _broadcast_level(t, x_t.shape[0])
    alpha_t = sched["alphas"][t_b]
    abar_t = sched["abar"][t_b]
    beta_t = sched["betas"][t_b]

    coef = _as_broadcastable(beta_t / jnp.sqrt(1.0 - abar_t), x_t)
    inv_sqrt_alpha = _as_broadcastable(1.0 / jnp.sqrt(alpha_t), x_t)
    return inv_sqrt_alpha * (x_t - coef * eps_pred)


def reverse_step(x_t: jax.Array,
                 t: Any,
                 sched: dict[str, jax.Array],
                 eps_pred: jax.Array,
                 key: jax.Array) -> jax.Array:
    """One stochastic DDPM reverse step; fresh noise is omitted when t == 1."""
    mean = reverse_mean(x_t, t, sched, eps_pred)
    t_b = _broadcast_level(t, x_t.shape[0])
    beta_t = sched["betas"][t_b]

    z = jax.random.normal(key, x_t.shape, dtype=x_t.dtype)
    add = _as_broadcastable((t_b > 1).astype(x_t.dtype), x_t)
    scale = _as_broadcastable(jnp.sqrt(beta_t), x_t)
    return mean + add * scale * z


# ---- Neural noise predictor ------------------------------------------------

def time_embedding(t: Any,
                   dim: int = P1_TIME_DIM,
                   num_levels: int = P1_NUM_LEVELS) -> jax.Array:
    """Sinusoidal embedding of the level ratio u = (t - 0.5) / num_levels."""
    t = jnp.asarray(t, dtype=jnp.float32)
    u = (t - 0.5) / num_levels
    half = dim // 2
    freqs = jnp.power(2.0, jnp.linspace(0.0, 9.0, half, dtype=jnp.float32))
    args = (2.0 * math.pi) * u[..., None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    bound = math.sqrt(3.0 / in_dim)
    W = jax.random.uniform(key, (in_dim, out_dim), minval=-bound, maxval=bound, dtype=jnp.float32)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"W": W, "b": b}


def _apply_linear(layer: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return x @ layer["W"] + layer["b"]


def init_predictor(key: jax.Array,
                   time_dim: int = P1_TIME_DIM,
                   hidden: tuple[int, ...] = P1_HIDDEN,
                   out_dim: int = 2) -> dict[str, Any]:
    """Initialise the MLP noise predictor's parameters."""
    keys = jax.random.split(key, len(hidden) + 1)
    layers = []
    in_dim = 2 + time_dim
    for i, units in enumerate(hidden):
        layers.append(_init_linear(keys[i], in_dim, units))
        in_dim = units
    layers.append(_init_linear(keys[len(hidden)], in_dim, out_dim))
    return {"mlp": layers, "time_dim": time_dim}


def predictor_apply(params: dict[str, Any], x: jax.Array, t: Any) -> jax.Array:
    """Evaluate the noise predictor eps_theta(x, t)."""
    te = time_embedding(t, params["time_dim"])
    h = jnp.concatenate([x, te], axis=-1)
    layers = params["mlp"]
    for layer in layers[:-1]:
        h = jax.nn.silu(_apply_linear(layer, h))
    return _apply_linear(layers[-1], h)


def noise_loss(params: dict[str, Any],
               x0: jax.Array,
               t: Any,
               key: jax.Array,
               sched: dict[str, jax.Array]) -> jax.Array:
    """Mean squared noise-prediction error over batch and coordinates."""
    x_t, eps = noise_forward(key, x0, t, sched)
    pred = predictor_apply(params, x_t, t)
    return jnp.mean((pred - eps) ** 2)


def ddpm_sample(params: dict[str, Any],
                sched: dict[str, jax.Array],
                key: jax.Array,
                n_samples: int,
                num_levels: int = P1_NUM_LEVELS) -> jax.Array:
    """Full 1000-step DDPM reverse process, no clipping."""
    x = jax.random.normal(key, (n_samples, 2), dtype=jnp.float32)

    def body(i, x):
        t = num_levels - i
        t_arr = jnp.full((n_samples,), t, dtype=jnp.int32)
        eps_pred = predictor_apply(params, x, t_arr)
        step_key = jax.random.fold_in(key, i + 1)
        return reverse_step(x, t_arr, sched, eps_pred, step_key)

    return jax.lax.fori_loop(0, num_levels, body, x)


# ---- Part 1 parameter serialisation ----------------------------------------

def init_p1_predictor(key: jax.Array) -> dict[str, Any]:
    return init_predictor(key, time_dim=P1_TIME_DIM, hidden=P1_HIDDEN, out_dim=2)


def p1_template_params() -> dict[str, Any]:
    """A parameter tree with the same structure as any Part 1 predictor."""
    return init_p1_predictor(jax.random.PRNGKey(0))


def save_params(params: dict[str, Any], path: str) -> None:
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez(path, **{f"arr_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)})


def load_params(path: str, template: dict[str, Any]) -> dict[str, Any]:
    flat, treedef = jax.tree_util.tree_flatten(template)
    data = np.load(path)
    leaves = []
    for i, template_leaf in enumerate(flat):
        leaf = jnp.asarray(data[f"arr_{i}"])
        leaves.append(leaf.astype(template_leaf.dtype))
    return jax.tree_util.tree_unflatten(treedef, leaves)