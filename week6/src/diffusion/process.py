"""Forward noising: x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def forward_noise(key, x0, t, sched):
    """Return (x_t, eps) for a batch x0 (B, H, W) or (B, H, W, C)."""
    eps = jax.random.normal(key, x0.shape, dtype=x0.dtype)
    t_b = jnp.broadcast_to(jnp.asarray(t), (x0.shape[0],))
    a = sched["sqrt_abar"][t_b]
    b = sched["sqrt_1m_abar"][t_b]
    shape = a.shape + (1,) * (x0.ndim - 1)
    return a.reshape(shape) * x0 + b.reshape(shape) * eps, eps