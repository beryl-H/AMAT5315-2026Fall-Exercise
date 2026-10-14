"""Noise schedule: linear beta from 0.0001 to 0.02 over 1000 levels.

Arrays are indexed by level 0..T with beta[0] = 0 and abar[0] = 1, matching the
Gaussian-mixture schedule of Part 1.
"""

from __future__ import annotations

import jax.numpy as jnp

T_STEPS = 1000
BETA_MIN = 1e-4
BETA_MAX = 0.02


def make_schedule(t_steps: int = T_STEPS,
                  beta_min: float = BETA_MIN,
                  beta_max: float = BETA_MAX) -> dict:
    betas = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.linspace(beta_min, beta_max, t_steps, dtype=jnp.float32),
    ])
    alphas = 1.0 - betas
    abar = jnp.cumprod(alphas)
    return {
        "betas": betas,
        "alphas": alphas,
        "abar": abar,
        "sqrt_abar": jnp.sqrt(abar),
        "sqrt_1m_abar": jnp.sqrt(jnp.maximum(1.0 - abar, 0.0)),
    }