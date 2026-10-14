"""Exponential moving average of parameters (sheet equation 4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

DECAY = 0.999


def ema_init(params):
    return jax.tree_util.tree_map(jnp.asarray, params)


def ema_update(ema, params, decay: float = DECAY):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema, params
    )