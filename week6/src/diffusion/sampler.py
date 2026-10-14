"""Reverse grid, effective alpha/beta jumps, and DDPM sampling.

Reverse grid (sheet Part 2): levels `1 + round(linspace(0, 999, steps))`
visited in reverse, followed by level 0. A jump t -> s uses
alpha_eff = abar_t / abar_s and beta_eff = 1 - alpha_eff; the added noise has
variance beta_eff and is omitted when s = 0 (abar_0 = 1).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def reverse_grid(steps: int, t_steps: int = 1000) -> np.ndarray:
    if not 2 <= steps <= t_steps:
        raise ValueError(f"steps must be in [2, {t_steps}], got {steps}")
    idx = np.round(np.linspace(0, t_steps - 1, steps)).astype(int)
    levels = 1 + idx
    grid = np.sort(levels)[::-1]
    return np.concatenate([grid, [0]])


def effective_alpha_beta(sched, t, s):
    """(alpha_eff, beta_eff) for a reverse jump from level t to level s."""
    abar = sched["abar"]
    alpha_eff = abar[t] / abar[s]
    beta_eff = 1.0 - alpha_eff
    return alpha_eff, beta_eff


def predicted_x0(x, t, sched, eps_pred):
    """Predicted clean image (sheet equation 6)."""
    t_b = jnp.broadcast_to(jnp.asarray(t), (x.shape[0],))
    a = sched["sqrt_abar"][t_b]
    b = sched["sqrt_1m_abar"][t_b]
    shape = a.shape + (1,) * (x.ndim - 1)
    return (x - b.reshape(shape) * eps_pred) / a.reshape(shape)


def ddpm_step(x, t, s, sched, eps_pred, key):
    """One stochastic reverse update from level t to level s (equation 5)."""
    alpha_eff, beta_eff = effective_alpha_beta(sched, t, s)
    abar_t = sched["abar"][t]
    mean = (x - (beta_eff / jnp.sqrt(1.0 - abar_t)) * eps_pred) / jnp.sqrt(alpha_eff)
    add_noise = (jnp.asarray(s) > 0).astype(x.dtype)
    z = jax.random.normal(key, x.shape, dtype=x.dtype)
    return mean + add_noise * jnp.sqrt(beta_eff) * z


def ddpm_sample(apply_fn, sched, key, n: int, steps: int, size: int = 32,
                recorder=None):
    """Run the full reverse process; apply_fn maps (x, t) -> eps prediction.

    Each reverse transition t -> s consumes fresh noise fold_in(key, t), so a
    seed reproduces the run exactly. When a recorder is given it observes the
    current image and predicted clean image at recorded levels, and the final
    image at t = 0.
    """
    t_steps = len(sched["betas"]) - 1
    grid = reverse_grid(steps, t_steps=t_steps)
    x = jax.random.normal(key, (n, size, size, 1))
    for t, s in zip(grid[:-1], grid[1:]):
        eps_pred = apply_fn(x, int(t))
        if recorder is not None:
            recorder.observe(int(t), x, predicted_x0(x, int(t), sched, eps_pred))
        x = ddpm_step(x, int(t), int(s), sched, eps_pred,
                      jax.random.fold_in(key, int(t)))
    if recorder is not None:
        recorder.final(x)
    return x