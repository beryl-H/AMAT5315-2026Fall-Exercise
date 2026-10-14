"""Restricted Boltzmann machine for MNIST (Week 6 Part 2), JAX/CPU.

Definitions follow the learning sheet:
  E(v,h)  = -a.v - b.h - v W h                        (eq. 3)
  F(v)    = -a.v - sum_j softplus(b_j + (v W)_j)      (eq. 4)
  p(h|v)  = sigmoid(b + v W),  p(v|h) = sigmoid(a + h W^T)  (eq. 6-7)
CD surrogate = mean_batch[F(v0) - F(stop_gradient(vk))]       (eq. 10)

Key management: all stochasticity flows from one explicit PRNG key that the
caller splits and threads through; no hidden global state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

PIXEL_ON_THRESHOLD = 128  # v_i = 1 iff pixel >= 128 (127->0, 128->1)


def binarize(pixels):
    """Binarize uint8 pixels at half intensity: >=128 -> 1.0, else 0.0."""
    return (jnp.asarray(pixels) >= PIXEL_ON_THRESHOLD).astype(jnp.float32)


def init_params(key, *, n_visible: int, n_hidden: int, pixel_mean=None):
    """Initialize theta.

    Visible bias a = logit of the TRAIN pixel mean, clipped to
    [1e-3, 1-1e-3] before the logit (documented choice in SPEC.md: bounds
    keep logits finite for nearly-always-on/off pixels and use no held-out
    data). Hidden bias b = 0; W ~ N(0, 0.01^2).
    """
    kw, _ = jax.random.split(key)
    W = 0.01 * jax.random.normal(kw, (n_visible, n_hidden), dtype=jnp.float32)
    b = jnp.zeros(n_hidden, dtype=jnp.float32)
    if pixel_mean is None:
        a = jnp.zeros(n_visible, dtype=jnp.float32)
    else:
        pm = jnp.clip(jnp.asarray(pixel_mean, dtype=jnp.float32), 1e-3, 1.0 - 1e-3)
        a = jnp.log(pm / (1.0 - pm))
    return {"a": a, "b": b, "W": W}


def energy(v, h, a, b, W):
    """RBM energy E(v,h) = -a.v - b.h - v W h (sheet eq. 3).

    Batched contract: v shape (..., n_visible), h shape (..., n_hidden);
    broadcasting batch dims applies, and the result has shape (...).
    """
    return -(v @ a) - (h @ b) - jnp.sum((v @ W) * h, axis=-1)


def p_h_given_v(v, b, W):
    return jax.nn.sigmoid(b + v @ W)


def p_v_given_h(h, a, W):
    return jax.nn.sigmoid(a + h @ W.T)


def free_energy(v, a, b, W):
    """Visible free energy F(v) (eq. 4); v shape (..., n_visible)."""
    return -(v @ a) - jnp.sum(jax.nn.softplus(b + v @ W), axis=-1)


def gibbs_step(v, params, key):
    """One block-Gibbs sweep: draw h ~ p(h|v), then v' ~ p(v|h)."""
    k1, k2 = jax.random.split(key)
    h = jax.random.bernoulli(k1, p_h_given_v(v, params["b"], params["W"]))
    h = h.astype(v.dtype)
    v_new = jax.random.bernoulli(
        k2, p_v_given_h(h, params["a"], params["W"])).astype(v.dtype)
    return v_new


def gibbs_chain(v0, params, key, k: int):
    """Run k block-Gibbs steps from v0; k=0 returns v0 unchanged."""
    def body(i, carry):
        v, key = carry
        key, sub = jax.random.split(key)
        return gibbs_step(v, params, sub), key
    vk, _ = jax.lax.fori_loop(0, k, body, (v0, key))
    return vk


def clamped_gibbs_chain(v0, params, key, k, clamp_values, clamp_mask):
    """Clamped block-Gibbs chain for image completion (sheet: inference).

    After every Gibbs sweep the clamped entries are reset: v = where(
    clamp_mask, clamp_values, v). clamp_mask/clamp_values broadcast against
    v0's shape. The clamped entries therefore leave p(v_M | v_O) invariant
    for the missing entries v_M.
    """
    def body(i, carry):
        v, key = carry
        key, sub = jax.random.split(key)
        v = gibbs_step(v, params, sub)
        v = jnp.where(clamp_mask, clamp_values, v)
        return v, key
    vk, _ = jax.lax.fori_loop(0, k, body, (v0, key))
    return vk


def cd_surrogate(params, v0, vk):
    """CD surrogate loss (eq. 10): mean F(v0) - F(sg(vk)).

    The stop_gradient holds the sampled endpoint fixed while autodiff
    differentiates with respect to theta.
    """
    f0 = free_energy(v0, params["a"], params["b"], params["W"])
    fk = free_energy(
        jax.lax.stop_gradient(vk), params["a"], params["b"], params["W"])
    return jnp.mean(f0 - fk)


def one_step_reconstruction_hamming(v, params, key):
    """One-step sampled reconstruction Hamming fraction per image (eq. sheet:
    fraction of changed pixels after one Gibbs step from v)."""
    v1 = gibbs_step(v, params, key)
    return jnp.mean(v1 != v, axis=-1)


def dice_ink(pred, truth):
    """Ink overlap on a pixel block: 2|A cap B| / (|A| + |B|); 0 if the sum
    of ink counts is zero (including both-blank), per sheet definition."""
    pred = np.asarray(pred).ravel() > 0.5
    truth = np.asarray(truth).ravel() > 0.5
    denom = pred.sum() + truth.sum()
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, truth).sum() / denom)
