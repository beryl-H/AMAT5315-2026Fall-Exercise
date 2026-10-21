"""Part 3: flow matching — velocity network, training, exact marginal
velocity and Euler sampling.

Path: x_s = (1-s) x0 + s z, s in [0,1], target velocity u = z - x0
(dxs/ds, with respect to s — no factor 1000 anywhere in the target or the
Euler update). Time features use phase 1000*s*omega_j, omega_j =
10000**(-j/16), 16 sines then 16 cosines — NO minus-one offset.

Exact marginal velocity (analytic derivation). Given x_s = x with
a = 1-s, each mixture component contributes x ~ N(a mu_k, V I) with
V = 0.09 a^2 + s^2. Responsibilities w_k = softmax of component
log-densities. Within component k:
    E[z | x, k] = (s/V)(x - a mu_k)
    E[x0 | x, k] = mu_k + (0.09 a / V)(x - a mu_k)
so the component velocity is
    v_k = E[z - x0 | x, k] = -mu_k + (s - 0.09 a)/V * (x - a mu_k)
and the marginal velocity is sum_k w_k v_k. Checks: at s=1 (a=0, V=1,
equal weights, sum mu_k = 0) v = x; at s=0 (a=1, V=0.09) v = -x.
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.diffusion import CENTRES, CLUSTER_STD, sample_target

HIDDEN_WIDTHS = (128, 128, 128)


def flow_features(s):
    """r(s): phase 1000*s*omega_j, 16 sines then 16 cosines (no -1 offset)."""
    s = jnp.asarray(s, dtype=jnp.float32)
    j = jnp.arange(16, dtype=jnp.float32)
    omega = 10000.0 ** (-j / 16.0)
    phase = (1000.0 * s)[..., None] * omega
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def flow_init_params(key):
    """Same architecture/initialization as the diffusion net: 34->128x3->2,
    Normal(1/sqrt(fan_in)) weights, zero biases. New independent draw."""
    widths = [2 + 32, *HIDDEN_WIDTHS, 2]
    params = []
    for fan_in, fan_out in zip(widths[:-1], widths[1:]):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (fan_in, fan_out), dtype=jnp.float32)
        params.append({"w": w / jnp.sqrt(jnp.float32(fan_in)),
                       "b": jnp.zeros(fan_out, dtype=jnp.float32)})
    return params


def _silu(u):
    return u / (1.0 + jnp.exp(-u))


def flow_predict(params, x, s):
    """v_theta(x, s): input concat(x, flow_features(s)), SiLU MLP, linear."""
    h = jnp.concatenate([x, flow_features(s)], axis=-1)
    for layer in params[:-1]:
        h = _silu(h @ layer["w"] + layer["b"])
    return h @ params[-1]["w"] + params[-1]["b"]


def flow_loss(params, x0, z, s):
    """Algorithm 4 line 9: mean over batch of 1/d ||v_theta - (z - x0)||^2."""
    xs = (1.0 - s)[:, None] * x0 + s[:, None] * z
    pred = flow_predict(params, xs, s)
    se = (pred - (z - x0)) ** 2
    return jnp.mean(0.5 * jnp.sum(se, axis=-1))


def train_flow(run_dir, seed=2029, updates=25000, batch_size=512,
               lr0=1e-3, lr_final=0.0, beta1=0.9, beta2=0.999, eps_adam=1e-8,
               checkpoint_updates=(0, 50, 25000), loss_every=1):
    """Algorithm 4 with Adam + cosine decay. Refuses to overwrite a
    completed run (metadata.json present). Saves init + requested stages,
    all loss records, metadata; raw arrays under run_dir (gitignored)."""
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_path):
        raise RuntimeError(f"refusing to overwrite completed run: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_checkpoint(step, params):
        leaves = []
        for p in params:
            leaves.append(np.asarray(p["b"]))
            leaves.append(np.asarray(p["w"]))
        np.savez(os.path.join(ckpt_dir, f"params_{step:05d}.npz"), *leaves)

    key = jax.random.PRNGKey(seed)
    params = flow_init_params(key)
    save_checkpoint(0, params)

    def compiled_step(params, m, v, key, eta, step):
        k_x0, k_z, k_s = jax.random.split(key, 3)
        x0 = sample_target(k_x0, batch_size)
        z = jax.random.normal(k_z, (batch_size, 2), dtype=jnp.float32)
        s = jax.random.uniform(k_s, (batch_size,), dtype=jnp.float32)
        loss, grads = jax.value_and_grad(flow_loss)(params, x0, z, s)
        m = jax.tree_util.tree_map(
            lambda mi, g: beta1 * mi + (1 - beta1) * g, m, grads)
        v = jax.tree_util.tree_map(
            lambda vi, g: beta2 * vi + (1 - beta2) * g * g, v, grads)
        mhat = jax.tree_util.tree_map(lambda mi: mi / (1 - beta1 ** step), m)
        vhat = jax.tree_util.tree_map(lambda vi: vi / (1 - beta2 ** step), v)
        new_params = jax.tree_util.tree_map(
            lambda p, mi, vi: p - eta * mi / (jnp.sqrt(vi) + eps_adam),
            params, mhat, vhat)
        return new_params, m, v, loss

    step_fn = jax.jit(compiled_step)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    losses = []
    start = time.time()
    for step in range(1, updates + 1):
        key, sub = jax.random.split(key)
        frac = jnp.float32(step) / jnp.float32(updates)
        eta = lr_final + 0.5 * (lr0 - lr_final) * (1.0 + jnp.cos(jnp.pi * frac))
        params, m, v, loss = step_fn(params, m, v, sub, eta, jnp.int32(step))
        losses.append(float(loss))
        if step in checkpoint_updates:
            save_checkpoint(step, params)
        if step % 1000 == 0:
            print(f"update {step}/{updates} loss {float(loss):.6f} "
                  f"({time.time()-start:.0f}s)", flush=True)

    np.savez(os.path.join(run_dir, "loss_history.npz"),
             steps=np.arange(1, updates + 1, dtype=np.int64),
             loss=np.asarray(losses, dtype=np.float32))
    meta = {
        "seed": seed, "updates": updates, "batch_size": batch_size,
        "optimizer": "Adam", "beta1": beta1, "beta2": beta2,
        "epsilon": eps_adam, "learning_rate": lr0,
        "learning_rate_decay": "cosine", "final_learning_rate": lr_final,
        "precision": "float32", "checkpoint_updates": list(checkpoint_updates),
        "wall_time_seconds": time.time() - start,
        "jax_version": jax.__version__,
        "final_loss": float(losses[-1]),
        "target": "u = z - x0 (d/ds, no factor 1000)",
        "feature_phase": "1000*s*10000**(-j/16), 16 sines then 16 cosines",
        "rng": "PRNGKey(2029); per-update split into (x0, z, s) keys",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return params


def exact_marginal_velocity(x, s, centres=CENTRES, cluster_std=CLUSTER_STD):
    """v(x, s) = sum_k w_k v_k with v_k = -mu_k + (s-0.09a)/V (x - a mu_k)."""
    a = 1.0 - s
    V = jnp.float32(cluster_std) ** 2 * a * a + s * s
    d = x[:, None, :] - a * centres[None]          # (N, K, 2)
    logp = -0.5 * (d ** 2).sum(-1) / V
    w = jax.nn.softmax(logp, axis=-1)
    v_k = -centres[None] + (s - 0.09 * a) / V * d  # (N, K, 2)
    return (w[..., None] * v_k).sum(axis=1)


def euler_sample(velocity, params, x1, times, record_states=False):
    """Algorithm 5: x_next = x - (s - s_next) * v(x, s) over a decreasing
    grid from 1 to 0. Original s values reach the predictor; no clipping.
    """
    import numpy as _np
    t = _np.asarray(times)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("times must be a 1-D grid with >= 2 entries")
    if not np.all(t[1:] < t[:-1]):
        raise ValueError("times must be strictly decreasing")
    if t[0] != 1.0:
        raise ValueError("times must start at s = 1")
    tj = jnp.asarray(t, dtype=jnp.float32)

    def body(x, i):
        s = tj[i]
        s_next = tj[i + 1]
        s_arr = jnp.full((x.shape[0],), s, dtype=jnp.float32)
        v = velocity(params, x, s_arr)
        x_next = x - (s - s_next) * v
        return x_next, x_next

    x0, xs = jax.lax.scan(body, x1, jnp.arange(len(tj) - 1))
    if record_states:
        return x0, tj, xs
    return x0
