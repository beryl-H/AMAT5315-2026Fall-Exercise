"""Algorithm 1: train the noise predictor on the four-cluster mixture.

Indexing convention (single, used everywhere):
- betas has length T=1000 and is indexed beta[t-1] for mathematical level t.
- abar has length T+1=1001 and is indexed abar[t] (abar[0] = 1).
- The network's time features use phase (t-1)*omega_j.
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from src import network
from src.diffusion import abar_from_betas, beta_schedule, sample_target

CHECKPOINT_UPDATES = (0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000)


def cosine_learning_rate(update, total, lr0=1e-3, lr_final=0.0):
    """eta_j decaying smoothly from lr0 (j=1) to lr_final (j=total)."""
    frac = jnp.float32(update) / jnp.float32(total)
    return lr_final + 0.5 * (lr0 - lr_final) * (1.0 + jnp.cos(jnp.pi * frac))


def batch_loss(params, x0, eps, t, abar_arr):
    """Mean squared noise-prediction error, factor 1/d with d = 2.

    x_t = sqrt(abar[t]) x0 + sqrt(1-abar[t]) eps (production convention).
    """
    a = abar_arr[t]
    xt = jnp.sqrt(a)[:, None] * x0 + jnp.sqrt(1.0 - a)[:, None] * eps
    pred = network.predict(params, xt, t)
    se = (pred - eps) ** 2
    return jnp.mean(0.5 * jnp.sum(se, axis=-1))


def train(run_dir, seed=2026, updates=25000, batch_size=512,
          lr0=1e-3, lr_final=0.0, betas1=0.9, betas2=0.999, eps_adam=1e-8,
          t_levels=1000, checkpoint_updates=CHECKPOINT_UPDATES,
          loss_every=1):
    """Run Algorithm 1; save checkpoints, loss history and metadata.

    Returns the final params. Raw arrays are written under run_dir
    (gitignored); small summaries are written by the caller.
    """
    os.makedirs(run_dir, exist_ok=True)
    betas = beta_schedule(t_levels)
    abar = abar_from_betas(betas)
    key = jax.random.PRNGKey(seed)

    params = network.init_params(key)

    losses = []
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_checkpoint(step, params):
        flat = jax.device_get(jax.tree_util.tree_leaves(params))
        np.savez(os.path.join(ckpt_dir, f"params_{step:05d}.npz"), *flat)

    save_checkpoint(0, params)

    def compiled_step(params, m, v, key, eta, step):
        k_x0, k_eps, k_t = jax.random.split(key, 3)
        x0 = sample_target(k_x0, batch_size)
        eps = jax.random.normal(k_eps, (batch_size, 2), dtype=jnp.float32)
        t = jax.random.randint(k_t, (batch_size,), 1, t_levels + 1)
        loss, grads = jax.value_and_grad(batch_loss)(params, x0, eps, t, abar)
        new_m = jax.tree_util.tree_map(
            lambda mi, g: betas1 * mi + (1 - betas1) * g, m, grads)
        new_v = jax.tree_util.tree_map(
            lambda vi, g: betas2 * vi + (1 - betas2) * g * g, v, grads)
        mhat = jax.tree_util.tree_map(
            lambda mi: mi / (1 - betas1 ** step), new_m)
        vhat = jax.tree_util.tree_map(
            lambda vi: vi / (1 - betas2 ** step), new_v)
        new_params = jax.tree_util.tree_map(
            lambda p, mi, vi: p - eta * mi / (jnp.sqrt(vi) + eps_adam),
            params, mhat, vhat)
        return new_params, new_m, new_v, loss

    step_fn = jax.jit(compiled_step)

    start = time.time()
    m = v = None
    for step in range(1, updates + 1):
        key, sub = jax.random.split(key)
        eta = cosine_learning_rate(step, updates, lr0, lr_final)
        if m is None:
            m = jax.tree_util.tree_map(jnp.zeros_like, params)
            v = jax.tree_util.tree_map(jnp.zeros_like, params)
        params, m, v, loss = step_fn(params, m, v, sub, eta, jnp.int32(step))

        if step % loss_every == 0 or step == updates:
            losses.append((step, float(loss)))
        if step in checkpoint_updates:
            save_checkpoint(step, params)
        if step % 1000 == 0:
            print(f"update {step}/{updates} loss {float(loss):.6f} "
                  f"({time.time()-start:.0f}s)", flush=True)

    np.savez(os.path.join(run_dir, "loss_history.npz"),
             steps=np.array([s for s, _ in losses], dtype=np.int64),
             loss=np.array([l for _, l in losses], dtype=np.float32))

    meta = {
        "seed": seed, "updates": updates, "batch_size": batch_size,
        "optimizer": "Adam", "beta1": betas1, "beta2": betas2,
        "epsilon": eps_adam, "learning_rate": lr0,
        "learning_rate_decay": "cosine", "final_learning_rate": lr_final,
        "precision": "float32", "t_levels": t_levels,
        "checkpoint_updates": list(checkpoint_updates),
        "loss_every": loss_every,
        "wall_time_seconds": time.time() - start,
        "jax_version": jax.__version__,
        "final_loss": float(loss),
        "abar_1000": float(abar[t_levels]),
        "indexing": "betas[t-1] (len T); abar[t] (len T+1); features (t-1)*omega",
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return params
