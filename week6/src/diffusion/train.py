"""Training loop for the MNIST noise predictor (denoising score matching).

Documented design choices within the sheet's fixed budget:
- Adam, constant learning rate 2e-4 (the DDPM standard);
- uniform integer levels t in {1..1000}, MSE noise-prediction loss averaged
  over batch and pixels;
- EMA decay 0.999 applied every update (sheet equation 4);
- batches are drawn uniformly with replacement from ALL 60000 training images;
  the test split is never touched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from . import data
from .checkpoint import save_checkpoint
from .ema import ema_init, ema_update
from .model import UNetConfig, init_unet, unet_apply
from .process import forward_noise
from .schedule import BETA_MAX, BETA_MIN, T_STEPS, make_schedule


def _adam_init(params):
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }


def _adam_update(state, params, grads, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda x: x / (1.0 - b1**step), m)
    vhat = jax.tree_util.tree_map(lambda x: x / (1.0 - b2**step), v)
    params = jax.tree_util.tree_map(
        lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mhat, vhat
    )
    return {"m": m, "v": v}, params


def run_training(data_path, out_dir, updates: int, batch: int, seed: int,
                 t_steps: int = T_STEPS, beta=(BETA_MIN, BETA_MAX),
                 learning_rate: float = 2e-4) -> dict:
    config = UNetConfig()
    sched = make_schedule(t_steps, beta[0], beta[1])

    images = data.load_training_images(data_path)
    n_train = images.shape[0]
    x_train = jnp.asarray(images)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    params = init_unet(k_init, config)
    ema = ema_init(params)
    opt = _adam_init(params)

    def loss_fn(p, x0, t, eps):
        pred = unet_apply(p, config, x0[:, :, :, None], t)
        return jnp.mean((pred[:, :, :, 0] - eps) ** 2)

    @jax.jit
    def train_step(params, ema, opt, x0, t, eps, step):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(params, x0, t, eps)
        opt, params = _adam_update(opt, params, grads, step, learning_rate)
        ema = ema_update(ema, params)
        return params, ema, opt, loss

    loss_path = out_dir / "loss.jsonl"
    t0 = time.time()
    with loss_path.open("w") as f:
        for step in range(1, updates + 1):
            key, k_idx, k_t, k_eps = jax.random.split(key, 4)
            idx = jax.random.randint(k_idx, (batch,), 0, n_train)
            x0 = x_train[idx]
            t = jax.random.randint(k_t, (batch,), 1, t_steps + 1)
            key, k_eps = jax.random.split(key)
            x_t, eps = forward_noise(k_eps, x0, t, sched)
            params, ema, opt, loss = train_step(params, ema, opt, x_t, t, eps, step)
            f.write(json.dumps({"step": step, "loss": float(loss)}) + "\n")
            if step % 100 == 0:
                f.flush()
            if step == 1 or step % 500 == 0 or step == updates:
                elapsed = time.time() - t0
                print(f"step {step}/{updates} loss {float(loss):.6f} "
                      f"elapsed {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    settings = {
        "case": "mnist",
        "size": config.image_size,
        "t_steps": t_steps,
        "beta": [beta[0], beta[1]],
        "updates": updates,
        "batch": batch,
        "seed": seed,
        "learning_rate": learning_rate,
    }
    save_checkpoint(out_dir / "checkpoint.npz", config, ema, settings)
    print(f"finished {updates} updates in {elapsed:.1f}s; "
          f"final loss {float(loss):.6f}; wrote {out_dir / 'checkpoint.npz'}",
          flush=True)
    return settings