"""Train the Part 1 noise predictor on the four-mode Gaussian mixture.

Run from week6/:

    uv run python scripts/train_p1.py

Writes artifacts/p1/params.npz, artifacts/p1/train-loss.jsonl and
artifacts/p1/run.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

WEEK6 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEEK6))

import ddm  # noqa: E402


def adam_init(params):
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }


def adam_update(state, params, grads, step, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state["v"], grads)
    c1 = 1.0 - b1**step
    c2 = 1.0 - b2**step
    mhat = jax.tree_util.tree_map(lambda x: x / c1, m)
    vhat = jax.tree_util.tree_map(lambda x: x / c2, v)
    params = jax.tree_util.tree_map(
        lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mhat, vhat
    )
    return {"m": m, "v": v}, params


def main() -> int:
    seed = 2026
    num_levels = ddm.P1_NUM_LEVELS
    n_updates = 25000
    batch = 512
    learning_rate = 1e-3

    out_dir = WEEK6 / "artifacts" / "p1"
    out_dir.mkdir(parents=True, exist_ok=True)

    sched = ddm.make_schedule()
    key = jax.random.PRNGKey(seed)
    key, init_key, train_key = jax.random.split(key, 3)

    params = ddm.init_p1_predictor(init_key)
    adam_state = adam_init(params)

    @jax.jit
    def train_step(params, adam_state, x0, t, noise_key, step):
        loss, grads = jax.value_and_grad(ddm.noise_loss, argnums=0)(
            params, x0, t, noise_key, sched
        )
        adam_state, params = adam_update(adam_state, params, grads, step, lr=learning_rate)
        return params, adam_state, loss

    losses = []
    t0 = time.time()
    for step in range(1, n_updates + 1):
        key, k_x, k_t, k_noise = jax.random.split(key, 4)
        x0 = ddm.sample_mixture(k_x, batch, ddm.P1_CENTERS, ddm.P1_DATA_STD)
        t = jax.random.randint(k_t, (batch,), minval=1, maxval=num_levels + 1)
        params, adam_state, loss = train_step(params, adam_state, x0, t, k_noise, step)
        losses.append(float(loss))
        if step % 1000 == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"step {step:5d}  loss {float(loss):.6f}  "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    final_loss = losses[-1]
    print(f"finished {n_updates} updates in {elapsed:.1f}s; final loss {final_loss:.6f}")

    ddm.save_params(params, str(out_dir / "params.npz"))

    with (out_dir / "train-loss.jsonl").open("w") as f:
        for i, loss in enumerate(losses, start=1):
            f.write(json.dumps({"step": i, "loss": loss}) + "\n")

    run_info = {
        "seed": seed,
        "num_levels": num_levels,
        "beta": [ddm.P1_BETA_MIN, ddm.P1_BETA_MAX],
        "updates": n_updates,
        "batch": batch,
        "learning_rate": learning_rate,
        "hidden": list(ddm.P1_HIDDEN),
        "time_dim": ddm.P1_TIME_DIM,
        "final_loss": final_loss,
        "elapsed_seconds": round(elapsed, 3),
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())