"""DDPM sampling from a trained checkpoint.

The reverse grid is levels 1 + round(linspace(0, 999, steps)) visited in
reverse followed by level 0. Final MNIST samples are clipped to [-1, 1], and
the trajectory's t = 0 frame holds that same saved final image in both x and
x0. With --trajectory a SEPARATE 8-sample, 1000-step DDPM run records frames
at t = 1000, 960, ..., 0; its key is the second half of the command seed's key
split, so the same seed reproduces both runs exactly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import load_checkpoint
from .io import sample_rows
from .model import unet_apply
from .sampler import ddpm_sample
from .schedule import make_schedule
from .trajectory import TrajectoryRecorder, trajectory_levels

TRAJECTORY_N = 8


def run_trajectory(apply_fn, sched, key, size: int, t_steps: int, recorder):
    """Separate 8-sample, 1000-step DDPM trajectory run.

    Intermediate frames keep the unbounded current image x and predicted x0;
    the t = 0 frame holds the final SAVED image (clipped to [-1, 1], matching
    samples.jsonl) in both x and x0.
    """
    x = ddpm_sample(apply_fn, sched, key, recorder.n, t_steps,
                    size=size, recorder=recorder)
    saved = np.clip(np.asarray(x), -1.0, 1.0)
    recorder.final(saved)
    return saved


def run_sampling(checkpoint_path, out_dir, sampler: str, steps: int, n: int,
                 seed: int, trajectory: bool = False) -> dict:
    if sampler != "ddpm":
        raise ValueError(
            f"sampler {sampler!r} is not implemented; use ddpm (ddim arrives in Part 3)"
        )
    config, ema, settings = load_checkpoint(checkpoint_path)
    t_steps = int(settings["t_steps"])
    beta = settings["beta"]
    size = int(settings["size"])
    sched = make_schedule(t_steps, beta[0], beta[1])

    apply_fn = jax.jit(
        lambda x, t: unet_apply(ema, config, x,
                                jnp.full((x.shape[0],), t, dtype=jnp.int32))
    )

    key = jax.random.PRNGKey(seed)
    key_main, key_traj = jax.random.split(key)

    t0 = time.time()
    x = ddpm_sample(apply_fn, sched, key_main, n, steps, size=size)
    sampling_seconds = time.time() - t0
    final = np.clip(np.asarray(x), -1.0, 1.0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "case": "mnist",
        "size": size,
        "t_steps": t_steps,
        "beta": [beta[0], beta[1]],
        "sampler": "ddpm",
        "sample_steps": steps,
        "n_samples": n,
        "seed": seed,
        "cases": ["mnist"],
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")

    with (out_dir / "samples.jsonl").open("w") as f:
        for row in sample_rows(final):
            f.write(json.dumps(row) + "\n")

    if trajectory:
        rec = TrajectoryRecorder(TRAJECTORY_N, size, trajectory_levels(t_steps))
        run_trajectory(apply_fn, sched, key_traj, size, t_steps, rec)
        with (out_dir / "trajectory.jsonl").open("w") as f:
            for frame in rec.frames():
                f.write(json.dumps(frame) + "\n")

    print(f"sampled {n} images with {steps} DDPM steps in "
          f"{sampling_seconds:.1f}s; wrote {out_dir}", flush=True)
    return run_info