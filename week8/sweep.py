"""Part 3 sweep: one worker trains one (width, seed); runner plan only.

Conventions: 1500 steps, batch 16, Part-2 schedule adapted to 1500
(warmup 100, cosine 3e-3 -> 3e-4, AdamW wd 0.1), seeds 1 and 2, widths
8..128. Workers run one per pinned core, at most 4 concurrent, total WSL
memory < 6 GB (RSS measured per worker). This module does NOT launch the
formal sweep; it provides build_worker_cmds for the audited runner plan.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import baselines as bl
import run_part2 as rp
import scaling as sc
import transformer as tf

STEPS = 1500
BATCH = 16
HEADS = 2


def train_one(width: int, seed: int, steps: int, train_ids: np.ndarray,
              VAL: np.ndarray, outdir: Path) -> dict:
    """Train one worker run; save checkpoint and metrics; return metrics."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.json"
    assert not metrics_path.exists(), f"refusing to overwrite run: {outdir}"
    master = jax.random.PRNGKey(seed)
    k_init, k_data = jax.random.split(master)
    params = tf.init_params(k_init, width=width, heads=HEADS, mlp=4 * width)
    state = tf.adamw_init(params)
    n_total = tf.count_params(params)
    n_nonemb = sc.nonembedding_params(width)
    t0 = time.time()
    log = []
    for step in range(1, steps + 1):
        k_data, sub = jax.random.split(k_data)
        windows = rp.tf.sample_windows(sub, train_ids, BATCH)
        lr = tf.lr_schedule_adapted(step, steps)
        params, state, loss = rp.train_step(params, state, windows, lr, step, True)
        if step % 100 == 0 or step == steps:
            val = rp.score_validation(params, VAL, n_blocks=rp.EVAL_BLOCKS)
            entry = {"step": step, "lr": lr, "train_loss": float(loss),
                     "val_loss_96blk": val["loss"], "elapsed_s": round(time.time() - t0, 1)}
            log.append(entry)
            with open(outdir / "train-log.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
    full = rp.score_validation(params, VAL)
    rp.save_params(params, outdir / "weights.npz")
    _, sha = rp.verify_saved_weights(params, outdir / "weights.npz")
    metrics = {"width": width, "seed": seed, "steps": steps, "heads": HEADS,
               "n_params_total": n_total, "n_params_nonembedding": n_nonemb,
               "full_heldout_loss": full["loss"], "full_heldout_ppl": full["ppl"],
               "n_scored": full["n_scored"], "train_seconds": round(time.time() - t0, 1),
               "checkpoint_sha256": sha, "log": log,
               "rss_mb_end": worker_rss_mb(os.getpid())}
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def build_worker_cmds(widths: list[int], seeds: list[int], attempt_dir: Path,
                      n_workers: int = 4, steps: int = STEPS) -> list[str]:
    """One pinned core per worker, at most `n_workers` (<= 4) concurrent.

    Returns the full ordered command list; the runner executes them with at
    most n_workers concurrent processes (e.g. `xargs -P`).
    """
    if n_workers > 4:
        raise ValueError("at most 4 concurrent workers are allowed")
    attempt_dir = Path(attempt_dir)
    cmds = []
    for w in widths:
        for s in seeds:
            d = attempt_dir / f"w{w:03d}-s{s}"
            cmds.append(f"taskset -c 0-{n_workers - 1} "
                        f"JAX_PLATFORMS=cpu uv run python sweep.py "
                        f"--width {w} --seed {s} --steps {steps} --out {d}")
    return cmds


def worker_rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    raise ValueError(f"no VmRSS for pid {pid}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    text = (Path(__file__).resolve().parent / "data" / "input.txt").read_text(encoding="utf-8")
    ids = bl.encode(text, bl.sorted_vocab(text))
    TRAIN, VAL = bl.train_validation_split(ids)
    m = train_one(args.width, args.seed, args.steps, TRAIN, VAL, Path(args.out))
    print(json.dumps({k: m[k] for k in ("width", "seed", "n_params_nonembedding",
                                        "full_heldout_loss", "train_seconds")}))


if __name__ == "__main__":
    main()
