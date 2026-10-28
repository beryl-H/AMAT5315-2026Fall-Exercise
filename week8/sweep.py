"""Part 3 sweep: one worker trains one (width, seed); runner plan only.

Conventions: 1500 steps, batch 16, Part-2 schedule adapted to 1500
(warmup 100, cosine 3e-3 -> 3e-4, AdamW wd 0.1), seeds 1 and 2, widths
8..128, heads=2 everywhere (explicit, never inferred from width). Workers
run one per pinned core, at most 4 concurrent, total WSL memory < 6 GB
(RSS measured live; scorer batches bounded at 16 windows).

Worker save protocol (part3-review-01 item 5):
  1. refuse ANY existing outdir (interrupted attempts are preserved);
  2. settings.json with true process start timestamp and source hashes
     BEFORE optimizer step 1;
  3. checkpoint + completion receipt BEFORE full-validation scoring, so
     scoring/plot failures never force retraining;
  4. verification result asserted, not discarded.
This module does NOT launch the formal sweep; build_worker_cmds provides
the executable argv list for the audited runner plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
SCORE_BATCH = 16  # windows per jit call in the scorer (review item 7)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {name: _sha(root / name) for name in
            ("transformer.py", "run_part2.py", "scaling.py", "sweep.py")}


def worker_rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    raise ValueError(f"no VmRSS for pid {pid}")


def total_worker_rss_mb(pids: list[int]) -> float:
    """Live total RSS of all worker pids; the runner aborts at 6 GB."""
    return float(sum(worker_rss_mb(p) for p in pids))


def train_one(width: int, seed: int, steps: int, train_ids: np.ndarray,
              VAL: np.ndarray, outdir: Path, heads: int = HEADS) -> dict:
    outdir = Path(outdir)
    assert not outdir.exists(), f"refusing ANY existing outdir: {outdir}"
    outdir.mkdir(parents=True)

    started_unix_ts = time.time()  # true process timestamp, not a guess
    is_smoke = any("smoke" in p for p in outdir.parts)
    settings = {"width": width, "seed": seed, "steps": steps, "heads": heads,
                "batch": BATCH, "score_batch": SCORE_BATCH,
                "started_unix_ts": started_unix_ts,
                "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_hashes": source_hashes(),
                "formal": not is_smoke}
    (outdir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    master = jax.random.PRNGKey(seed)
    k_init, k_data = jax.random.split(master)
    params = tf.init_params(k_init, width=width, heads=heads, mlp=4 * width)
    state = tf.adamw_init(params)
    # attention shape audit inside the worker's actual model/scoring config
    attn = tf.attention_weights(params, jnp.asarray(np.arange(128) % 65), heads=heads)
    assert attn.shape == (2, heads, 128, 128), attn.shape
    settings["attention_shape"] = list(attn.shape)
    (outdir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    n_total = tf.count_params(params)
    n_nonemb = sc.nonembedding_params(width)
    t0 = time.time()
    for step in range(1, steps + 1):
        k_data, sub = jax.random.split(k_data)
        windows = rp.tf.sample_windows(sub, train_ids, BATCH)
        lr = tf.lr_schedule_adapted(step, steps)
        params, state, loss = rp.train_step(params, state, windows, lr, step,
                                            True, heads=heads)
        if step % 100 == 0 or step == steps:
            val = rp.score_validation(params, VAL, n_blocks=rp.EVAL_BLOCKS,
                                      heads=heads, batch_size=SCORE_BATCH)
            entry = {"step": step, "lr": lr, "train_loss": float(loss),
                     "val_loss_96blk": val["loss"],
                     "elapsed_s": round(time.time() - t0, 1),
                     "rss_mb": worker_rss_mb(_getpid())}
            with open(outdir / "train-log.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    # checkpoint + completion receipt BEFORE scoring (review item 5)
    rp.save_params(params, outdir / "weights.npz")
    verified, sha = rp.verify_saved_weights(params, outdir / "weights.npz")
    assert verified, f"checkpoint verification failed for {outdir}"
    receipt = {"width": width, "seed": seed, "steps": steps,
               "n_params_total": n_total, "n_params_nonembedding": n_nonemb,
               "train_seconds": round(time.time() - t0, 1),
               "checkpoint_sha256": sha, "checkpoint_verified": verified,
               "finished_unix_ts": time.time(),
               "utc_finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (outdir / "completion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    full = rp.score_validation(params, VAL, heads=heads, batch_size=SCORE_BATCH)
    metrics = dict(receipt, full_heldout_loss=full["loss"],
                   full_heldout_ppl=full["ppl"], n_scored=full["n_scored"],
                   rss_mb_end=worker_rss_mb(_getpid()))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _getpid() -> int:
    import os
    return os.getpid()


def build_worker_cmds(widths: list[int], seeds: list[int], attempt_dir: Path,
                      n_workers: int = 4, steps: int = STEPS) -> list[str]:
    """Executable argv: one DISTINCT core per concurrent slot, max 4 workers.

    `env` sets JAX_PLATFORMS=cpu (a bare `JAX_PLATFORMS=cpu taskset ...`
    string is not executable); the venv python is used explicitly.
    """
    if n_workers > 4:
        raise ValueError("at most 4 concurrent workers are allowed")
    cmds = []
    for i, w in enumerate(widths):
        for s in seeds:
            d = Path(attempt_dir) / f"w{w:03d}-s{s}"
            core = len(cmds) % n_workers  # distinct core per concurrent slot
            cmds.append(f"env JAX_PLATFORMS=cpu taskset -c {core} "
                        f".venv/bin/python sweep.py --width {w} --seed {s} "
                        f"--steps {steps} --out {d}")
    return cmds


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
