"""Independent verifier for the Part 2 smoke runs (no training, no sampling).

Run from week6/ after the smoke commands:

    uv run python scripts/verify_smoke.py

Checks the persisted smoke artifacts and full logs:
  - train: exit status 0, no traceback, exactly 200 loss records in step order,
    all losses finite
  - checkpoint: loads, EMA weights finite, settings fixed (seed 2026, batch 64,
    200 updates, mnist, size 32, t_steps 1000, beta [0.0001, 0.02])
  - sample: run.json contract fields; samples.jsonl = 4 rows, i = 0..3, each
    x has 1024 finite values in [-1, 1] rounded to 3 decimals
  - trajectory: 26 frames at t = 1000, 960, ..., 0; x and x0 of 8192 finite
    values each; the t = 0 frame has x == x0 inside [-1, 1] (the saved final
    image); intermediate frames may be unbounded but finite
  - all three smoke logs contain no Traceback

Exits non-zero on any failure. Smoke outputs are NOT formal training results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

WEEK6 = Path(__file__).resolve().parent.parent
RUNS = WEEK6 / "runs"

EXPECTED_RUN_JSON = {
    "case": "mnist",
    "size": 32,
    "t_steps": 1000,
    "beta": [0.0001, 0.02],
    "sampler": "ddpm",
    "sample_steps": 1000,
    "n_samples": 4,
    "seed": 1,
    "cases": ["mnist"],
}


def all_finite(values) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, message: str) -> None:
        print(f"  {'ok' if cond else 'FAIL'}: {message}")
        if not cond:
            failures.append(message)

    # --- training -----------------------------------------------------------
    log = (RUNS / "smoke" / "train.log").read_text()
    check("Traceback" not in log, "smoke train log has no traceback")
    check("Exit status: 0" in log, "smoke train exited 0")
    losses = [json.loads(line) for line in
              (RUNS / "smoke" / "loss.jsonl").read_text().splitlines() if line.strip()]
    check(len(losses) == 200, f"200 loss records, got {len(losses)}")
    check([r["step"] for r in losses] == list(range(1, 201)),
          "loss steps are exactly 1..200 in order")
    check(all_finite(r["loss"] for r in losses), "all losses finite")
    print(f"       first loss {losses[0]['loss']:.6f}, last loss {losses[-1]['loss']:.6f}")

    from diffusion.checkpoint import load_checkpoint
    config, ema, settings = load_checkpoint(RUNS / "smoke" / "checkpoint.npz")
    check(settings == {
        "case": "mnist", "size": 32, "t_steps": 1000, "beta": [0.0001, 0.02],
        "updates": 200, "batch": 64, "seed": 2026, "learning_rate": 0.0002,
    }, f"checkpoint settings fixed: {settings}")
    leaves = [np.asarray(x) for x in
              __import__("jax").tree_util.tree_leaves(ema)]
    check(all(np.isfinite(x).all() for x in leaves),
          f"all {len(leaves)} EMA weight arrays finite")

    # --- sampling -----------------------------------------------------------
    run = json.loads((RUNS / "smoke-samples" / "run.json").read_text())
    check(run == EXPECTED_RUN_JSON, f"run.json contract: {run}")
    rows = [json.loads(line) for line in
            (RUNS / "smoke-samples" / "samples.jsonl").read_text().splitlines()
            if line.strip()]
    check([r["i"] for r in rows] == [0, 1, 2, 3], "sample indices 0..3")
    check(all(len(r["x"]) == 1024 for r in rows), "each sample has 1024 values")
    check(all_finite(v for r in rows for v in r["x"]), "all sample values finite")
    check(all(-1.0 <= float(v) <= 1.0 for r in rows for v in r["x"]),
          "all sample values within [-1, 1]")
    check(all(float(v) == round(float(v), 3) for r in rows for v in r["x"]),
          "all sample values rounded to 3 decimals")

    # --- trajectory ---------------------------------------------------------
    tlog = (RUNS / "smoke-trajectory" / "sample.log").read_text()
    check("Traceback" not in tlog, "trajectory sample log has no traceback")
    frames = [json.loads(line) for line in
              (RUNS / "smoke-trajectory" / "trajectory.jsonl").read_text().splitlines()
              if line.strip()]
    expected_t = list(range(1000, 0, -40)) + [0]
    check([f["t"] for f in frames] == expected_t,
          f"26 trajectory frames at t = 1000, 960, ..., 0 (got {len(frames)})")
    check(all(len(f["x"]) == 8192 and len(f["x0"]) == 8192 for f in frames),
          "every frame has 8192 values per array")
    check(all_finite(v for f in frames for v in f["x"] + f["x0"]),
          "all trajectory values finite")
    last = frames[-1]
    check(last["t"] == 0 and last["x"] == last["x0"],
          "final frame: x == x0 at t = 0")
    check(all(-1.0 <= float(v) <= 1.0 for v in last["x"]),
          "final frame values are the clipped saved image (within [-1, 1])")
    inter = [f for f in frames if f["t"] != 0]
    out_of_range = sum(1 for f in inter for v in f["x"] + f["x0"]
                       if not (-1.0 <= float(v) <= 1.0))
    print(f"       intermediate frames: {out_of_range} unbounded values "
          f"(allowed); t=1000 x0 range "
          f"[{min(inter[0]['x0']):.3f}, {max(inter[0]['x0']):.3f}]")

    if failures:
        print(f"\nverify_smoke: {len(failures)} check(s) FAILED")
        return 1
    print("\nverify_smoke: all checks passed (smoke outputs are NOT formal results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())