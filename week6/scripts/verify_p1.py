"""Standalone acceptance verifier for Week 6 Part 1.

Run from week6/:

    uv run python scripts/verify_p1.py

Reads the persisted raw artifacts and checks, without retraining:

  - generated samples are finite with shape (2000, 2)
  - exactly 25000 training loss steps were recorded
  - the training run used the fixed seed 2026, batch 512, updates 25000 and
    1000 noise levels
  - the mean relative L2 score error is below 0.15
  - all four mixture modes have generated support
  - the raw run artifacts are git-ignored

Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WEEK6 = Path(__file__).resolve().parent.parent
OUT = WEEK6 / "artifacts" / "p1"


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, message: str) -> None:
        if cond:
            print(f"  ok: {message}")
        else:
            print(f"  FAIL: {message}")
            failures.append(message)

    # 1. Generated samples: finite, shape (2000, 2).
    samples = np.load(OUT / "samples.npz")["x"]
    check(samples.shape == (2000, 2), f"samples shape (2000, 2), got {samples.shape}")
    check(bool(np.isfinite(samples).all()), "all generated samples finite")

    # 2. Exactly 25000 recorded training updates.
    with (OUT / "train-loss.jsonl").open() as f:
        losses = [json.loads(line) for line in f if line.strip()]
    check(len(losses) == 25000, f"25000 loss steps recorded, got {len(losses)}")

    # 3. Fixed training settings.
    run = json.loads((OUT / "run.json").read_text())
    check(run["seed"] == 2026, f"training seed 2026, got {run['seed']}")
    check(run["updates"] == 25000, f"updates 25000, got {run['updates']}")
    check(run["batch"] == 512, f"batch 512, got {run['batch']}")
    check(run["num_levels"] == 1000, f"num_levels 1000, got {run['num_levels']}")

    # 4. Mean relative L2 score error below the acceptance limit.
    m = json.loads((OUT / "measurements.json").read_text())
    mean = m["score_mean_relative_l2_error"]
    check(mean < 0.15, f"mean relative L2 error {mean:.6f} < 0.15")

    # 5. All four modes have generated support.
    counts = m["generated"]["nearest_mode_counts"]
    check(
        all(v > 0 for v in counts.values()),
        f"all four modes supported: {counts}",
    )

    # 6. Raw run artifacts are git-ignored.
    rel_paths = [
        "artifacts/p1/params.npz",
        "artifacts/p1/samples.npz",
        "artifacts/p1/target-samples.npz",
        "artifacts/p1/train-loss.jsonl",
        "artifacts/p1/measurements.json",
    ]
    proc = subprocess.run(
        ["git", "check-ignore", *rel_paths],
        cwd=str(WEEK6),
        capture_output=True,
        text=True,
    )
    ignored = set(proc.stdout.split())
    check(ignored == set(rel_paths), "raw run artifacts are git-ignored")

    if failures:
        print(f"\nverify_p1: {len(failures)} check(s) FAILED")
        return 1
    print("\nverify_p1: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())