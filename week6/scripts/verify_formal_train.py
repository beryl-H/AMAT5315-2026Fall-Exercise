"""Independent verifier for the formal MNIST training run after copy-back.

Run from week6/:

    uv run python scripts/verify_formal_train.py --run <retrieval-run-dir> \
        [--exit <run.exit.txt>] [--wall <run.wall-seconds.txt>]

Checks, without retraining:
  - exactly 30000 loss records, steps 1..30000 in order, all finite
  - the checkpoint loads with the exact formal settings (case mnist, size 32,
    t_steps 1000, beta [0.0001, 0.02], updates 30000, batch 64, seed 2026)
  - all EMA weight arrays are finite
  - optional markers: exit code 0, positive wall seconds

SHA256 equality between the remote and local copies is checked in the shell
(sha256sum on both sides) and recorded in cloud/README.md; it is not repeated
here. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="directory holding loss.jsonl and checkpoint.npz")
    parser.add_argument("--exit", dest="exit_file", default=None)
    parser.add_argument("--wall", dest="wall_file", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run)
    failures: list[str] = []

    def check(cond: bool, message: str) -> None:
        print(f"  {'ok' if cond else 'FAIL'}: {message}")
        if not cond:
            failures.append(message)

    rows = [json.loads(line) for line in
            (run_dir / "loss.jsonl").read_text().splitlines() if line.strip()]
    check(len(rows) == 30000, f"30000 loss records, got {len(rows)}")
    check([r["step"] for r in rows] == list(range(1, 30001)),
          "loss steps exactly 1..30000 in order")
    losses = [r["loss"] for r in rows]
    check(all(math.isfinite(v) for v in losses), "all losses finite")
    print(f"       first {losses[0]:.6f}, last {losses[-1]:.6f}, "
          f"min {min(losses):.6f} at step {losses.index(min(losses)) + 1}")

    import jax
    from diffusion.checkpoint import load_checkpoint
    config, ema, settings = load_checkpoint(run_dir / "checkpoint.npz")
    check(settings == {
        "case": "mnist", "size": 32, "t_steps": 1000, "beta": [0.0001, 0.02],
        "updates": 30000, "batch": 64, "seed": 2026, "learning_rate": 0.0002,
    }, f"checkpoint settings exact: {settings}")
    check(config.image_size == 32 and config.channels == (16, 32, 64),
          "checkpoint architecture matches the documented UNet")
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(ema)]
    check(all(np.isfinite(x).all() for x in leaves),
          f"all {len(leaves)} EMA weight arrays finite")

    if args.exit_file and Path(args.exit_file).exists():
        code = Path(args.exit_file).read_text().strip()
        check(code == "0", f"remote training exit code 0, got {code!r}")
    if args.wall_file and Path(args.wall_file).exists():
        wall = int(Path(args.wall_file).read_text().strip())
        check(wall > 0, f"wall seconds {wall} positive")

    if failures:
        print(f"\nverify_formal_train: {len(failures)} check(s) FAILED")
        return 1
    print("\nverify_formal_train: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())