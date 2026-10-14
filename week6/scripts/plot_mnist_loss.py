"""Plot the formal MNIST training loss (logarithmic loss axis).

Run from week6/:

    uv run python scripts/plot_mnist_loss.py --loss <path-to-loss.jsonl>

Writes evidence/mnist-loss.png: loss against parameter-update count on a
logarithmic loss axis, per the Part 2 lecture requirements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK6 = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", required=True, help="path to loss.jsonl")
    parser.add_argument("--out", default=str(WEEK6 / "evidence" / "mnist-loss.png"))
    parser.add_argument("--label", default="MNIST")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.loss).read_text().splitlines()
            if line.strip()]
    steps = np.array([r["step"] for r in rows])
    losses = np.array([r["loss"] for r in rows])
    assert np.all(np.diff(steps) == 1), "loss steps must be consecutive"
    assert np.isfinite(losses).all(), "losses must be finite"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, losses, lw=1.0, label=args.label)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(f"{args.label} training loss ({steps[-1]} updates)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"wrote {args.out}: {len(rows)} steps, "
          f"loss {losses[0]:.6f} -> {losses[-1]:.6f}, "
          f"min {losses.min():.6f} at step {steps[losses.argmin()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())