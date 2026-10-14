"""Draw the 100 saved MNIST preview samples as a 10 by 10 grid.

Run from week6/:

    uv run python scripts/plot_mnist_grid.py --samples <path-to-samples.jsonl>

Writes evidence/mnist-grid.png per the Part 2 lecture requirements.
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
    parser.add_argument("--samples", required=True, help="path to samples.jsonl")
    parser.add_argument("--out", default=str(WEEK6 / "evidence" / "mnist-grid.png"))
    parser.add_argument("--grid", type=int, default=10)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.samples).read_text().splitlines()
            if line.strip()]
    n = args.grid * args.grid
    assert len(rows) >= n, f"need at least {n} samples, got {len(rows)}"
    rows = rows[:n]
    assert [r["i"] for r in rows] == list(range(n)), "sample indices must be 0..n-1"

    images = np.array([r["x"] for r in rows], dtype=np.float64).reshape(n, 32, 32)
    assert np.isfinite(images).all()
    assert images.min() >= -1.0 and images.max() <= 1.0

    fig, axes = plt.subplots(args.grid, args.grid, figsize=(10, 10))
    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img, vmin=-1, vmax=1, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"{n} DDPM samples from the trained MNIST noise predictor")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"wrote {args.out}: {n} samples, value range "
          f"[{images.min():.3f}, {images.max():.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())