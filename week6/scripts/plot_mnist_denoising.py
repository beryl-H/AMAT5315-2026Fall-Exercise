"""Draw the denoising recording: current image above, predicted clean below.

Run from week6/:

    uv run python scripts/plot_mnist_denoising.py --trajectory <path-to-trajectory.jsonl>

Writes evidence/mnist-denoising.png: four samples (like the lecture figure) at
t = 1000, 800, 200, 0, with the current noisy image on the top row and the
predicted clean image on the bottom row. NOTE: the official evidence for the
course is the viewer (https://giggleliu.github.io/AMAT5315-2026Fall/week6-viewer.html)
loading trajectory.jsonl and saving a PNG at t = 200; this matplotlib figure is
a reproducible substitute, not viewer output.
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

FRAME_T = [1000, 800, 200, 0]
N_SAMPLES = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, help="path to trajectory.jsonl")
    parser.add_argument("--out", default=str(WEEK6 / "evidence" / "mnist-denoising.png"))
    args = parser.parse_args()

    frames = {f["t"]: f for f in
              (json.loads(line) for line in Path(args.trajectory).read_text().splitlines()
               if line.strip())}
    assert 0 in frames and frames[0]["x"] == frames[0]["x0"], \
        "the t = 0 frame must hold the final image in both arrays"
    missing = [t for t in FRAME_T if t not in frames]
    assert not missing, f"missing frames: {missing}"

    size = 32
    fig, axes = plt.subplots(2, len(FRAME_T), figsize=(3 * len(FRAME_T), 6.4))
    for j, t in enumerate(FRAME_T):
        x = np.asarray(frames[t]["x"], dtype=np.float64).reshape(-1, size, size)[:N_SAMPLES]
        x0 = np.asarray(frames[t]["x0"], dtype=np.float64).reshape(-1, size, size)[:N_SAMPLES]
        # one sample per column; sample-major flattening -> row j of the stack
        for row, arr, label in [(0, x, "current x"), (1, x0, "predicted x0")]:
            ax = axes[row, j]
            ax.imshow(arr[0], vmin=-1, vmax=1, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {t}")
            if j == 0:
                ax.set_ylabel(label)
    fig.suptitle("Denoising recording (matplotlib substitute for the course viewer; "
                 "viewer evidence at t = 200 is saved separately by the human)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    last = frames[0]
    print(f"wrote {args.out}: frames t = {FRAME_T}; final frame x == x0, "
          f"range [{min(last['x']):.3f}, {max(last['x']):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())