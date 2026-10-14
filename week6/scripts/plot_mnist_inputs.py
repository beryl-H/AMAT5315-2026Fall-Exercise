"""Plot 16 preprocessed MNIST training images and one image noised at four levels.

Run from week6/:

    uv run python scripts/plot_mnist_inputs.py

Writes evidence/mnist-inputs.png: a 4x4 grid of the first 16 preprocessed
training images with a pixel-value color bar, and one image drawn through the
forward process at t = 0, 200, 500, 1000.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

WEEK6 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEEK6))

from diffusion import data, process, schedule  # noqa: E402


def main() -> int:
    input_path = WEEK6 / "inputs" / "mnist.npz"
    images = data.load_training_images(input_path)  # (60000, 32, 32) in [-1, 1]
    sched = schedule.make_schedule()

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(5, 5, width_ratios=[1, 1, 1, 1, 0.08],
                          hspace=0.25, wspace=0.12)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(5)]
    cax_top = fig.add_subplot(gs[0:4, 4])
    cax_bot = fig.add_subplot(gs[4, 4])
    norm = Normalize(-1.0, 1.0)
    cmap = "gray"

    for ax, img in zip([a for row in axes[:4] for a in row], images[:16]):
        ax.imshow(img, vmin=-1, vmax=1, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax_top,
                 label="pixel value")

    x0 = jnp.asarray(images[:1])
    noised = {}
    key = jax.random.PRNGKey(2026)
    for j, t in enumerate([0, 200, 500, 1000]):
        x_t, _ = process.forward_noise(key, x0, jnp.array([t]), sched)
        x_t = np.asarray(x_t)[0]
        noised[t] = x_t
        ax = axes[4][j]
        ax.imshow(x_t, vmin=-1, vmax=1, cmap=cmap)
        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax_bot,
                 label="pixel value")

    fig.suptitle("MNIST training images after preprocessing, and one image "
                 "through the forward process", y=0.995)
    out = WEEK6 / "evidence" / "mnist-inputs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # programmatic sanity output (recorded in README)
    print(f"wrote {out}")
    print(f"preprocessed shape: {images.shape}, range "
          f"[{float(images.min()):.3f}, {float(images.max()):.3f}]")
    border = np.concatenate([images[0, :2].ravel(), images[0, -2:].ravel(),
                             images[0, :, :2].ravel(), images[0, :, -2:].ravel()])
    print(f"image 0 border: min {float(border.min()):.3f} max {float(border.max()):.3f}")
    for t, x_t in noised.items():
        print(f"t = {t:4d}: pixel mean {x_t.mean():+.3f}, std {x_t.std():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())