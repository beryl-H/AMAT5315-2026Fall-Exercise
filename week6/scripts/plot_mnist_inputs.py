"""Plot 16 preprocessed MNIST training images and one image noised at four levels.

Run from week6/:

    uv run python scripts/plot_mnist_inputs.py

Writes evidence/mnist-inputs.png: a 4x4 grid of the first 16 preprocessed
training images with a dedicated narrow color-bar column, and a bottom row
showing one image drawn through the forward process at t = 0, 200, 500, 1000
with its own separate color bar. The script asserts that no two axes overlap
before saving.
"""

from __future__ import annotations

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

NOISE_LEVELS = [0, 200, 500, 1000]


def assert_no_overlapping_axes(fig) -> None:
    """Fail loudly if any two axes bounding boxes intersect."""
    boxes = [(ax, ax.get_window_extent(renderer=fig.canvas.get_renderer()))
             for ax in fig.axes]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i][1], boxes[j][1]
            if b1.overlaps(b2):
                raise RuntimeError(
                    f"axes {i} and {j} overlap: {b1} vs {b2}"
                )


def main() -> int:
    input_path = WEEK6 / "inputs" / "mnist.npz"
    images = data.load_training_images(input_path)  # (60000, 32, 32) in [-1, 1]
    sched = schedule.make_schedule()

    # GridSpec: 4 image columns + one dedicated narrow color-bar column for the
    # digit block, and a separate color bar cell for the noise row.
    fig = plt.figure(figsize=(11.0, 13.5))
    gs = fig.add_gridspec(
        nrows=5, ncols=5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.10],
        height_ratios=[1.0, 1.0, 1.0, 1.0, 0.75],
        left=0.04, right=0.90, top=0.94, bottom=0.03,
        wspace=0.10, hspace=0.14,
    )
    norm = Normalize(-1.0, 1.0)
    cmap = "gray"

    img_axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(4)]
    cax_digits = fig.add_subplot(gs[0:4, 4])

    for ax, img in zip([a for row in img_axes for a in row], images[:16]):
        ax.imshow(img, vmin=-1, vmax=1, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar_digits = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                               cax=cax_digits)
    cbar_digits.set_label("pixel value", fontsize=11, labelpad=10)

    noise_axes = [fig.add_subplot(gs[4, c]) for c in range(4)]
    cax_noise = fig.add_subplot(gs[4, 4])

    x0 = jnp.asarray(images[:1])
    key = jax.random.PRNGKey(2026)
    stats = {}
    for ax, t in zip(noise_axes, NOISE_LEVELS):
        x_t, _ = process.forward_noise(key, x0, jnp.array([t]), sched)
        x_t = np.asarray(x_t)[0]
        stats[t] = x_t
        ax.imshow(x_t, vmin=-1, vmax=1, cmap=cmap)
        ax.set_title(f"t = {t}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar_noise = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                              cax=cax_noise)
    cbar_noise.set_label("pixel value", fontsize=11, labelpad=10)

    fig.suptitle("MNIST training images after preprocessing (top) and one "
                 "image through the forward process (bottom)", fontsize=13)

    assert_no_overlapping_axes(fig)

    out = WEEK6 / "evidence" / "mnist-inputs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    print(f"preprocessed shape: {images.shape}, range "
          f"[{float(images.min()):.3f}, {float(images.max()):.3f}]")
    border = np.concatenate([images[0, :2].ravel(), images[0, -2:].ravel(),
                             images[0, :, :2].ravel(), images[0, :, -2:].ravel()])
    print(f"image 0 border: min {float(border.min()):.3f} max {float(border.max()):.3f}")
    for t, x_t in stats.items():
        print(f"t = {t:4d}: pixel mean {x_t.mean():+.3f}, std {x_t.std():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())