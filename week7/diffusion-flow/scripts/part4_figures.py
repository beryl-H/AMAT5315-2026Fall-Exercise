"""Part 4 figures: four-row reverse-stage strip (native nearest pixels,
columns labeled with original t) and the final 4-image gallery, plus the
Week 6 real-CIFAR comparison. Uses ONLY saved generation arrays.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "artifacts/part4/generation"
EVID = BASE / "evidence"
STAGES = [1000, 800, 600, 400, 200, 100, 50, 20, 0]

states = np.load(RAW / "stage_states.npz")  # keys t1000..t0


def to_display(x_nchw):
    """Clean range [-1,1] -> [0,1], clipped ONLY for display."""
    x = np.transpose(x_nchw, (0, 2, 3, 1))
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


# four rows (images), columns = stages, native nearest pixels
fig, axes = plt.subplots(4, len(STAGES), figsize=(1.3 * len(STAGES), 5.6))
for j, lv in enumerate(STAGES):
    imgs = to_display(states[f"t{lv}"])
    for i in range(4):
        ax = axes[i, j]
        ax.imshow(imgs[i], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(f"t={lv}", fontsize=8)
fig.suptitle("DDPM reverse process, 4 images, pretrained CIFAR-10 predictor "
             "(1000 steps; states clipped to [0,1] for display only)")
fig.tight_layout()
fig.savefig(EVID / "part4_reverse_stages.png", dpi=130)
plt.close(fig)

# final gallery: all 4 generated images, no selection
final = to_display(states["t0"])
fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
for i, ax in enumerate(axes):
    ax.imshow(final[i], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"image {i}", fontsize=8)
fig.suptitle("4 generated images (seed 7401, 1000 DDPM steps, no selection)")
fig.tight_layout()
fig.savefig(EVID / "part4_generated_gallery.png", dpi=130)
plt.close(fig)

# comparison panel via the ACTUAL Week 6 gallery scripts (lecture requirement)
repo = Path(__file__).resolve().parents[3]
ebm = repo / "week6/energy-based"
sys.path.insert(0, str(ebm / "src"))
from ebm.data import load_cifar10_tar          # noqa: E402
from ebm.viz import CIFAR10_CLASSES, image_gallery  # noqa: E402

tar = ebm / "inputs/cifar10/cifar-10-binary.tar.gz"
if not tar.exists():
    raise SystemExit(f"Week 6 CIFAR tarball missing: {tar}")
imgs, labs = load_cifar10_tar(tar, member="cifar-10-batches-bin/test_batch.bin")
assert imgs.dtype == np.uint8 and imgs.shape[1:] == (32, 32, 3)
# Week 6 loader output is already native RGB NHWC uint8.
disp = imgs[:4]
image_gallery(disp, labels=labs[:4], class_names=CIFAR10_CLASSES,
              path=str(EVID / "part4_real_cifar_comparison.png"), ncols=4,
              cmap=None,
              title="Real CIFAR-10 test examples (first 4, deterministic; "
                    "Week 6 gallery scripts)")
print("week6 gallery comparison written; labels:", list(labs[:4]))
print("part4 figures written")
