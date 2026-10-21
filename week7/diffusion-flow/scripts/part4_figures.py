"""Part 4 figures: four-row reverse-stage strip (native nearest pixels,
columns labeled with original t) and the final 4-image gallery, plus the
Week 6 real-CIFAR comparison. Uses ONLY saved generation arrays.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# comparison panel: real CIFAR-10 examples (deterministic first examples)
TAR = BASE.parent.parent / "week6/energy-based/inputs/cifar10/" \
    "cifar-10-binary.tar.gz"
gallery_png = BASE.parent.parent / "week6/energy-based/evidence/" \
    "cifar10_test_gallery.png"
real = None
if TAR.exists():
    import tarfile
    with tarfile.open(TAR) as tf:
        for m in tf.getmembers():
            if m.name.endswith("test_batch.bin"):
                data = tf.extractfile(m).read(3073 * 8)  # first 8 examples
                break
    recs = np.frombuffer(data, dtype=np.uint8).reshape(8, 3073)
    labels = recs[:, 0]
    imgs = recs[:, 1:].reshape(8, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    real = (imgs[:4], labels[:4])

if real is not None:
    imgs, labels = real
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"label {labels[i]}", fontsize=8)
    fig.suptitle("Real CIFAR-10 test examples (first 4, deterministic; "
                 "from the Week 6 original tarball)")
    fig.tight_layout()
    fig.savefig(EVID / "part4_real_cifar_comparison.png", dpi=130)
    plt.close(fig)
    print("real comparison panel written; Week 6 gallery exists:",
          gallery_png.exists())
else:
    print("CIFAR tarball not found; real comparison skipped")
print("part4 figures written")
