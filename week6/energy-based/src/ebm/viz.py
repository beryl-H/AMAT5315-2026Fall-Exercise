"""Reusable visualization for Week 6 Part 1: galleries and class frequencies."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MNIST_CLASSES = [str(d) for d in range(10)]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def first_per_class(images: np.ndarray, labels: np.ndarray, per_class: int = 5,
                    *, num_classes: int = 10):
    """Deterministic gallery selection: the FIRST stored examples per class.

    Using stored order (not a hand-picked or "nice-looking" subset) avoids
    cherry-picking; documented in the README.
    """
    chosen_img, chosen_lab = [], []
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)[:per_class]
        chosen_img.append(images[idx])
        chosen_lab.append(labels[idx])
    return np.concatenate(chosen_img), np.concatenate(chosen_lab)


def image_gallery(images, labels=None, *, class_names=None, path,
                  ncols=10, title=None, cmap="gray"):
    """Render a gallery preserving native pixels (interpolation='nearest')."""
    images = np.asarray(images)
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    figsize = (1.2 * ncols, 1.35 * nrows + (0.5 if title else 0))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             squeeze=False, constrained_layout=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, ax in enumerate(axes.ravel()[:n]):
        img = images[i]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        if img.ndim == 3:  # RGB: show stored colors unscaled
            ax.imshow(img, interpolation="nearest")
        else:  # grayscale: fixed scale, not per-image autoscaling
            vmax = 255 if img.dtype == np.uint8 else 1
            ax.imshow(img, interpolation="nearest", cmap=cmap,
                      vmin=0, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if labels is not None:
            name = class_names[int(labels[i])] if class_names else str(int(labels[i]))
            ax.set_title(name, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def frequency_bars(labels, *, class_names, title, path, num_classes=10):
    """Bar chart of empirical class frequencies from ALL labels (sheet eq. 2)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes)
    freq = counts / labels.size
    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    ax.bar(range(num_classes), freq, color="tab:blue")
    ax.set_xticks(range(num_classes), class_names, rotation=45, ha="right")
    ax.set_ylabel(r"empirical frequency $\hat{p}(y=c)$")
    ax.set_title(title)
    ax.set_ylim(0, max(freq) * 1.25)
    for c, (f, n) in enumerate(zip(freq, counts)):
        ax.text(c, f, f"{f:.3f}\n(n={n})", ha="center", va="bottom", fontsize=6)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, freq, counts
