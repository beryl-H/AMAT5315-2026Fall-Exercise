"""MNIST loading and preprocessing.

Training uses ALL training images from the npz; the test split is reserved and
never preprocessed into training data. Images are padded with two zero pixels
on each side FIRST, then scaled 0..255 -> -1..1, so the border is exactly -1.
Labels are never used for training.
"""

from __future__ import annotations

import numpy as np

PAD = 2


def load_splits(path):
    """Return (x_train, y_train, x_test, y_test) as NumPy arrays."""
    with np.load(path) as npz:
        x_train = np.asarray(npz["x_train"])
        y_train = np.asarray(npz["y_train"])
        x_test = np.asarray(npz["x_test"])
        y_test = np.asarray(npz["y_test"])
    return x_train, y_train, x_test, y_test


def preprocess(images):
    """Pad 2 zero pixels per edge, then scale 0..255 to -1..1 (float32)."""
    images = np.asarray(images)
    padded = np.pad(images, ((0, 0), (PAD, PAD), (PAD, PAD)),
                    mode="constant", constant_values=0)
    return padded.astype(np.float32) / 127.5 - 1.0


def load_training_images(path):
    """Load and preprocess ONLY the train split; the test split is reserved."""
    x_train, _, _, _ = load_splits(path)
    return preprocess(x_train)