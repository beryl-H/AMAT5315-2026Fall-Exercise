"""Safe dataset loaders for MNIST (IDX) and CIFAR-10 (binary), Week 6 Part 1.

All loaders validate magic numbers, counts, dimensions, dtypes and value
ranges before returning data, and never extract untrusted archives to disk
(tar members are read in memory by name).
"""

from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

import numpy as np

IDX_IMAGE_MAGIC = 2051
IDX_LABEL_MAGIC = 2049
CIFAR_RECORD_BYTES = 3073  # 1 label byte + 3 x 1024 channel planes (R, G, B)
MNIST_ROWS = 28
MNIST_COLS = 28
CIFAR_ROWS = 32
CIFAR_COLS = 32
NUM_CLASSES = 10


def _read_gzip_bytes(path: str | Path) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()


def load_idx_images(path: str | Path, *, expected_rows: int = MNIST_ROWS,
                    expected_cols: int = MNIST_COLS) -> np.ndarray:
    """Load a gzipped IDX3 image file with full validation."""
    buf = _read_gzip_bytes(path)
    if len(buf) < 16:
        raise ValueError(f"{path}: IDX image header truncated")
    magic, n, rows, cols = struct_unpack_image_header(buf[:16])
    if magic != IDX_IMAGE_MAGIC:
        raise ValueError(f"{path}: bad IDX image magic {magic}, expected {IDX_IMAGE_MAGIC}")
    if rows != expected_rows or cols != expected_cols:
        raise ValueError(
            f"{path}: unexpected image dimension {rows}x{cols}, "
            f"expected {expected_rows}x{expected_cols}"
        )
    expected_size = 16 + n * rows * cols
    if len(buf) != expected_size:
        raise ValueError(f"{path}: file size {len(buf)} != expected {expected_size} (truncated?)")
    images = np.frombuffer(buf, dtype=np.uint8, count=n * rows * cols, offset=16)
    return images.reshape(n, rows, cols).copy()


def load_idx_labels(path: str | Path, *, expected_count: int | None = None,
                    num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Load a gzipped IDX1 label file with full validation."""
    buf = _read_gzip_bytes(path)
    if len(buf) < 8:
        raise ValueError(f"{path}: IDX label header truncated")
    magic, n = struct_unpack_label_header(buf[:8])
    if magic != IDX_LABEL_MAGIC:
        raise ValueError(f"{path}: bad IDX label magic {magic}, expected {IDX_LABEL_MAGIC}")
    if expected_count is not None and n != expected_count:
        raise ValueError(f"{path}: label count {n} != image count {expected_count}")
    if len(buf) != 8 + n:
        raise ValueError(f"{path}: file size {len(buf)} != expected {8 + n} (truncated?)")
    labels = np.frombuffer(buf, dtype=np.uint8, count=n, offset=8).copy()
    if labels.size and labels.max() >= num_classes:
        raise ValueError(f"{path}: label value {labels.max()} outside class range 0..{num_classes - 1}")
    return labels


def load_mnist_pair(images_path: str | Path, labels_path: str | Path):
    """Load a matched (images, labels) MNIST split; validates pairing."""
    images = load_idx_images(images_path)
    labels = load_idx_labels(labels_path, expected_count=len(images))
    return images, labels


def load_cifar10(batch_paths, *, num_classes: int = NUM_CLASSES):
    """Load one or more CIFAR-10 binary batch files.

    Each record is 3073 bytes: 1 label byte followed by the red, green and
    blue 32x32 planes in row-major order.
    """
    all_images, all_labels = [], []
    for p in batch_paths:
        raw = Path(p).read_bytes()
        if len(raw) % CIFAR_RECORD_BYTES != 0:
            raise ValueError(
                f"{p}: size {len(raw)} is not a multiple of the {CIFAR_RECORD_BYTES}-byte record"
            )
        records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, CIFAR_RECORD_BYTES)
        labels = records[:, 0].copy()
        if labels.size and labels.max() >= num_classes:
            raise ValueError(f"{p}: label value {labels.max()} outside class range")
        # label | R plane | G plane | B plane  ->  (N, 32, 32, 3) RGB
        planes = records[:, 1:].reshape(-1, 3, CIFAR_ROWS, CIFAR_COLS)
        images = np.transpose(planes, (0, 2, 3, 1)).copy()  # N, H, W, C
        all_images.append(images)
        all_labels.append(labels)
    return np.concatenate(all_images), np.concatenate(all_labels)


def load_cifar10_tar(tar_path: str | Path, *, member: str,
                     num_classes: int = NUM_CLASSES):
    """Read one CIFAR-10 batch from a (possibly gzipped) tar archive.

    No extraction to disk: the named member is opened and read in memory.
    """
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, mode="r:*") as tar:
        names = {m.name: m for m in tar.getmembers() if m.isfile()}
        if member not in names:
            raise ValueError(f"{tar_path}: expected member {member!r} not found; "
                             f"have {sorted(names)[:5]}...")
        fh = tar.extractfile(names[member])
        raw = fh.read()
    if len(raw) % CIFAR_RECORD_BYTES != 0:
        raise ValueError(f"{tar_path}/{member}: size not a multiple of {CIFAR_RECORD_BYTES} bytes")
    records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, CIFAR_RECORD_BYTES)
    labels = records[:, 0].copy()
    if labels.size and labels.max() >= num_classes:
        raise ValueError(f"{tar_path}/{member}: label outside class range")
    planes = records[:, 1:].reshape(-1, 3, CIFAR_ROWS, CIFAR_COLS)
    images = np.transpose(planes, (0, 2, 3, 1)).copy()
    return images, labels


def class_frequencies(labels: np.ndarray, *, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Empirical class frequencies, sheet eq. (2): p_hat(c) = (1/N) sum 1[y=c]."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes)
    return counts / labels.size


# -- small helpers kept separate for testability ------------------------------
import struct


def struct_unpack_image_header(header: bytes):
    return struct.unpack(">IIII", header)


def struct_unpack_label_header(header: bytes):
    return struct.unpack(">II", header)
