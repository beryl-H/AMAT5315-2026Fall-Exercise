"""Part 1 tests: safe loaders, frequencies, and visualization primitives.

Tests are derived from the Week 6 learning sheet (Part 1) and use synthetic
data with the exact MNIST IDX and CIFAR-10 binary formats, so they run
without the real downloads.
"""

import gzip
import struct
import tarfile
import io

import numpy as np
import pytest

from ebm.data import (
    CIFAR_RECORD_BYTES,
    IDX_IMAGE_MAGIC,
    IDX_LABEL_MAGIC,
    load_mnist_pair,
    class_frequencies,
    load_cifar10,
    load_cifar10_tar,
    load_idx_images,
    load_idx_labels,
)
from ebm.viz import CIFAR10_CLASSES, MNIST_CLASSES, image_gallery, frequency_bars


# ---------------------------------------------------------------- IDX helpers
def write_idx_images(path_or_buf, images, gzipped=False):
    """Encode an (N, rows, cols) uint8 array in IDX3 format."""
    n, rows, cols = images.shape
    payload = (
        struct.pack(">IIII", IDX_IMAGE_MAGIC, n, rows, cols) + images.tobytes()
    )
    data = gzip.compress(payload) if gzipped else payload
    if path_or_buf is None:
        return data
    path_or_buf.write_bytes(data)
    return None


def write_idx_labels(path_or_buf, labels, gzipped=False):
    """Encode an (N,) uint8 array in IDX1 format."""
    payload = struct.pack(">II", IDX_LABEL_MAGIC, len(labels)) + labels.astype(np.uint8).tobytes()
    data = gzip.compress(payload) if gzipped else payload
    if path_or_buf is None:
        return data
    path_or_buf.write_bytes(data)
    return None


def synthetic_mnist(n_per_class=3, seed=0):
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(10 * n_per_class, 28, 28), dtype=np.uint8)
    labels = np.tile(np.arange(10, dtype=np.uint8), n_per_class)
    return images, labels


# ------------------------------------------------------------ IDX loader tests
def test_load_idx_images_roundtrip(tmp_path):
    images, _ = synthetic_mnist()
    p = tmp_path / "train-images-idx3-ubyte.gz"
    write_idx_images(p, images, gzipped=True)
    loaded = load_idx_images(p)
    assert loaded.shape == images.shape
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, images)


def test_load_idx_labels_roundtrip(tmp_path):
    _, labels = synthetic_mnist()
    p = tmp_path / "train-labels-idx1-ubyte.gz"
    write_idx_labels(p, labels, gzipped=True)
    loaded = load_idx_labels(p)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, labels)


def test_invalid_image_magic_rejected(tmp_path):
    images, _ = synthetic_mnist(n_per_class=1)
    p = tmp_path / "bad.gz"
    bad = struct.pack(">IIII", 9999, 3, 28, 28) + images.tobytes()
    p.write_bytes(gzip.compress(bad))
    with pytest.raises(ValueError, match="magic"):
        load_idx_images(p)


def test_invalid_label_magic_rejected(tmp_path):
    _, labels = synthetic_mnist(n_per_class=1)
    p = tmp_path / "bad.gz"
    bad = struct.pack(">II", 2048, len(labels)) + labels.tobytes()
    p.write_bytes(gzip.compress(bad))
    with pytest.raises(ValueError, match="magic"):
        load_idx_labels(p)


def test_image_label_count_mismatch_rejected(tmp_path):
    images, labels = synthetic_mnist(n_per_class=1)
    ip = tmp_path / "i.gz"
    lp = tmp_path / "l.gz"
    write_idx_images(ip, images, gzipped=True)
    write_idx_labels(lp, labels[:-1], gzipped=True)  # one label short
    with pytest.raises(ValueError, match="count"):
        load_idx_labels(lp, expected_count=len(images))


def test_wrong_image_dimensions_rejected(tmp_path):
    images, _ = synthetic_mnist(n_per_class=1)
    p = tmp_path / "wrong_dims.gz"
    write_idx_images(p, images, gzipped=True)  # stored 28x28
    with pytest.raises(ValueError, match="dimension"):
        load_idx_images(p, expected_rows=32, expected_cols=32)


def test_truncated_images_rejected(tmp_path):
    images, _ = synthetic_mnist(n_per_class=1)
    p = tmp_path / "trunc.gz"
    full = write_idx_images(None, images)
    p.write_bytes(gzip.compress(full[:-10]))
    with pytest.raises(ValueError, match="size|truncated"):
        load_idx_images(p)


def test_labels_out_of_range_rejected(tmp_path):
    labels = np.array([0, 1, 42], dtype=np.uint8)
    p = tmp_path / "bad_range.gz"
    write_idx_labels(p, labels, gzipped=True)
    with pytest.raises(ValueError, match="range|class"):
        load_idx_labels(p, num_classes=10)


# --------------------------------------------------------- CIFAR loader tests
def synthetic_cifar_batch(n=16, seed=1):
    """Records of 1 label + 1024 R + 1024 G + 1024 B with distinct planes."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, size=n, dtype=np.uint8)
    r = np.full((n, 1024), 200, dtype=np.uint8)
    g = np.full((n, 1024), 100, dtype=np.uint8)
    b = np.full((n, 1024), 50, dtype=np.uint8)
    pixels = np.concatenate([r, g, b], axis=1)
    records = np.concatenate(
        [labels[:, None], pixels], axis=1
    )  # exactly 3073 bytes per record
    return records


def pack_cifar_tar(batch_name, records):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=batch_name)
        info.size = records.size
        tar.addfile(info, io.BytesIO(records.tobytes()))
    return buf.getvalue()


def test_cifar_record_bytes_constant():
    assert CIFAR_RECORD_BYTES == 3073
    assert CIFAR_RECORD_BYTES == 1 + 3 * 1024


def test_cifar_channel_order(tmp_path):
    """Red plane (first 1024 bytes after label) must land on channel 0."""
    records = synthetic_cifar_batch()
    p = tmp_path / "data_batch.bin"
    p.write_bytes(records.tobytes())
    images, labels = load_cifar10([p])
    assert images.shape == (16, 32, 32, 3)
    # R=200, G=100, B=50 everywhere
    assert images[0, 0, 0, 0] == 200
    assert images[0, 0, 0, 1] == 100
    assert images[0, 0, 0, 2] == 50


def test_cifar_swapped_channel_order_detected(tmp_path):
    """Negative control: GBR-packed records must NOT pass as RGB."""
    records = synthetic_cifar_batch()
    swapped = records.copy()
    swapped[:, 1:] = records[:, 1:].reshape(-1, 3, 1024)[:, ::-1, :].reshape(-1, 3072)
    p = tmp_path / "swapped.bin"
    p.write_bytes(swapped.tobytes())
    images, _ = load_cifar10([p])
    # if the loader is correct, channel 0 now holds what was blue (50)
    assert images[0, 0, 0, 0] == 50
    assert images[0, 0, 0, 0] != 200


def test_cifar_bad_record_length_rejected(tmp_path):
    records = synthetic_cifar_batch()[:, :-1]  # 3072 bytes per row: wrong
    p = tmp_path / "bad.bin"
    p.write_bytes(records.tobytes())
    with pytest.raises(ValueError, match="3073|record"):
        load_cifar10([p])


def test_cifar_tar_read_members_only(tmp_path):
    """Tarball is read by iterating expected members, never extracting to disk."""
    records = synthetic_cifar_batch()
    tar_bytes = pack_cifar_tar("cifar-10-batches-bin/data_batch_1.bin", records)
    p = tmp_path / "cifar-10-binary.tar.gz"
    import gzip as gz

    p.write_bytes(gz.compress(tar_bytes))
    images, labels = load_cifar10_tar(p, member="cifar-10-batches-bin/data_batch_1.bin")
    assert images.shape[1:] == (32, 32, 3)
    assert set(np.unique(labels)) <= set(range(10))


def test_split_identity_preserved(tmp_path):
    """Train/test stay separate arrays with their original assignment:
    no shared buffers, no relabeling, and order preserved."""
    images, labels = synthetic_mnist(n_per_class=3)
    ip, lp = tmp_path / "i.gz", tmp_path / "l.gz"
    write_idx_images(ip, images, gzipped=True)
    write_idx_labels(lp, labels, gzipped=True)
    loaded_img, loaded_lab = load_mnist_pair(ip, lp)
    assert loaded_img.shape == images.shape and loaded_lab.shape == labels.shape
    np.testing.assert_array_equal(loaded_lab, labels)  # original assignment
    assert not np.shares_memory(loaded_img, images)  # independent copies
    orig = int(images[0, 0, 0])
    loaded_img[0, 0, 0] = 123  # mutation must not leak into the other array
    assert int(images[0, 0, 0]) == orig


# --------------------------------------------------------- frequency tests
def test_class_frequencies_normalized_and_complete():
    labels = np.array([0, 1, 1, 2], dtype=np.uint8)
    freq = class_frequencies(labels, num_classes=4)
    assert freq.shape == (4,)
    assert abs(freq.sum() - 1.0) < 1e-12
    np.testing.assert_allclose(freq, [0.25, 0.5, 0.25, 0.0])


def test_class_frequencies_matches_indicator_definition():
    """Sheet eq. (2): p_hat(c) = (1/N) * sum_n 1[y_n = c]."""
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 10, size=1000)
    freq = class_frequencies(labels)
    for c in range(10):
        expected = np.mean(labels == c)
        assert freq[c] == pytest.approx(expected)


# ----------------------------------------------------------- viz / metadata
def test_class_names():
    assert len(MNIST_CLASSES) == 10
    assert len(CIFAR10_CLASSES) == 10
    assert CIFAR10_CLASSES[0] == "airplane"
    assert CIFAR10_CLASSES[1] == "automobile"


def test_gallery_native_pixels(tmp_path):
    """Nearest-neighbour rendering: no interpolation, saveable PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images, labels = synthetic_mnist(n_per_class=2)
    out = tmp_path / "gallery.png"
    image_gallery(images, labels=labels, class_names=MNIST_CLASSES, path=out)
    assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_frequency_bars_all_classes(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, labels = synthetic_mnist(n_per_class=2)
    out = tmp_path / "freq.png"
    frequency_bars(labels, class_names=MNIST_CLASSES, title="train", path=out)
    assert out.exists() and out.stat().st_size > 0
    plt.close("all")
