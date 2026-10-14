"""Part 1 evidence: load real MNIST + CIFAR-10, render galleries and
class-frequency plots, verify learning-sheet numbers, save summary JSON."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ebm.data import load_mnist_pair, class_frequencies, load_cifar10_tar
from ebm.viz import (CIFAR10_CLASSES, MNIST_CLASSES, first_per_class,
                     frequency_bars, image_gallery)

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"


def main():
    summary = {"mnist": {}, "cifar10": {}}

    # ------------------------------------------------------------- MNIST
    train_images, train_labels = load_mnist_pair(
        ROOT / "inputs/mnist/train-images-idx3-ubyte.gz",
        ROOT / "inputs/mnist/train-labels-idx1-ubyte.gz")
    test_images, test_labels = load_mnist_pair(
        ROOT / "inputs/mnist/t10k-images-idx3-ubyte.gz",
        ROOT / "inputs/mnist/t10k-labels-idx1-ubyte.gz")

    for split, imgs, labs in [("train", train_images, train_labels),
                              ("test", test_images, test_labels)]:
        assert imgs.shape[1:] == (28, 28) and imgs.dtype == np.uint8
        assert imgs.min() >= 0 and imgs.max() <= 255
        gi, gl = first_per_class(imgs, labs, per_class=5)
        image_gallery(gi, labels=gl, class_names=MNIST_CLASSES,
                      path=EV / f"mnist_{split}_gallery.png",
                      title=f"MNIST {split}: first 5 stored examples per class (native 28x28 pixels)")
        _, freq, counts = frequency_bars(
            labs, class_names=MNIST_CLASSES, title=f"MNIST {split} class frequencies (all labels)",
            path=EV / f"mnist_{split}_frequencies.png")
        summary["mnist"][split] = {
            "n_images": int(len(imgs)), "shape": list(imgs.shape[1:]),
            "dtype": str(imgs.dtype), "pixel_range": [int(imgs.min()), int(imgs.max())],
            "counts": counts.tolist(), "frequencies": freq.tolist(),
        }
        print(f"MNIST {split}: n={len(imgs)}, shape={imgs.shape[1:]}, "
              f"dtype={imgs.dtype}, range=[{imgs.min()},{imgs.max()}]")
        print(f"  counts: {dict(zip(MNIST_CLASSES, counts.tolist()))}")

    # VERIFY: digit 1 most common, digit 5 least common, both splits
    for split in ("train", "test"):
        freq = summary["mnist"][split]["frequencies"]
        assert int(np.argmax(freq)) == 1, f"MNIST {split}: digit 1 not most common"
        assert int(np.argmin(freq)) == 5, f"MNIST {split}: digit 5 not least common"
    assert summary["mnist"]["train"]["n_images"] == 60000
    assert summary["mnist"]["test"]["n_images"] == 10000
    print("MNIST VERIFY ok: 60000/10000, 28x28 grayscale uint8 0-255, "
          "digit 1 most common and 5 least common in both splits")

    # ------------------------------------------------------------- CIFAR-10
    train_batches = [load_cifar10_tar(
        ROOT / "inputs/cifar10/cifar-10-binary.tar.gz",
        member=f"cifar-10-batches-bin/data_batch_{i}.bin") for i in range(1, 6)]
    test_batch = load_cifar10_tar(
        ROOT / "inputs/cifar10/cifar-10-binary.tar.gz",
        member="cifar-10-batches-bin/test_batch.bin")

    cifar = {"train": (np.concatenate([b[0] for b in train_batches]),
                       np.concatenate([b[1] for b in train_batches])),
             "test": test_batch}
    for split, (imgs, labs) in cifar.items():
        assert imgs.shape[1:] == (32, 32, 3) and imgs.dtype == np.uint8
        gi, gl = first_per_class(imgs, labs, per_class=5)
        image_gallery(gi, labels=gl, class_names=CIFAR10_CLASSES,
                      path=EV / f"cifar10_{split}_gallery.png", cmap=None,
                      title=f"CIFAR-10 {split}: first 5 stored examples per class (native 32x32 RGB)")
        _, freq, counts = frequency_bars(
            labs, class_names=CIFAR10_CLASSES,
            title=f"CIFAR-10 {split} class frequencies (all labels)",
            path=EV / f"cifar10_{split}_frequencies.png")
        summary["cifar10"][split] = {
            "n_images": int(len(imgs)), "shape": list(imgs.shape[1:]),
            "counts": counts.tolist(), "frequencies": freq.tolist(),
        }
        print(f"CIFAR-10 {split}: n={len(imgs)}, shape={imgs.shape[1:]}")
        print(f"  counts: {dict(zip(CIFAR10_CLASSES, counts.tolist()))}")

    # VERIFY: 50000/10000; exactly 5000 train / 1000 test per class (0.1)
    assert summary["cifar10"]["train"]["n_images"] == 50000
    assert summary["cifar10"]["test"]["n_images"] == 10000
    assert all(c == 5000 for c in summary["cifar10"]["train"]["counts"])
    assert all(c == 1000 for c in summary["cifar10"]["test"]["counts"])
    assert all(abs(f - 0.1) < 1e-12 for f in summary["cifar10"]["train"]["frequencies"])
    print("CIFAR-10 VERIFY ok: 50000/10000, 32x32 RGB, exactly 5000 train / "
          "1000 test per class, every fraction 0.1")

    (EV / "part1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {EV/'part1_summary.json'}")


if __name__ == "__main__":
    main()
