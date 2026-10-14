"""Record dataset provenance: requested URL, observed effective URL (via a
fresh HEAD probe — never invented), byte size and SHA-256, fetch date.

The download commands themselves were run without --write-out logging, so the
effective URL recorded here is re-observed now with a HEAD request; if a probe
fails, effective_url is recorded as null rather than guessed.
"""

import hashlib
import json
import subprocess
import datetime
from pathlib import Path

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
CIFAR_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
CIFAR_CANONICAL_BYTES = 170052171  # official size used by root verification
MNIST_FILES = [
    "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_effective_url(url: str):
    """HEAD request; return the final effective URL, or None on failure."""
    try:
        out = subprocess.run(
            ["curl", "-sIL", "-o", "/dev/null", "--max-time", "20",
             "--write-out", "%{url_effective}", url],
            capture_output=True, text=True, timeout=25, check=True)
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def main():
    root = Path(__file__).resolve().parent.parent
    provenance = {
        "fetched": datetime.date.today().isoformat(),
        "effective_url_method": "fresh HEAD probe at provenance time "
                                "(original downloads did not log redirects)",
        "files": [],
    }
    for name in MNIST_FILES:
        p = root / "inputs/mnist" / name
        if p.exists():
            provenance["files"].append({
                "name": name,
                "requested_url": f"{MNIST_URL}/{name}",
                "effective_url": probe_effective_url(f"{MNIST_URL}/{name}"),
                "bytes": p.stat().st_size, "sha256": sha256(p),
            })
    p = root / "inputs/cifar10/cifar-10-binary.tar.gz"
    if p.exists():
        provenance["files"].append({
            "name": "cifar-10-binary.tar.gz",
            "requested_url": CIFAR_URL,
            "effective_url": probe_effective_url(CIFAR_URL),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            # Official author page https://www.cs.toronto.edu/~kriz/cifar.html
            # lists MD5 c32a1d4ab5d03f1284b67883e8d87530 (verified by root
            # against the fetched page; not re-asserted here if archive changed).
            "official_md5_from_author_page": "c32a1d4ab5d03f1284b67883e8d87530",
            "matches_official_size": p.stat().st_size == CIFAR_CANONICAL_BYTES,
        })
    out = root / "evidence/provenance.json"
    out.write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
