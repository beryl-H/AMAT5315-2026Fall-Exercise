"""Compare hand-written forward mode, hand-written reverse mode and a
centered finite difference against the analytic dU/dr of the Lennard-Jones
pair energy, on 601 equally spaced separations from 0.95 to 2.5.

Saves week5/artifacts/ad/modes.png (two panels: dU/dr vs r; absolute error
vs r on a log scale) and prints each method's maximum absolute error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lj_ad  # noqa: E402

R_MIN, R_MAX, N_POINTS = 0.95, 2.5, 601
H = 1e-6


def sample_separations() -> np.ndarray:
    return np.linspace(R_MIN, R_MAX, N_POINTS)


def analytic_du_dr(r: np.ndarray) -> np.ndarray:
    return -48.0 * r ** -13.0 + 24.0 * r ** -7.0


def lj_energy(r: np.ndarray) -> np.ndarray:
    return 4.0 * (r ** -12.0 - r ** -6.0)


def run_comparison() -> dict[str, float]:
    """Run all methods on the sample grid, plot, and return max abs errors."""
    r = sample_separations()
    ref = analytic_du_dr(r)

    forward = np.array(
        [lj_ad.forward_tangents(float(x), 1.0)[1]["U"] for x in r]
    )
    reverse = np.array([lj_ad.reverse_adjoints(float(x))[1]["r"] for x in r])
    finite_diff = (lj_energy(r + H) - lj_energy(r - H)) / (2.0 * H)

    err = {
        "forward": float(np.max(np.abs(forward - ref))),
        "reverse": float(np.max(np.abs(reverse - ref))),
        "finite_difference": float(np.max(np.abs(finite_diff - ref))),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(r, ref, "k-", lw=2, label="analytic")
    ax1.plot(r, forward, ".", ms=4, label="forward (tangents)")
    ax1.plot(r, reverse, "x", ms=4, label="reverse (adjoints)")
    ax1.plot(r, finite_diff, ".", ms=3, label=f"centered FD, h={H:g}")
    ax1.set_xlabel("separation r")
    ax1.set_ylabel("dU/dr")
    ax1.set_title("Lennard-Jones pair force, reduced units")
    ax1.legend()

    ax2.loglog(r, np.abs(forward - ref) + 1e-32, ".", ms=4, label="forward")
    ax2.loglog(r, np.abs(reverse - ref) + 1e-32, "x", ms=4, label="reverse")
    ax2.loglog(r, np.abs(finite_diff - ref), ".", ms=3, label="finite diff")
    ax2.set_xlabel("separation r")
    ax2.set_ylabel("|error| vs analytic")
    ax2.set_title("absolute error (log scale)")
    ax2.legend()

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "ad"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "modes.png", dpi=150)

    for name, value in err.items():
        print(f"max |error| {name:>16s}: {value:.3e}")
    print(f"wrote {out_dir / 'modes.png'}")
    return err


if __name__ == "__main__":
    run_comparison()
