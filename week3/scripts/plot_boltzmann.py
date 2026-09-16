#!/usr/bin/env python3
"""Part 1 Boltzmann-distribution verification plot for Week 3.

Reads the completed Part 1 Metropolis runs at T=3.0 and T=3.1, converts the
stored energy per site E to total energy E_total = E * 4096 (L = 64), builds
energy histograms with bins 40 total-energy units wide, and plots
ln(P_3.1(E) / P_3.0(E)) against total energy for every bin containing at
least 5 observations in BOTH histograms.

The theoretical Boltzmann prediction for the same quantity is a straight
line with slope (1/T0 - 1/T1) = 1/3.0 - 1/3.1 = 0.010752688... . Its
vertical intercept is arbitrary, so the line is anchored to the centroid
(mean x, mean y) of the measured points; only the anchor is chosen from
data, never the slope.

Output: week3/evidence/boltzmann.png

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/plot_boltzmann.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK3 = Path(__file__).resolve().parent.parent
RUNS = WEEK3 / "runs"
EVIDENCE = WEEK3 / "evidence"

L = 64
N = L * L  # 4096 sites
BIN_WIDTH = 40.0  # total-energy units per histogram bin
MIN_COUNT = 5  # required observations per bin in BOTH histograms
T0, T1 = 3.0, 3.1
SERIES = {T0: RUNS / "T3.0" / "series.jsonl", T1: RUNS / "T3.1" / "series.jsonl"}


def load_total_energies(path: Path) -> np.ndarray:
    """Read series.jsonl and return total energies E_total = E * N."""
    totals = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["L"] != L:
                raise ValueError(f"unexpected L={row['L']} in {path}")
            totals.append(row["E"] * N)
    return np.asarray(totals, dtype=float)


def main() -> None:
    energies = {t: load_total_energies(p) for t, p in SERIES.items()}

    # One shared bin grid spanning both histograms, so every bin edge and
    # width is identical and the count ratio equals the probability ratio.
    lo = math.floor(min(e.min() for e in energies.values()) / BIN_WIDTH) * BIN_WIDTH
    hi = math.ceil(max(e.max() for e in energies.values()) / BIN_WIDTH) * BIN_WIDTH
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    centers = edges[:-1] + BIN_WIDTH / 2

    counts = {
        t: np.histogram(e, bins=edges)[0] for t, e in energies.items()
    }

    # Keep bins with at least MIN_COUNT observations in BOTH histograms.
    keep = (counts[T0] >= MIN_COUNT) & (counts[T1] >= MIN_COUNT)
    x = centers[keep]
    # Same bin widths and same number of samples => P ratio = count ratio.
    y = np.log(counts[T1][keep].astype(float) / counts[T0][keep].astype(float))

    # Theoretical Boltzmann slope, fixed a priori; line anchored at the
    # centroid of the measured points (intercept only, slope untouched).
    slope = 1.0 / T0 - 1.0 / T1
    anchor_y = y.mean() - slope * x.mean()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out_png = EVIDENCE / "boltzmann.png"

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(x, y, "o", color="tab:blue", label="measured  ln(P$_{3.1}$/P$_{3.0}$)")
    xs = np.linspace(edges[0], edges[-1], 200)
    ax.plot(
        xs,
        anchor_y + slope * xs,
        "-",
        color="tab:red",
        label=(f"Boltzmann slope 1/{T0:g} - 1/{T1:g} = {slope:.9f}"),
    )
    ax.set_xlabel("total energy  $E$  (units of $J$)")
    ax.set_ylabel(r"$\ln\,\left(P_{3.1}(E)\,/\,P_{3.0}(E)\right)$")
    ax.set_title(
        f"Boltzmann verification, {L}x{L} Metropolis Ising\n"
        f"({len(x)} bins with >= {MIN_COUNT} counts in both runs)",
        fontsize=11,
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)

    print(f"samples: T={T0}: {len(energies[T0])}, T={T1}: {len(energies[T1])}")
    print(f"bin width: {BIN_WIDTH:g} total-energy units; bins kept: {len(x)}")
    print(f"theoretical slope: {slope:.9f}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
