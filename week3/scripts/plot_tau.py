#!/usr/bin/env python3
"""Week 3 Part 3: integrated autocorrelation time vs temperature.

Draws tau_int of the per-spin |M| series against temperature for L = 32
and L = 64, using the completed Metropolis runs in week3/artifacts/:

- coarse runs (t_grid 1.5..3.5 step 0.1,  5000 measured sweeps)
  -> the full temperature range,
- window runs (t_grid 2.0..2.6 step 0.05, 100000 measured sweeps)
  -> the refined critical region.

tau_int uses exactly the Week 3 rule of scripts/errors.py:

    tau_int = 1/2 + sum rho(t),

with the sum stopped once the lag exceeds six times the running tau_int.
Error bars are jackknife estimates over 20 contiguous blocks: each
leave-one-block-out replica recomputes rho and re-applies the stopping
rule, so the bars include both statistical noise and the sensitivity of
the stopping rule.

Caveat (visible in the figure): for the coarse runs near T_c the block
length (250 sweeps) is comparable to or shorter than tau_int itself, so
both the point estimates and their error bars are truncated/optimistic
there; the window runs (5000-sweep blocks) are far more reliable.

Output: week3/evidence/tau.png (logarithmic vertical axis).

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/plot_tau.py
"""

from __future__ import annotations

import numpy as np

from errors import tau_int  # noqa: E402  (same scripts/ directory)
from plot_order_parameters import (  # noqa: E402  (same scripts/ directory)
    ARTIFACTS,
    COLORS,
    EVIDENCE,
    TC,
    load_series,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = (32, 64)
N_BLOCKS = 20


def tau_with_error(abs_m: np.ndarray) -> tuple[float, float]:
    """Week 3 tau_int and a jackknife error over N_BLOCKS blocks."""
    point = tau_int(abs_m)
    blocks = np.array_split(abs_m, N_BLOCKS)
    replicas = []
    for j in range(N_BLOCKS):
        kept = np.concatenate(
            [blocks[i] for i in range(N_BLOCKS) if i != j]
        )
        replicas.append(tau_int(kept))
    replicas = np.asarray(replicas)
    err = np.sqrt(
        (N_BLOCKS - 1) / N_BLOCKS * ((replicas - replicas.mean()) ** 2).sum()
    )
    return point, float(err)


def main() -> None:
    coarse = {L: load_series(ARTIFACTS / f"coarse-l{L}" / "series.jsonl") for L in SIZES}
    window = {L: load_series(ARTIFACTS / f"window-l{L}" / "series.jsonl") for L in SIZES}

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(12.0, 5.0), sharey=True
    )
    for L in SIZES:
        for series, ax, label, mk in (
            (coarse[L], ax_full, "coarse", "o"),
            (window[L], ax_zoom, "window", "s"),
        ):
            temps, values, errors = [], [], []
            for t in sorted(series):
                v, e = tau_with_error(np.abs(series[t]))
                temps.append(t)
                values.append(v)
                errors.append(e)
            ax.errorbar(
                temps, values, yerr=errors,
                marker=mk, ms=4, lw=1.2, capsize=2,
                color=COLORS[L], label=f"L = {L} ({label})",
            )

    for ax in (ax_full, ax_zoom):
        ax.axvline(TC, color="gray", ls="--", lw=1.0, alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("temperature  $T$")
        ax.grid(True, alpha=0.3, which="both")
    ax_full.set_ylabel(r"$\tau_{int}$ of $|M|$ (sweeps)")
    ax_full.set_title("full temperature range (coarse runs)", fontsize=10)
    ax_zoom.set_title("refined critical region (window runs)", fontsize=10)
    ax_full.legend(loc="upper right", fontsize=9)
    ax_zoom.legend(loc="upper left", fontsize=9)
    ax_full.text(TC, ax_full.get_ylim()[1], f"  $T_c$ = {TC:.3f}",
                 va="top", fontsize=8, color="gray")

    fig.suptitle(
        "Week 3 Part 3: integrated autocorrelation time of |M| "
        "(Week 3 rule, jackknife errors)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "tau.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
