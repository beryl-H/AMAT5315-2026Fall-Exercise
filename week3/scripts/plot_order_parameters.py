#!/usr/bin/env python3
"""Week 3 Part 2 order-parameter plots from the completed Metropolis runs.

Draws mean |M| per spin and the fluctuation susceptibility against
temperature for L = 32 and L = 64, using the completed runs under
week3/artifacts/:

- coarse runs  (t_grid 1.5..3.5 step 0.1,  5000 measured sweeps)
  -> the full temperature range
- window runs  (t_grid 2.0..2.6 step 0.05, 100000 measured sweeps)
  -> the refined critical region

Susceptibility uses exactly the Week 3 definition

    chi(T) = L^2 * (mean(M^2) - mean(|M|)^2) / T

where M is the per-spin magnetization stored in series.jsonl.

Uncertainties are jackknife estimates over contiguous blocks of the
measured time series (block length chosen so that blocks are much longer
than the sweep-level autocorrelation), for both mean |M| and chi.

Outputs:
- week3/evidence/magnetization.png
- week3/evidence/susceptibility.png

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/plot_order_parameters.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK3 = Path(__file__).resolve().parent.parent
ARTIFACTS = WEEK3 / "artifacts"
EVIDENCE = WEEK3 / "evidence"

RUNS = {
    "coarse": {32: ARTIFACTS / "coarse-l32" / "series.jsonl",
               64: ARTIFACTS / "coarse-l64" / "series.jsonl"},
    "window": {32: ARTIFACTS / "window-l32" / "series.jsonl",
               64: ARTIFACTS / "window-l64" / "series.jsonl"},
}
COLORS = {32: "tab:blue", 64: "tab:orange"}
# Onsager critical temperature of the 2D Ising model, for reference.
TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


def load_series(path: Path) -> dict[float, np.ndarray]:
    """Read series.jsonl -> {T: array of per-spin M values, ascending T}."""
    by_t: dict[float, list[float]] = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            by_t.setdefault(round(row["T"], 3), []).append(row["M"])
    return {t: np.asarray(ms) for t, ms in sorted(by_t.items())}


def jackknife_blocks(values: np.ndarray, n_blocks: int = 20) -> np.ndarray:
    """Split the series into contiguous blocks, return leave-one-out means."""
    blocks = np.array_split(values, n_blocks)
    totals = np.asarray([b.sum() for b in blocks])
    counts = np.asarray([len(b) for b in blocks])
    grand = totals.sum()
    return (grand - totals) / (counts.sum() - counts)


def block_length(n: int) -> int:
    """~20 contiguous jackknife blocks, at least 100 sweeps per block."""
    return max(100, -(-n // 20))


def jackknife_statistic(values: np.ndarray, kind: str, T: float, L: int) -> tuple[float, float]:
    """Jackknife mean and error for mean|M| ('m') or chi(T) ('chi')."""
    n = len(values)
    b = block_length(n)
    n_blocks = n // b
    used = values[: n_blocks * b]
    blocks = used.reshape(n_blocks, b)

    def statistic(samples: np.ndarray) -> float:
        abs_m = np.abs(samples)
        if kind == "m":
            return abs_m.mean()
        return L * L * ((samples**2).mean() - abs_m.mean() ** 2) / T

    full = statistic(used)
    leave_out = np.asarray([statistic(np.delete(blocks, j, axis=0).ravel()) for j in range(n_blocks)])
    mean_lo = leave_out.mean()
    # Bias-corrected point estimate (Efron): 2*full - mean of leave-one-out.
    estimate = 2.0 * full - mean_lo
    error = np.sqrt((n_blocks - 1) / n_blocks * ((leave_out - mean_lo) ** 2).sum())
    return estimate, error


def summarize(series: dict[float, np.ndarray], L: int, label: str) -> dict:
    """Compute mean|M|, chi and jackknife errors at every temperature."""
    rows = []
    for T, ms in series.items():
        m, m_err = jackknife_statistic(ms, "m", T, L)
        chi, chi_err = jackknife_statistic(ms, "chi", T, L)
        rows.append({"T": T, "m": m, "m_err": m_err, "chi": chi, "chi_err": chi_err})
        print(
            f"{label} L={L} T={T:5.2f}  |M|={m:.4f}+-{m_err:.4f}  "
            f"chi={chi:8.2f}+-{chi_err:6.2f}  ({len(ms)} sweeps, block {block_length(len(ms))})"
        )
    return {"L": L, "rows": rows}


def panel(ax, stats: list[dict], coarse: list[dict], key: str, err_key: str, ylabel: str, logy: bool = False):
    """Left panel: coarse full range; window points faded on top if present."""
    for s in stats:
        rows = s["rows"]
        T = np.asarray([r["T"] for r in rows])
        y = np.asarray([r[key] for r in rows])
        e = np.asarray([r[err_key] for r in rows])
        ax.errorbar(T, y, yerr=e, marker="o", ms=3.5, lw=1.2, capsize=2,
                    color=COLORS[s["L"]], label=f"L = {s['L']} (coarse)")
    if coarse:
        for s in coarse:
            rows = s["rows"]
            T = np.asarray([r["T"] for r in rows])
            y = np.asarray([r[key] for r in rows])
            ax.plot(T, y, "x", ms=4, mew=1.0, alpha=0.45, color=COLORS[s["L"]],
                    label=f"L = {s['L']} (coarse, faded)" if s is coarse[0] else None)
    ax.axvline(TC, color="gray", ls="--", lw=1.0, alpha=0.8)
    ax.text(TC, ax.get_ylim()[1], f"  $T_c$ = {TC:.3f}", va="top", fontsize=8, color="gray")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("temperature  $T$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def zoom_panel(ax, stats: list[dict], key: str, err_key: str, ylabel: str, logy: bool = False):
    """Right panel: window runs only (refined critical region)."""
    for s in stats:
        rows = s["rows"]
        T = np.asarray([r["T"] for r in rows])
        y = np.asarray([r[key] for r in rows])
        e = np.asarray([r[err_key] for r in rows])
        ax.errorbar(T, y, yerr=e, marker="o", ms=3.5, lw=1.2, capsize=2,
                    color=COLORS[s["L"]], label=f"L = {s['L']} (window)")
    ax.axvline(TC, color="gray", ls="--", lw=1.0, alpha=0.8)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("temperature  $T$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def make_figure(coarse: list[dict], window: list[dict], key: str, err_key: str,
                ylabel: str, suptitle: str, out_png: Path, logy: bool = False) -> None:
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    panel(ax_full, coarse, window, key, err_key, ylabel, logy)
    ax_full.set_title("full temperature range (coarse runs)", fontsize=10)
    zoom_panel(ax_zoom, window, key, err_key, ylabel, logy)
    ax_zoom.set_title("refined critical region (window runs)", fontsize=10)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> None:
    coarse = [summarize(load_series(RUNS["coarse"][L]), L, "coarse") for L in (32, 64)]
    window = [summarize(load_series(RUNS["window"][L]), L, "window") for L in (32, 64)]

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    make_figure(
        coarse, window, "m", "m_err",
        r"mean $|M|$ per spin",
        f"Metropolis Ising: mean |M| vs temperature (jackknife errors, $T_c$ = {TC:.3f})",
        EVIDENCE / "magnetization.png",
    )
    make_figure(
        coarse, window, "chi", "chi_err",
        r"$\chi(T) = L^2\,(\langle M^2\rangle - \langle|M|\rangle^2)/T$",
        "Metropolis Ising: susceptibility vs temperature (jackknife errors)",
        EVIDENCE / "susceptibility.png",
        logy=True,
    )


if __name__ == "__main__":
    main()
