#!/usr/bin/env python3
"""Week 3 Part 3: block-bootstrap sampling error of the finite-size T_c.

Uses the existing critical-window Metropolis runs (week3/artifacts/
window-l32, window-l64; 13 temperatures, 100000 measured sweeps each).

For each block length BL in {2000, 4000, 8000} sweeps the per-temperature
|M| series is cut into consecutive blocks of BL sweeps (complete blocks
only; the remainder is dropped).  500 bootstrap replicates are drawn by
resampling the blocks with replacement, independently for each lattice
size and temperature.  For every replicate:

1. the susceptibility chi(T) = L^2*(mean(M^2) - mean(|M|)^2)/T is
   recomputed from the resampled series at every temperature;
2. the prescribed five-point quadratic peak fit is applied for L = 32 and
   L = 64 (five temperatures centered on the replicate's chi maximum);
3. Tc = 2*T_peak(64) - T_peak(32).

A replicate fit fails when the fitted parabola does not bend downward
(a >= 0) or its vertex falls outside the five fitted temperatures.

The bootstrap standard error of Tc is reported separately per block
length.  The Week 3 stability rule is applied: the sampling error is
stable only if the three block-length error estimates agree within one
tenth of their mean; otherwise the sampling error is reported as
unresolved.  The reported sampling error contains statistical noise only;
finite-size bias and five-point-fit bias are deliberately excluded.

Outputs:
- prints the report,
- saves week3/evidence/chi-bootstrap.png.

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/chi_bootstrap.py
"""

from __future__ import annotations

import numpy as np

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
BLOCK_LENGTHS = (2000, 4000, 8000)
N_BOOT = 500
N_FIT = 5
SEED = 20260303  # fixed seed so the bootstrap is exactly reproducible
STABILITY_TOL = 0.1  # Week 3 rule: spread <= 0.1 * mean of the three SEs


def block_sums(abs_m: np.ndarray, bl: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-block mean(M^2) and mean(|M|) for consecutive complete blocks."""
    n = (len(abs_m) // bl) * bl
    blocks = abs_m[:n].reshape(-1, bl)
    return (blocks**2).mean(axis=1), blocks.mean(axis=1)


def chi_from_blocks(L: int, T: float, m2: np.ndarray, mabs: np.ndarray) -> float:
    """chi of a resample: equal block sizes -> mean of block means."""
    return L * L * (m2.mean() - mabs.mean() ** 2) / T


def fit_peak(temps: np.ndarray, chis: np.ndarray) -> tuple[float, bool]:
    """Prescribed five-point quadratic peak; returns (T_peak, ok)."""
    i_max = int(np.argmax(chis))
    half = N_FIT // 2
    lo = min(max(i_max - half, 0), len(temps) - N_FIT)
    fit_T = temps[lo : lo + N_FIT]
    fit_chi = chis[lo : lo + N_FIT]
    a, b = np.polyfit(fit_T, fit_chi, 2)[:2]
    vertex = -b / (2 * a)
    ok = a < 0 and fit_T[0] <= vertex <= fit_T[-1]
    return vertex, ok


def bootstrap_tc(
    series: dict[int, dict[float, np.ndarray]], bl: int, rng: np.random.Generator
) -> dict:
    """Run N_BOOT replicates for one block length; return peak/Tc samples."""
    temps = np.asarray(sorted(series[32]))
    # Precomputed block statistics: stats[L][k] = (mean(M^2) blocks, mean(|M|) blocks).
    stats: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for L in SIZES:
        stats[L] = [block_sums(np.abs(series[L][t]), bl) for t in temps]
    n_blocks = stats[32][0][0].size

    peaks: dict = {L: np.full(N_BOOT, np.nan) for L in SIZES}
    peaks["Tc"] = np.full(N_BOOT, np.nan)
    ok = {L: np.zeros(N_BOOT, dtype=bool) for L in SIZES}
    curves = {L: np.empty((N_BOOT, len(temps))) for L in SIZES}
    for r in range(N_BOOT):
        chi_reps: dict[int, np.ndarray] = {}
        for L in SIZES:
            c = np.empty(len(temps))
            for k in range(len(temps)):
                m2_b, mabs_b = stats[L][k]
                idx = rng.integers(0, n_blocks, n_blocks)
                c[k] = chi_from_blocks(L, temps[k], m2_b[idx], mabs_b[idx])
            chi_reps[L] = c
            curves[L][r] = c
        all_ok = True
        for L in SIZES:
            vertex, good = fit_peak(temps, chi_reps[L])
            peaks[L][r] = vertex
            ok[L][r] = good
            all_ok &= good
        if all_ok:
            peaks["Tc"][r] = 2 * peaks[64][r] - peaks[32][r]
        else:
            peaks["Tc"][r] = np.nan
    return {"temps": temps, "peaks": peaks, "ok": ok, "curves": curves, "n_blocks": n_blocks}


def main() -> None:
    series = {L: load_series(ARTIFACTS / f"window-l{L}" / "series.jsonl") for L in SIZES}
    temps = np.asarray(sorted(series[32]))

    # Point-estimate chi curves (full series, Week 3 formula).
    point_chi = {}
    for L in SIZES:
        c = []
        for t in temps:
            ms = series[L][t]
            abs_m = np.abs(ms)
            c.append(L * L * ((ms**2).mean() - abs_m.mean() ** 2) / t)
        point_chi[L] = np.asarray(c)

    rng = np.random.default_rng(SEED)
    results = {bl: bootstrap_tc(series, bl, rng) for bl in BLOCK_LENGTHS}

    lines: list[str] = []
    lines.append("Week 3 Part 3: block-bootstrap sampling error of Tc = 2*T_peak(64) - T_peak(32)")
    lines.append(f"data: window-l32/window-l64 (13 temperatures, 100000 sweeps each); seed {SEED}")
    lines.append(f"{N_BOOT} replicates per block length; complete consecutive blocks only")
    lines.append("sampling error contains bootstrap noise only (no finite-size or fit bias)")
    lines.append("")

    tc_se = {}
    for bl in BLOCK_LENGTHS:
        res = results[bl]
        tc = res["peaks"]["Tc"]
        valid = tc[~np.isnan(tc)]
        n_valid = valid.size
        n_fail32 = int((~res["ok"][32]).sum())
        n_fail64 = int((~res["ok"][64]).sum())
        se = float(valid.std(ddof=1))
        tc_se[bl] = se
        lines.append(f"block length {bl:5d} sweeps ({res['n_blocks']} blocks per series):")
        lines.append(
            f"  failed fits: L=32 {n_fail32}/{N_BOOT}, L=64 {n_fail64}/{N_BOOT}"
            f"  ->  {n_valid}/{N_BOOT} valid Tc replicates"
        )
        lines.append(f"  Tc bootstrap SE = {se:.4f}")
        lines.append("")

    mean_se = float(np.mean(list(tc_se.values())))
    spread = max(tc_se.values()) - min(tc_se.values())
    lines.append(
        f"stability rule (spread <= {STABILITY_TOL:.0%} of mean): "
        f"spread = {spread:.4f}, mean = {mean_se:.4f}"
    )
    if spread <= STABILITY_TOL * mean_se:
        lines.append(f"sampling error STABLE: Tc SE = {mean_se:.4f}")
    else:
        lines.append("sampling error UNRESOLVED: block-length error estimates do not agree")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "chi-bootstrap.txt"
    out.write_text(report, encoding="utf-8")

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax32, ax64 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_tc = fig.add_subplot(gs[1, :])
    fig.suptitle(
        f"Week 3 Part 3: block-bootstrap of the susceptibility peaks "
        f"({N_BOOT} replicates per block length, seed {SEED})",
        fontsize=11,
    )

    BL_SHADES = {2000: "tab:green", 4000: "tab:purple", 8000: "tab:brown"}

    for ax, L in ((ax32, 32), (ax64, 64)):
        # Envelope: pooled bootstrap susceptibility curves, faint, plus the
        # pooled 16-84 percentile band and the point-estimate curve.
        pooled = np.vstack([results[bl]["curves"][L] for bl in BLOCK_LENGTHS])
        for r in range(pooled.shape[0]):
            ax.plot(temps, pooled[r], color="gray", alpha=0.02, lw=0.5)
        lo, hi = np.percentile(pooled, [16, 84], axis=0)
        ax.fill_between(temps, lo, hi, color="tab:blue", alpha=0.15,
                        label="bootstrap 16-84% band (pooled)")
        ax.plot(temps, point_chi[L], "o-", color=COLORS[L], lw=1.6, ms=4,
                label=f"point estimate, L = {L}")
        ax.set_yscale("log")
        # Fitted-peak envelopes: 16-84% band of T_peak per block length,
        # stacked just above the x-axis where the panel is empty.
        ymin, ymax = ax.get_ylim()
        for i, bl in enumerate(BLOCK_LENGTHS):
            p = results[bl]["peaks"][L][results[bl]["ok"][L]]
            q16, q50, q84 = np.percentile(p, [16, 50, 84])
            y = ymin * (ymax / ymin) ** (0.04 + 0.05 * i)
            ax.errorbar(q50, y, xerr=[[q50 - q16], [q84 - q50]], fmt="|",
                        ms=10, color=BL_SHADES[bl], ecolor=BL_SHADES[bl],
                        elinewidth=2.5, capsize=4, capthick=1.5,
                        label=f"T_peak 16-84%, BL = {bl}")
        ax.set_xlabel("temperature  $T$")
        ax.set_ylabel(r"$\chi(T)$")
        ax.set_title(f"L = {L}: susceptibility and fitted-peak envelopes", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)

    ax_tc.axhline(TC, color="gray", ls="--", lw=1.0)
    ax_tc.text(2.62, TC, f" Onsager $T_c$ = {TC:.4f}", va="bottom", ha="right",
               fontsize=8, color="gray")
    for i, bl in enumerate(BLOCK_LENGTHS):
        tc = results[bl]["peaks"]["Tc"]
        valid = tc[~np.isnan(tc)]
        x = i + np.random.default_rng(SEED + bl).uniform(-0.18, 0.18, valid.size)
        ax_tc.scatter(x, valid, s=4, alpha=0.25, color=BL_SHADES[bl])
        m, se = valid.mean(), valid.std(ddof=1)
        ax_tc.errorbar(i, m, yerr=se, fmt="_", ms=18, mec="black", ecolor="black",
                       capsize=4, lw=1.2)
        ax_tc.annotate(f"SE = {se:.4f}", (i, m), xytext=(8, -2),
                       textcoords="offset points", fontsize=8)
    ax_tc.set_xticks(range(len(BLOCK_LENGTHS)), [str(b) for b in BLOCK_LENGTHS])
    ax_tc.set_xlabel("block length (sweeps)")
    ax_tc.set_ylabel(r"$T_c$ replicate")
    ax_tc.set_title("bootstrap Tc replicates per block length (bar: mean ± SE)", fontsize=10)
    ax_tc.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = EVIDENCE / "chi-bootstrap.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
