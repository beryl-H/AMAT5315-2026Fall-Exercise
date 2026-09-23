#!/usr/bin/env python3
"""Week 3 Part 3: metropolis vs wolff comparison for L = 64, plus peaks.

Panel A: mean |M| against temperature for L = 64 from the completed
critical-window runs in week3/artifacts/ - window-l64 (metropolis sweeps)
and wolff-l64 (wolff cluster flips, 100000 measured steps each) - with
block-bootstrap error bars (blocks of 2000 measured steps, 500 resamples;
the same bootstrap scheme as scripts/chi_bootstrap.py).

Panel B: the susceptibility chi(T) = L^2*(mean(M^2) - mean(|M|)^2)/T
computed from the wolff runs for L = 32 and L = 64, with the prescribed
five-point quadratic peak fit for each size, the vertex locations
T_peak(32) and T_peak(64), and the extrapolated critical temperature
Tc = 2*T_peak(64) - T_peak(32).  The Onsager value 2.26919 is marked.

Output: week3/evidence/magnetization-compare.png

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/plot_magnetization_compare.py
"""

from __future__ import annotations

import numpy as np

from plot_order_parameters import (  # noqa: E402  (same scripts/ directory)
    ARTIFACTS,
    EVIDENCE,
    TC,
    load_series,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = (32, 64)
L_PANEL = 64
BLOCK_LEN = 2000
N_BOOT = 500
N_FIT = 5
SEED = 20260303  # fixed seed so the bootstrap is exactly reproducible

UPDATES = {
    "metropolis": "window-l64",
    "wolff": "wolff-l64",
}


def block_bootstrap_se(abs_m: np.ndarray, bl: int, n_boot: int, rng: np.random.Generator) -> tuple[float, int]:
    """SE of the mean |M| from block-bootstrap (blocks of `bl` sweeps)."""
    n_blocks = len(abs_m) // bl
    means = abs_m[: n_blocks * bl].reshape(n_blocks, bl).mean(axis=1)
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    replicate_means = means[idx].mean(axis=1)
    return float(replicate_means.std(ddof=1)), n_blocks


def chi(L: int, t: float, ms: np.ndarray) -> float:
    """Week 3 susceptibility from a measured series."""
    abs_m = np.abs(ms)
    return L * L * ((ms**2).mean() - abs_m.mean() ** 2) / t


def fit_peak(temps: np.ndarray, chis: np.ndarray) -> dict:
    """Prescribed five-point quadratic peak; returns vertex and fit points."""
    i_max = int(np.argmax(chis))
    half = N_FIT // 2
    lo = min(max(i_max - half, 0), len(temps) - N_FIT)
    fit_T = temps[lo : lo + N_FIT]
    fit_chi = chis[lo : lo + N_FIT]
    a, b = np.polyfit(fit_T, fit_chi, 2)[:2]
    vertex = -b / (2 * a)
    assert a < 0 and fit_T[0] <= vertex <= fit_T[-1], "peak fit failed"
    return {"T_peak": vertex, "fit_T": fit_T, "fit_chi": fit_chi}


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ---- Panel A: mean |M| for L = 64, metropolis vs wolff ---------------
    panel_a: dict[str, dict] = {}
    for update, folder in UPDATES.items():
        series = load_series(ARTIFACTS / folder / "series.jsonl")
        temps = sorted(series)
        means, ses, n_blocks = [], [], None
        for t in temps:
            abs_m = np.abs(series[t])
            means.append(abs_m.mean())
            se, n_blocks = block_bootstrap_se(abs_m, BLOCK_LEN, N_BOOT, rng)
            ses.append(se)
        panel_a[update] = {
            "temps": np.asarray(temps),
            "means": np.asarray(means),
            "ses": np.asarray(ses),
            "n_blocks": n_blocks,
        }
        print(f"panel A: {update} ({folder}), blocks of {BLOCK_LEN}, {n_blocks} blocks, {N_BOOT} resamples")
        for t, m, se in zip(temps, means, ses):
            print(f"  T={t:.2f}  mean|M| = {m:.4f} +- {se:.4f}")

    # ---- Panel B: wolff susceptibility, peaks, extrapolated Tc -----------
    peaks: dict[int, dict] = {}
    wolff_chi: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L in SIZES:
        series = load_series(ARTIFACTS / f"wolff-l{L}" / "series.jsonl")
        temps = np.asarray(sorted(series))
        chis = np.asarray([chi(L, t, series[t]) for t in temps])
        wolff_chi[L] = (temps, chis)
        peaks[L] = fit_peak(temps, chis)
        fit_list = ", ".join(f"({t:.2f}, {c:.2f})" for t, c in zip(peaks[L]["fit_T"], peaks[L]["fit_chi"]))
        print(f"panel B: wolff L = {L}: 5-point fit {fit_list}")
        print(f"  T_peak({L}) = {peaks[L]['T_peak']:.4f}")
    tc_wolff = 2 * peaks[64]["T_peak"] - peaks[32]["T_peak"]
    print(f"panel B: Tc = 2*T_peak(64) - T_peak(32) = {tc_wolff:.4f} (wolff)")
    print(f"Onsager Tc = {TC:.5f}")

    # ---- figure -----------------------------------------------------------
    fig, (ax_m, ax_chi) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.suptitle(
        f"Week 3 Part 3: metropolis vs wolff, L = {L_PANEL} "
        f"(block-bootstrap errors, blocks of {BLOCK_LEN} steps, {N_BOOT} resamples)",
        fontsize=11,
    )

    colors = {"metropolis": "tab:blue", "wolff": "tab:orange"}
    labels = {"metropolis": "metropolis (window-l64)", "wolff": "wolff (wolff-l64)"}
    for update, d in panel_a.items():
        ax_m.errorbar(
            d["temps"], d["means"], yerr=d["ses"],
            marker="o", ms=3.5, lw=1.3, capsize=2,
            color=colors[update], label=labels[update],
        )
    ax_m.axvline(TC, color="gray", ls="--", lw=1.0, alpha=0.8)
    ax_m.set_xlabel("temperature  $T$")
    ax_m.set_ylabel(r"mean $|M|$ per spin")
    ax_m.set_title(f"mean |M|, L = {L_PANEL}", fontsize=10)
    ax_m.legend(loc="upper right", fontsize=8)
    ax_m.grid(True, alpha=0.3)

    chi_colors = {32: "tab:blue", 64: "tab:orange"}
    for L in SIZES:
        temps, chis = wolff_chi[L]
        ax_chi.plot(temps, chis, "o-", lw=1.3, ms=4, color=chi_colors[L],
                    label=f"wolff, L = {L}")
        p = peaks[L]
        ax_chi.plot(p["fit_T"], p["fit_chi"], "o", mfc="none", ms=9,
                    mew=1.6, color=chi_colors[L])
        ax_chi.axvline(p["T_peak"], color=chi_colors[L], ls=":", lw=1.0, alpha=0.7)
        ax_chi.annotate(
            f"$T_{{peak}}$({L}) = {p['T_peak']:.4f}",
            (p["T_peak"], ax_chi.get_ylim()[1]),
            xytext=(5 if L == 64 else -5, -14),
            textcoords="offset points", fontsize=8, color=chi_colors[L],
            ha="left" if L == 64 else "right", va="top",
        )
    ax_chi.axvline(TC, color="gray", ls="--", lw=1.2)
    ax_chi.text(TC, 0.5, f" Onsager $T_c$ = 2.26919", rotation=90,
                va="bottom", ha="right", fontsize=8, color="gray",
                transform=ax_chi.get_xaxis_transform())
    ax_chi.annotate(
        f"$T_c$ = 2$T_{{peak}}$(64)$-T_{{peak}}$(32) = {tc_wolff:.4f}",
        (tc_wolff, 1.0), xytext=(8, -24), textcoords="offset points",
        fontsize=8, ha="left", va="top",
    )
    ax_chi.set_yscale("log")
    ax_chi.set_xlabel("temperature  $T$")
    ax_chi.set_ylabel(r"$\chi(T) = L^2(\langle M^2\rangle-\langle|M|\rangle^2)/T$")
    ax_chi.set_title("wolff cluster susceptibility and peaks", fontsize=10)
    ax_chi.legend(loc="lower left", fontsize=8)
    ax_chi.grid(True, alpha=0.3, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "magnetization-compare.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
