#!/usr/bin/env python3
"""Week 3 finite-size peak analysis from the completed Metropolis runs.

For each lattice size L in {32, 64} this script reads the critical-window
Metropolis runs in week3/artifacts/, computes the Week 3 susceptibility

    chi(T) = L^2 * (mean(M^2) - mean(|M|)^2) / T

at every temperature, then:

1. locates the largest chi in the critical-window data;
2. takes the five temperature points centered on that maximum;
3. fits a quadratic chi(T) = a*T^2 + b*T + c to those five points;
4. takes the quadratic vertex T_peak = -b / (2a) as the peak location.

The infinite-volume critical temperature is then estimated by the
first-order (in 1/L) finite-size extrapolation

    T_c = 2 * T_peak(64) - T_peak(32)

As a low-temperature consistency check the script also reports the mean
per-spin |M| at the lowest available temperature (T = 1.5, coarse runs).

Uncertainties are jackknife estimates over 20 contiguous blocks of the
measured time series: for each leave-one-block-out replica the chi values
at the five fit temperatures are recomputed and the quadratic refitted,
giving an error on T_peak that includes both the statistical noise of chi
and the critical slowing down near the peak.

Outputs: prints the values and writes week3/evidence/peaks.txt.

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/peaks.py
"""

from __future__ import annotations

import numpy as np

from plot_order_parameters import (  # noqa: E402  (same scripts/ directory)
    ARTIFACTS,
    WEEK3,
    load_series,
)

EVIDENCE = WEEK3 / "evidence"
SIZES = (32, 64)
N_BLOCKS = 20
N_FIT = 5  # temperature points used in the quadratic fit


def chis_by_temperature(series: dict[float, np.ndarray], L: int) -> dict[float, float]:
    """Week 3 susceptibility at every temperature of a completed run."""
    out = {}
    for T, ms in series.items():
        abs_m = np.abs(ms)
        out[T] = L * L * ((ms**2).mean() - abs_m.mean() ** 2) / T
    return out


def blocks_of(values: np.ndarray) -> np.ndarray:
    """Split into N_BLOCKS contiguous blocks (values length must divide)."""
    return np.array_split(values, N_BLOCKS)


def peak_and_error(series: dict[float, np.ndarray], L: int) -> dict:
    """Point estimate and jackknife error for the quadratic-vertex peak."""
    chis = chis_by_temperature(series, L)
    temps = np.asarray(sorted(chis))
    chi_vals = np.asarray([chis[t] for t in temps])

    # 1. largest chi in the critical-window data.
    i_max = int(np.argmax(chi_vals))
    half = N_FIT // 2
    lo, hi = i_max - half, i_max + half + 1
    assert lo >= 0 and hi <= len(temps), "peak too close to the window edge for 5 points"
    fit_T = temps[lo:hi]
    fit_chi = chi_vals[lo:hi]

    def vertex(t: np.ndarray, c: np.ndarray) -> float:
        a, b = np.polyfit(t, c, 2)[:2]
        return -b / (2 * a)

    t_peak = vertex(fit_T, fit_chi)

    # Jackknife: drop block j from every fit temperature, refit, repeat.
    fit_series = {t: blocks_of(series[t]) for t in fit_T}
    replicas = []
    for j in range(N_BLOCKS):
        chi_j = {}
        for t, blocks in fit_series.items():
            kept = np.delete(blocks, j, axis=0).ravel()
            abs_m = np.abs(kept)
            chi_j[t] = L * L * ((kept**2).mean() - abs_m.mean() ** 2) / t
        t_j = fit_T
        c_j = np.asarray([chi_j[t] for t in t_j])
        replicas.append(vertex(t_j, c_j))
    replicas = np.asarray(replicas)
    t_peak_err = np.sqrt(
        (N_BLOCKS - 1) / N_BLOCKS * ((replicas - replicas.mean()) ** 2).sum()
    )
    chi_peak = float(fit_chi.max())
    return {
        "T_peak": t_peak,
        "T_peak_err": t_peak_err,
        "chi_peak": chi_peak,
        "chi_peak_T": float(fit_T[int(np.argmax(fit_chi))]),
        "fit_T": fit_T,
        "fit_chi": fit_chi,
        "replicas": replicas,
    }


def mean_abs_m_at_lowest_t(series: dict[float, np.ndarray]) -> tuple[float, float, float]:
    """Mean |M| and jackknife error at the lowest temperature of a run."""
    T_low = min(series)
    ms = series[T_low]
    blocks = blocks_of(ms)
    full = np.abs(ms).mean()
    leave_out = np.asarray([np.delete(blocks, j, axis=0).ravel() for j in range(N_BLOCKS)])
    means = np.abs(leave_out).mean(axis=1)
    err = np.sqrt(
        (N_BLOCKS - 1) / N_BLOCKS * ((means - means.mean()) ** 2).sum()
    )
    return T_low, float(full), float(err)


def main() -> None:
    window = {L: load_series(ARTIFACTS / f"window-l{L}" / "series.jsonl") for L in SIZES}
    coarse = {L: load_series(ARTIFACTS / f"coarse-l{L}" / "series.jsonl") for L in SIZES}

    lines: list[str] = []
    say = lines.append

    say("Week 3 Ising Metropolis: susceptibility peaks and finite-size T_c")
    say(f"chi(T) = L^2 * (mean(M^2) - mean(|M|)^2) / T; quadratic vertex of {N_FIT} centered points")
    say(f"critical-window runs: {', '.join(f'window-l{L}' for L in SIZES)}")
    say("")

    peaks: dict[int, dict] = {}
    for L in SIZES:
        p = peak_and_error(window[L], L)
        peaks[L] = p
        fit_list = ", ".join(f"({t:.2f}, {c:.2f})" for t, c in zip(p["fit_T"], p["fit_chi"]))
        say(f"L = {L}:")
        say(f"  largest window chi: {p['chi_peak']:.2f} at T = {p['chi_peak_T']:.2f}")
        say(f"  5-point quadratic fit (T, chi): {fit_list}")
        say(f"  T_peak = {p['T_peak']:.4f} +- {p['T_peak_err']:.4f}  (jackknife, {N_BLOCKS} blocks)")
        say("")

    t_c = 2 * peaks[64]["T_peak"] - peaks[32]["T_peak"]
    t_c_err = float(
        np.sqrt(
            (2 * peaks[64]["T_peak_err"]) ** 2 + peaks[32]["T_peak_err"] ** 2
        )
    )
    say(f"T_c = 2 * T_peak(64) - T_peak(32) = {t_c:.4f} +- {t_c_err:.4f}")
    say(f"Onsager exact T_c = {2.0 / np.log(1.0 + np.sqrt(2.0)):.4f}")
    say("")

    say("mean |M| at the lowest temperature (coarse runs, T = 1.5):")
    for L in SIZES:
        t_low, m, m_err = mean_abs_m_at_lowest_t(coarse[L])
        say(f"  L = {L}: |M| = {m:.4f} +- {m_err:.4f} at T = {t_low:.2f}")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "peaks.txt"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
