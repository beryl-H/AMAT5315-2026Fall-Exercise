#!/usr/bin/env python3
"""Week 3 Part 3: autocorrelation and binning analysis, L = 64, T = 2.3.

Data source: week3/artifacts/window-l64/series.jsonl (critical-window run,
100000 measured sweeps).  T = 2.3 lies inside the critical window.

Panel 1: the normalized autocorrelation function rho(t) of the per-spin
|M| series against lag t.  The Week 3 rule
    tau_int = 1/2 + sum rho(t),
with the sum stopped once the lag exceeds six times the running tau_int,
is applied exactly as in scripts/errors.py, so tau_int is consistent with
the L = 64, T = 2.3 window entry in week3/evidence/errors.txt.

Panel 2: the standard error of mean |M| against block length (log x),
using consecutive non-overlapping blocks.  At block length 1 the estimate
reduces exactly to the naive standard error in
week3/evidence/errors.txt (std(|M|, ddof=1)/sqrt(n)).  Block lengths are
divisors of 100000, so every point uses all sweeps; the number of blocks
remaining at each block length is printed.  No plateau is claimed unless
the curve actually stabilizes.

Output: week3/evidence/acf-binning.png

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/plot_acf_binning.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK3 = Path(__file__).resolve().parent.parent
SERIES = WEEK3 / "artifacts" / "window-l64" / "series.jsonl"
EVIDENCE = WEEK3 / "evidence"

L = 64
T = 2.3
TAU_WINDOW_FACTOR = 6.0  # Week 3 rule: stop summing rho once lag > 6 * tau_int
RHO_MAX_LAG = 7000  # lags shown in panel 1


def load_abs_m(path: Path, T: float) -> np.ndarray:
    """|M| series at temperature T."""
    ms = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["T"] == T:
                ms.append(abs(row["M"]))
    return np.asarray(ms)


def rho_function(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation rho(0..max_lag) via FFT."""
    y = x - x.mean()
    n = len(y)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1].real / (
        n - np.arange(max_lag + 1)
    )
    return acov / acov[0]


def tau_int_week3(rho: np.ndarray) -> tuple[float, int]:
    """Week 3 rule on a precomputed rho; returns (tau_int, stopping lag).

    Same accumulation order as scripts/errors.py.
    """
    total = 0.0
    t = 1
    while t < len(rho):
        total += rho[t]
        tau = 0.5 + total
        if t > TAU_WINDOW_FACTOR * tau:
            break
        t += 1
    return 0.5 + total, t


def blocked_se(x: np.ndarray, bl: int) -> tuple[float, int]:
    """SE of the mean from consecutive blocks of length bl (equal blocks)."""
    n_blocks = len(x) // bl
    means = x[: n_blocks * bl].reshape(n_blocks, bl).mean(axis=1)
    return means.std(ddof=1) / np.sqrt(n_blocks), n_blocks


def main() -> None:
    abs_m = load_abs_m(SERIES, T)
    n = len(abs_m)

    rho = rho_function(abs_m, RHO_MAX_LAG)
    tau, stop_lag = tau_int_week3(rho)
    naive_se, _ = blocked_se(abs_m, 1)

    # Divisors of 100000 up to 5000 sweeps: every point uses all sweeps.
    block_lengths = [
        d for d in (1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 80, 100,
                    125, 160, 200, 250, 400, 500, 625, 800, 1000, 1250,
                    2000, 2500, 4000, 5000)
        if n % d == 0
    ]
    ses = []
    n_blocks_list = []
    for bl in block_lengths:
        se, nb = blocked_se(abs_m, bl)
        ses.append(se)
        n_blocks_list.append(nb)

    print(f"data: {SERIES.relative_to(WEEK3)} (L = {L}, T = {T:.1f}, {n} sweeps)")
    print(f"tau_int = {tau:.2f} sweeps, sum stopped at lag {stop_lag}")
    print(f"naive SE (block length 1) = {naive_se:.6f}")
    print("")
    print("binning table:")
    print(f"{'block_len':>9} {'n_blocks':>9} {'SE(mean|M|)':>12}")
    for bl, nb, se in zip(block_lengths, n_blocks_list, ses):
        print(f"{bl:9d} {nb:9d} {se:12.6f}")

    fig, (ax_rho, ax_bin) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    ax_rho.plot(range(len(rho)), rho, "-", lw=0.8, color="tab:blue")
    ax_rho.axhline(0.0, color="gray", lw=0.8, alpha=0.7)
    ax_rho.axvline(tau, color="tab:red", ls="--", lw=1.2,
                   label=f"$\\tau_{{int}}$ = {tau:.1f} (Week 3 rule)")
    ax_rho.axvline(stop_lag, color="tab:green", ls=":", lw=1.2,
                   label=f"sum stopped at lag {stop_lag} (= 6$\\tau_{{int}}$)")
    ax_rho.set_xlabel(r"lag $t$ (sweeps)")
    ax_rho.set_ylabel(r"$\rho(t)$ of $|M|$")
    ax_rho.set_title(
        f"autocorrelation, L = {L}, T = {T:.1f} (window run, {n} sweeps)",
        fontsize=10,
    )
    ax_rho.legend(loc="upper right", fontsize=8)
    ax_rho.grid(True, alpha=0.3)

    ax_bin.plot(block_lengths, ses, "o-", lw=1.2, ms=4, color="tab:blue")
    ax_bin.set_xscale("log")
    # Reference: the naive SE at block length 1 (equals errors.txt value).
    ax_bin.axhline(naive_se, color="tab:orange", ls="--", lw=1.2,
                   label=f"naive SE at block length 1 = {naive_se:.6f}")
    # Independent-sweep expectation for fully decorrelated blocks.
    ax_bin.axhline(naive_se * np.sqrt(2.0 * tau), color="tab:red", ls=":",
                   lw=1.2,
                   label=f"naive $\\cdot\\,\\sqrt{{2\\tau_{{int}}}}$ = "
                         f"{naive_se * np.sqrt(2.0 * tau):.6f}")
    # Mark the 50-block point (block length 2000) reported in errors.txt.
    i_50 = block_lengths.index(2000)
    se_50 = ses[i_50]
    ax_bin.plot(2000, se_50, "s", ms=9, mfc="none", mec="tab:purple", mew=1.8)
    ax_bin.annotate(
        f"50 blocks\nSE = {se_50:.6f}", (2000, se_50),
        xytext=(-10, -30), textcoords="offset points",
        fontsize=8, color="tab:purple", ha="right",
    )
    # Annotate the block count at the largest block length.
    ax_bin.annotate(
        f"{n_blocks_list[-1]} blocks", (block_lengths[-1], ses[-1]),
        xytext=(0, 8), textcoords="offset points", fontsize=8,
        ha="center",
    )
    ax_bin.set_xlabel("block length (sweeps, log scale)")
    ax_bin.set_ylabel("standard error of mean $|M|$")
    ax_bin.set_title(
        f"binning analysis, L = {L}, T = {T:.1f} (window run, {n} sweeps)",
        fontsize=10,
    )
    ax_bin.legend(loc="upper left", fontsize=8)
    ax_bin.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"Week 3 Part 3: why the naive error fails - L = {L} Metropolis at T = {T:.1f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "acf-binning.png"
    fig.savefig(out, dpi=150)
    print("")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
