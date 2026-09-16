#!/usr/bin/env python3
"""Week 3 Part 3: "How much to trust Part 2" - error analysis.

For every lattice size and temperature of the completed Metropolis runs in
week3/artifacts/ this script computes and prints, for the per-spin
magnetization time series M (using |M| for the correlated quantities):

- L, T
- mean |M|
- naive standard error over the measured sweeps,
      naive = std(|M|) / sqrt(n)          (assumes independent sweeps)
- standard error from 50 block averages,
      blocked = std(block means, ddof=1) / sqrt(50)
- ratio = blocked / naive
- integrated autocorrelation time with the Week 3 rule,
      tau_int = 1/2 + sum_{t>=1} rho(t),
  where the sum is stopped once the lag exceeds six times the running
  integrated autocorrelation time (rho = normalized autocorrelation of |M|).

IMPORTANT (Part 3): the 50-block estimate is only trustworthy when the
block length (n/50 sweeps) is comfortably longer than tau_int. Near the
critical temperature tau_int grows large and this can fail; the table
prints block_length/tau_int so the failure is visible, and the ratio is
NOT to be read as automatically final or honest.

The complete printed output is saved to week3/evidence/errors.txt.

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/errors.py
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
N_BLOCKS = 50
TAU_WINDOW_FACTOR = 6.0  # stop summing rho once lag > 6 * running tau_int


def naive_error(abs_m: np.ndarray) -> float:
    """Standard error of the mean under the independence assumption."""
    return abs_m.std(ddof=1) / np.sqrt(len(abs_m))


def blocked_error(abs_m: np.ndarray) -> tuple[float, float]:
    """Standard error from N_BLOCKS contiguous block averages.

    Returns (blocked_error, block_length). Requires n divisible by the
    block count for equal blocks; both run lengths here satisfy that.
    """
    n = len(abs_m)
    if n % N_BLOCKS != 0:
        raise ValueError(f"n = {n} not divisible by {N_BLOCKS} blocks")
    block_len = n // N_BLOCKS
    means = abs_m.reshape(N_BLOCKS, block_len).mean(axis=1)
    return means.std(ddof=1) / np.sqrt(N_BLOCKS), float(block_len)


def tau_int(abs_m: np.ndarray) -> float:
    """Integrated autocorrelation time of |M| with the Week 3 rule.

    tau_int = 1/2 + sum_{t>=1} rho(t); the sum stops once the lag exceeds
    six times the running integrated autocorrelation time.
    """
    x = abs_m - abs_m.mean()
    n = len(x)
    # Normalized autocorrelation via FFT (unbiased estimator).
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[:n].real / (n - np.arange(n))
    if acov[0] <= 0:
        return 0.5
    rho = acov / acov[0]

    total = 0.0  # sum of rho(1..t)
    t = 1
    while t < n:
        total += rho[t]
        tau = 0.5 + total  # running integrated autocorrelation time
        if t > TAU_WINDOW_FACTOR * tau:
            break
        t += 1
    return 0.5 + total


def analyze(label: str, series: dict[float, np.ndarray], L: int, lines: list[str]) -> None:
    lines.append(f"=== {label} (L = {L}) ===")
    header = (
        f"{'T':>5} {'n_sweeps':>9} {'mean|M|':>9} {'naive_err':>10} "
        f"{'blocked_err':>11} {'ratio':>6} {'tau_int':>8} {'block_len/tau':>13}"
    )
    lines.append(header)
    for T, ms in series.items():
        abs_m = np.abs(ms)
        n = len(abs_m)
        naive = naive_error(abs_m)
        blocked, block_len = blocked_error(abs_m)
        tau = tau_int(abs_m)
        ratio = blocked / naive if naive > 0 else float("nan")
        safety = block_len / tau if tau > 0 else float("inf")
        lines.append(
            f"{T:5.2f} {n:9d} {abs_m.mean():9.4f} {naive:10.6f} "
            f"{blocked:11.6f} {ratio:6.2f} {tau:8.2f} {safety:13.1f}"
        )
    lines.append("")


def main() -> None:
    lines: list[str] = []
    lines.append("Week 3 Part 3: error analysis of the Metropolis runs (quantity: |M|)")
    lines.append("naive_err   = std(|M|)/sqrt(n)                      (assumes independent sweeps)")
    lines.append("blocked_err = std(50 block means, ddof=1)/sqrt(50)     (needs block_len >> tau_int)")
    lines.append("ratio       = blocked_err / naive_err")
    lines.append("tau_int     = 1/2 + sum rho(t) on |M|, sum stopped once lag > 6 * running tau_int")
    lines.append("block_len/tau: how many autocorrelation times one block spans; >> 1 is required")
    lines.append("NOTE: the 50-block estimate is NOT automatically final or honest; near T_c,")
    lines.append("      where block_len/tau approaches 1, it still underestimates the true error.")
    lines.append("")

    for kind, n_sweeps in (("coarse", 5000), ("window", 100000)):
        for L in SIZES:
            series = load_series(ARTIFACTS / f"{kind}-l{L}" / "series.jsonl")
            analyze(f"{kind} runs, {n_sweeps} measured sweeps", series, L, lines)

    report = "\n".join(lines) + "\n"
    print(report, end="")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "errors.txt"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
