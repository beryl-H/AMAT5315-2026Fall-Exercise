#!/usr/bin/env python3
"""Week 3 Part 3: |M| traces of the L = 64 Metropolis runs.

Draws the absolute per-spin magnetization |M| over the first 2000 recorded
sweeps at T = 2.3 (inside the critical region) and T = 3.0 (paramagnetic
phase) for L = 64, from the completed coarse Metropolis run in
week3/artifacts/coarse-l64/series.jsonl.

The two traces make the Part 3 error-analysis lesson visible: near the
critical temperature the |M| series carries slow, large excursions (long
autocorrelation), while deep in the paramagnetic phase it fluctuates fast
around a small mean. Equal-length error bars on both would therefore rest
on very different amounts of independent information.

Output: week3/evidence/trace.png

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/plot_trace.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK3 = Path(__file__).resolve().parent.parent
ARTIFACTS = WEEK3 / "artifacts"
EVIDENCE = WEEK3 / "evidence"

L = 64
N_SWEEPS = 2000
SOURCES = {
    2.3: ARTIFACTS / "window-l64" / "series.jsonl",
    3.0: ARTIFACTS / "coarse-l64" / "series.jsonl",
}


def load_trace(path: Path, T: float, n: int) -> tuple[list[int], list[float]]:
    """First n recorded sweeps of |M| at temperature T."""
    sweeps, abs_m = [], []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["T"] == T:
                sweeps.append(row["sweep"])
                abs_m.append(abs(row["M"]))
                if len(sweeps) == n:
                    break
    if len(sweeps) < n:
        raise ValueError(f"only {len(sweeps)} recorded sweeps at T={T} in {path}")
    return sweeps, abs_m


def main() -> None:
    traces = {T: load_trace(p, T, N_SWEEPS) for T, p in SOURCES.items()}

    fig, axes = plt.subplots(
        len(SOURCES), 1, figsize=(10.0, 6.0), sharex=True
    )
    for ax, (T, source) in zip(axes, SOURCES.items()):
        sweeps, abs_m = traces[T]
        ax.plot(sweeps, abs_m, lw=0.4, color="tab:orange" if T == 2.3 else "tab:blue")
        ax.set_ylabel(r"$|M|$ per spin")
        ax.set_title(
            f"L = {L}, T = {T:.1f}, first {len(sweeps)} recorded sweeps "
            f"({source.parent.name})",
            fontsize=10,
        )
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("recorded sweep (per-temperature numbering, after 2000 discarded)")
    fig.suptitle(
        f"Metropolis Ising {L}x{L}: |M| traces at T = 2.3 (critical region) and T = 3.0",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "trace.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
