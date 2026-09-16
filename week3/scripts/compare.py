#!/usr/bin/env python3
"""Week 3 Part 3: work-normalized autocorrelation time, L = 64.

Compares the Metropolis window run (window-l64, one step = one l*l-proposal
sweep) with the Wolff run (wolff-l64, one step = one cluster flip) in
week3/artifacts/ by drawing the work-normalized integrated autocorrelation
time of |M| against temperature:

    tau_work(T) = tau_int(T) * cost(T),

where tau_int uses the Week 3 rule (1/2 + sum rho(t), sum stopped once the
lag exceeds six times the running tau_int, as in scripts/errors.py) and
the cost is measured in spin-update sweeps:

- metropolis: one step is already one sweep, cost = 1;
- wolff: one cluster move costs mean cluster size / L^2 sweeps.

Error bars are jackknife estimates over 20 contiguous blocks: each
leave-one-block-out replica recomputes both tau_int and the mean cluster
cost, so the bars include the sensitivity of the stopping rule and, for
wolff, of the cost estimate.

Output: week3/evidence/tau-compare.png (logarithmic vertical axis).

Reproduce from the repository root with:

    /home/keria_h/.venvs/amat5315/bin/python week3/scripts/compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from errors import tau_int  # noqa: E402  (same scripts/ directory)
from plot_order_parameters import (  # noqa: E402  (same scripts/ directory)
    ARTIFACTS,
    EVIDENCE,
    TC,
)

L = 64
N_BLOCKS = 20
RUNS = {
    "metropolis (window-l64)": ARTIFACTS / "window-l64" / "series.jsonl",
    "wolff (wolff-l64)": ARTIFACTS / "wolff-l64" / "series.jsonl",
}


def load_m_and_cluster(path: Path, T: float) -> tuple[np.ndarray, np.ndarray]:
    """|M| and cluster_size series at temperature T (cluster_size only in
    wolff rows; metropolis rows return an empty array)."""
    abs_m, cluster = [], []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["T"] == T:
                abs_m.append(abs(row["M"]))
                if "cluster_size" in row:
                    cluster.append(row["cluster_size"])
    return np.asarray(abs_m), np.asarray(cluster)


def tau_work_with_error(abs_m: np.ndarray, cluster: np.ndarray, l: int) -> tuple[float, float, float]:
    """tau_int, work-normalized tau and its jackknife error.

    cost = 1 sweep per step for metropolis (empty cluster series);
    cost = mean cluster size / l^2 sweeps per flip for wolff.
    """
    blocks_m = np.array_split(abs_m, N_BLOCKS)
    is_wolff = cluster.size > 0
    blocks_c = np.array_split(cluster, N_BLOCKS) if is_wolff else None

    tau = tau_int(abs_m)
    cost = cluster.mean() / (l * l) if is_wolff else 1.0
    tau_work = tau * cost

    replicas = []
    for j in range(N_BLOCKS):
        kept_m = np.concatenate([blocks_m[i] for i in range(N_BLOCKS) if i != j])
        tau_j = tau_int(kept_m)
        if is_wolff:
            kept_c = np.concatenate([blocks_c[i] for i in range(N_BLOCKS) if i != j])
            cost_j = kept_c.mean() / (l * l)
        else:
            cost_j = 1.0
        replicas.append(tau_j * cost_j)
    replicas = np.asarray(replicas)
    err = np.sqrt(
        (N_BLOCKS - 1) / N_BLOCKS * ((replicas - replicas.mean()) ** 2).sum()
    )
    return tau, tau_work, float(err)


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    colors = {"metropolis (window-l64)": "tab:blue", "wolff (wolff-l64)": "tab:orange"}

    for label, path in RUNS.items():
        # Grid from the first temperature's rows; every T has the same
        # number of measured steps.
        temps = sorted({json.loads(l)["T"] for l in open(path)})
        tw, twe = [], []
        for t in temps:
            abs_m, cluster = load_m_and_cluster(path, t)
            _, work, err = tau_work_with_error(abs_m, cluster, L)
            tw.append(work)
            twe.append(err)
            print(f"{label}  T={t:.2f}  tau_work = {work:9.2f} +- {err:8.2f} sweeps")
        ax.errorbar(temps, tw, yerr=twe, marker="o", ms=4, lw=1.3,
                    capsize=2, color=colors[label], label=label)

    ax.axvline(TC, color="gray", ls="--", lw=1.0, alpha=0.8)
    ax.text(TC, ax.get_ylim()[1], f"  $T_c$ = 2.26919", va="top",
            fontsize=8, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("temperature  $T$")
    ax.set_ylabel(r"$\tau_{int}\cdot$cost of $|M|$  (spin-update sweeps)")
    ax.set_title(
        f"Week 3 Part 3: work-normalized autocorrelation time, L = {L} "
        f"(jackknife errors, {N_BLOCKS} blocks)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "tau-compare.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
