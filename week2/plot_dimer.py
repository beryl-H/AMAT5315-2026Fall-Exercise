#!/usr/bin/env python3
"""Plot the two-atom dimer relative energy error (Part 3 verification).

All series come from the ``md`` crate in ``week2/md/`` (its
``dimer_energy`` example, which drives the shared ``run_experiment``
with Forward Euler and velocity-Verlet), so the figure uses exactly the
simulator implementation verified by the test suite.  Run from the
repository root:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/ljplotenv/bin/python week2/plot_dimer.py

Saves ``week2/dimer.png``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "week2" / "md"
OUT = ROOT / "week2" / "dimer.png"

# ----------------------------------------------------------------------
# 1. Energy-error series from the md crate.
# ----------------------------------------------------------------------

raw = subprocess.check_output(
    ["cargo", "run", "--quiet", "--example", "dimer_energy"],
    cwd=CRATE,
    text=True,
)

runs: dict[str, np.ndarray] = {}
name: str | None = None
rows: list[list[float]] = []
for line in raw.splitlines():
    if line.startswith("# run="):
        if name is not None:
            runs[name] = np.asarray(rows)
        name = dict(part.split("=", 1) for part in line[2:].split())["name"]
        rows = []
    elif line.strip():
        rows.append([float(v) for v in line.split()])
if name is not None:
    runs[name] = np.asarray(rows)

euler500 = runs["euler-500"]
verlet500 = runs["verlet-500"]
verlet5000 = runs["verlet-5000"]

# Sanity checks against the reference table in the approved design
# (docs/superpowers/specs/2026-09-09-week2-md-two-atom-design.md).
assert abs(euler500[-1, 3] - 1.932) < 0.01, "Euler final error off reference"
assert np.abs(verlet500[:, 3]).max() < 1e-3, "Verlet 500 max error off reference"
assert np.abs(verlet5000[:, 3]).max() < 1e-3, "Verlet 5000 max error off reference"

# ----------------------------------------------------------------------
# 2. Figure: 500-step comparison panel and long velocity-Verlet run.
# ----------------------------------------------------------------------

fig, (ax_compare, ax_long) = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax_compare.plot(
    euler500[:, 2], euler500[:, 3] * 1e3, color="tab:red", label="Forward Euler"
)
ax_compare.plot(
    verlet500[:, 2], verlet500[:, 3] * 1e3, color="tab:blue", label="velocity-Verlet"
)
ax_compare.axhline(0.0, color="0.7", lw=0.8, zorder=0)
ax_compare.set_title("500 steps, $\\mathrm{d}t = 0.01$")
ax_compare.set_xlabel("time $t$")
ax_compare.set_ylabel("relative energy error $\\times\\,10^{-3}$")
ax_compare.legend(loc="upper left")

ax_long.plot(verlet5000[:, 2], verlet5000[:, 3] * 1e3, color="tab:blue")
ax_long.axhline(0.0, color="0.7", lw=0.8, zorder=0)
ax_long.set_title("velocity-Verlet, 5000 steps, $\\mathrm{d}t = 0.01$")
ax_long.set_xlabel("time $t$")
ax_long.set_ylabel("relative energy error $\\times\\,10^{-3}$")

fig.suptitle(
    "Two-atom Lennard-Jones dimer: relative total-energy error "
    "$(E(t) - E_0)\\,/\\,|E_0|$"
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
