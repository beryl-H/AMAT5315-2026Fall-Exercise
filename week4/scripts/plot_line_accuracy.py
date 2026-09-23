#!/usr/bin/env python3
"""Accuracy evidence for the Week 4 integrators, two panels in one figure.

Panel 1 -- final profiles on one periodic lap: Gaussian pulse
(sigma = 0.25, centred at x = pi/2, n = 64, c = 1, nu = 0.002) integrated to
t = 2*pi by the advdiff library with RK4/Fourier, RK4/centred-FD, and
Euler/Fourier; the exact solution (Gaussian back at x = pi/2, diffused to
sigma_t = sqrt(sigma^2 + 2 nu t)) is the black reference. Data:
evidence/line_accuracy.json.

Panel 2 -- convergence: max error at t = 1 against dt on log-log axes for
forward Euler, explicit midpoint, classical RK4, and equal-weight RK4
evolving the sigma = 0.35 pulse (nu = 0.05, c = 1, n = 64, Fourier
derivatives) against the exact semi-discrete solution, so the measured error
is purely temporal. A linear fit in log-log space is drawn for each method
with its fitted slope in the legend. Data: evidence/line_convergence.json.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_line_accuracy.py

Saves week4/evidence/line-accuracy.png.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
acc = json.loads((ROOT / "evidence" / "line_accuracy.json").read_text())
conv = json.loads((ROOT / "evidence" / "line_convergence.json").read_text())

x = np.asarray(acc["x"])
exact = np.asarray(acc["exact"])

STYLES = [
    ("tab:blue", "o", "RK4, Fourier, dt = 0.02"),
    ("tab:orange", "s", "RK4, centred FD, dt = 0.02"),
    ("tab:green", "^", "Euler, Fourier, dt = 0.005"),
]
CONV_STYLES = [
    ("tab:red", "o", "forward Euler"),
    ("tab:purple", "s", "explicit midpoint"),
    ("tab:blue", "^", "classical RK4"),
    ("tab:green", "D", "equal-weight RK4"),
]

fig = plt.figure(figsize=(13.0, 5.4))
gs = GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# --- Panel 1: final profiles -------------------------------------------------
ax1.plot(x, exact, "k-", lw=2.0, label="exact (one lap, diffused)")
for run, (color, marker, label) in zip(acc["runs"], STYLES):
    u = np.asarray(run["u"])
    ax1.plot(x, u, marker=marker, ms=4, lw=1.0, color=color,
             label=f"{label} — max err {run['max_err']:.2e}")
ax1.set_xlabel("x")
ax1.set_ylabel("u")
ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
               ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax1.set_xlim(0.0, 2 * np.pi)
ax1.set_ylim(-0.08, 0.95)
ax1.grid(alpha=0.3)
ax1.set_title(
    "one-lap profiles (t = 2$\\pi$): $\\sigma=0.25$, "
    r"$\nu=0.002$",
    fontsize=10.5,
)
ax1.legend(loc="upper right", fontsize=8.5)

# --- Panel 2: convergence ----------------------------------------------------
dts = np.asarray(conv["dts"])
for method, (color, marker, label) in zip(conv["methods"], CONV_STYLES):
    errs = np.asarray(method["errors"])
    ax2.loglog(dts, errs, marker=marker, ms=5, lw=0.8, color=color,
               label=label)
    slope, intercept = np.polyfit(np.log10(dts), np.log10(errs), 1)
    ax2.loglog(dts, 10.0 ** (intercept + slope * np.log10(dts)),
               ls="--", lw=1.2, color=color,
               label=f"fit: slope = {slope:.2f}")
ax2.set_xlabel("dt")
ax2.set_ylabel("max error at t = 1")
ax2.set_title(
    "convergence (t = 1, $\\sigma=0.35$, $\\nu=0.05$, Fourier)",
    fontsize=10.5,
)
ax2.grid(alpha=0.3, which="both")
# one legend with method points and their fits: merge duplicate colours
handles, labels_ = ax2.get_legend_handles_labels()
ax2.legend(handles, labels_, fontsize=8, loc="upper left")

fig.suptitle("Week 4: accuracy of the line integrators (advdiff library)",
             fontsize=13)

out = ROOT / "evidence" / "line-accuracy.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
