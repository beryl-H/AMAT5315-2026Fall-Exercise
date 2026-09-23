#!/usr/bin/env python3
"""Final profiles of the Week 4 accuracy comparison on one periodic lap.

Gaussian pulse (sigma = 0.25, centred at x = pi/2, n = 64, c = 1,
nu = 0.002) integrated to t = 2*pi (one lap) by the advdiff library:

  1. RK4,  dt = 0.02,  Fourier derivatives
  2. RK4,  dt = 0.02,  centred finite differences
  3. Euler, dt = 0.005, Fourier derivatives

The exact solution (Gaussian back at x = pi/2, diffused to
sigma_t = sqrt(sigma^2 + 2 nu t)) is the black reference. Data:
evidence/line_accuracy.json from the run_accuracy binary. The maximum
error of each run is shown in the legend.

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

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "evidence" / "line_accuracy.json").read_text())

x = np.asarray(data["x"])
exact = np.asarray(data["exact"])

STYLES = [
    ("tab:blue", "o", "RK4, Fourier, dt = 0.02"),
    ("tab:orange", "s", "RK4, centred FD, dt = 0.02"),
    ("tab:green", "^", "Euler, Fourier, dt = 0.005"),
]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(x, exact, "k-", lw=2.0, label="exact (one lap, diffused)")
for run, (color, marker, label) in zip(data["runs"], STYLES):
    u = np.asarray(run["u"])
    ax.plot(x, u, marker=marker, ms=4, lw=1.0, color=color,
            label=f"{label} — max err {run['max_err']:.2e}")

ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
              ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_xlim(0.0, 2 * np.pi)
ax.set_ylim(-0.08, 0.85)
ax.grid(alpha=0.3)
ax.set_title(
    "Gaussian pulse after one lap (t = 2$\\pi$): $\\sigma=0.25$, "
    r"$n=64$, $c=1$, $\nu=0.002$"
)
ax.legend(loc="upper right", fontsize=9)

out = ROOT / "evidence" / "line-accuracy.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
