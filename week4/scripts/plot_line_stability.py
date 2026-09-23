#!/usr/bin/env python3
"""Stability plane of the Week 4 integrators for the advection-diffusion line.

The coloured field is the growth factor |y1| per step MEASURED by the advdiff
library: the measure_growth binary advanced y' = lambda*y one step of h = 1
from y0 = 1 with the library RK4 on a grid of z = lambda*h values (see
evidence/line_growth.json). The |R(z)| = 1 contours for all three integrators
are contours of those measured fields at level 1 -- no stability polynomial is
hard-coded. The dots are the Fourier pseudo-spectral eigenmodes of the
advection-diffusion line, lambda_k = -i c k - nu k^2 (Nyquist k = n/2 does not
travel: lambda = -nu k^2), scaled by z = lambda*dt for dt = 0.045 and 0.056.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_line_stability.py

Saves week4/evidence/line_stability.png.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "evidence" / "line_growth.json").read_text())
x = np.asarray(data["x"])
y = np.asarray(data["y"])
X, Y = np.meshgrid(x, y)  # rows of the stored fields vary with y
growth = {name: np.asarray(field) for name, field in data["growth"].items()}

# Fourier spectral eigenvalues of the line: nu = 0.05, c = 1, n = 64.
C, NU, N = 1.0, 0.05, 64
lambdas = []
for j in range(N):
    if j < N // 2:
        k = float(j)
        lambdas.append(complex(-NU * k * k, -C * k))
    elif j == N // 2:
        # Nyquist mode: first derivative zero, only diffusion acts.
        lambdas.append(complex(-NU * (N // 2) ** 2, 0.0))
    else:
        k = float(j - N)
        lambdas.append(complex(-NU * k * k, -C * k))
lambdas = np.asarray(lambdas)

STEPS = [(0.045, "o", "tab:blue"), (0.056, "s", "tab:red")]

fig, ax = plt.subplots(figsize=(8.0, 6.8))

field = np.ma.masked_less_equal(growth["rk4"], 0.0)
vmax = float(growth["rk4"].max())
mesh = ax.pcolormesh(
    X, Y, field,
    norm=LogNorm(vmin=1e-2, vmax=vmax),
    cmap="viridis", shading="auto",
)
cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
cbar.set_label(r"measured growth $|y_1|$ per step (library RK4, $h=1$)")

CONTOUR_STYLE = {
    "euler": ("tab:red", "--"),
    "midpoint": ("tab:orange", "-"),
    "rk4": ("white", "-"),
}
for name, (color, ls) in CONTOUR_STYLE.items():
    ax.contour(X, Y, growth[name], levels=[1.0], colors=[color],
               linestyles=ls, linewidths=1.8)

for dt, marker, color in STEPS:
    z = lambdas * dt
    ax.scatter(z.real, z.imag, s=28, marker=marker, facecolor=color,
               edgecolor="black", linewidths=0.4, zorder=5)
    # the Nyquist mode is the leftmost, purely real dot
    z_nyq = (-NU * (N // 2) ** 2) * dt
    ax.annotate(
        f"Nyquist, dt = {dt}: z = {z_nyq:.3f}",
        xy=(z_nyq, 0.0), xytext=(z_nyq - 0.05, 0.85 if dt == 0.045 else -1.05),
        fontsize=8, color="black", ha="center",
        bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.5),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
    )

legend_items = [
    Line2D([], [], color=CONTOUR_STYLE["euler"][0], ls="--", lw=1.8,
           label=r"Euler $|1+z|=1$ (measured)"),
    Line2D([], [], color=CONTOUR_STYLE["midpoint"][0], ls="-", lw=1.8,
           label=r"Midpoint $|1+z+z^2/2|=1$ (measured)"),
    Line2D([], [], color="white", ls="-", lw=1.8,
           label=r"RK4 $|1+z+z^2/2+z^3/6+z^4/24|=1$ (measured)"),
    Line2D([], [], marker="o", ls="none", mfc="tab:blue", mec="black", ms=7,
           label=r"modes $z=\lambda_k\,dt$, $dt=0.045$"),
    Line2D([], [], marker="s", ls="none", mfc="tab:red", mec="black", ms=7,
           label=r"modes $z=\lambda_k\,dt$, $dt=0.056$"),
]
ax.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.9)

ax.axhline(0.0, color="gray", lw=0.5)
ax.axvline(0.0, color="gray", lw=0.5)
ax.set_xlabel(r"$\Re(\lambda h)$")
ax.set_ylabel(r"$\Im(\lambda h)$")
ax.set_aspect("equal")
ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])
ax.set_title(
    "Measured one-step growth $|y_1|$ (library RK4) with $|R(z)|=1$ of all\n"
    "three integrators and advection-diffusion modes "
    r"($\nu=0.05$, $c=1$, $n=64$)"
)

out = ROOT / "evidence" / "line_stability.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
