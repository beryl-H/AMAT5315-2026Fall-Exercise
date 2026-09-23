#!/usr/bin/env python3
"""Stability plane and Gaussian-pulse evolution of the Week 4 integrators.

Three panels (saved as one figure, week4/evidence/line_stability.png):

1. Complex plane of z = lambda*h coloured by the growth factor per step
   MEASURED by the advdiff library: the measure_growth binary advanced
   y' = lambda*y one step of h = 1 from y0 = 1 with the library RK4 on a grid
   of z values (evidence/line_growth.json). The |R(z)| = 1 contours for all
   three integrators are contours of those measured fields at level 1 -- no
   stability polynomial is hard-coded. Dots: Fourier pseudo-spectral
   eigenmodes lambda_k = -i c k - nu k^2 of the advection-diffusion line
   (Nyquist k = n/2 does not travel: lambda = -nu k^2) at dt = 0.045, 0.056.

2-3. Space-time evolution u(x, t), x across and t downward, of a Gaussian
   pulse (sigma = 0.35, centred at x = pi/2, periodic on the box) integrated
   by the library RK4 to t = 6 at nu = 0.05, n = 64, c = 1, once at
   dt = 0.045 and once at dt = 0.056 (data: evidence/line_evolution.json;
   full steps of the requested dt plus one shorter final step land exactly
   on t = 6). Shared symlog colour scale: at dt = 0.056 the modes near the
   Nyquist edge (|R| = 1.85 per step at k = +/-31) blow up from the initial
   condition's spectral floor (~1.4e-7, seeded by the Gaussian's tiny
   non-periodicity at the box seam) to ~1e21 by t = 6, while dt = 0.045
   stays smooth.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_line_stability.py
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
growth_data = json.loads((ROOT / "evidence" / "line_growth.json").read_text())
evo_data = json.loads((ROOT / "evidence" / "line_evolution.json").read_text())

x = np.asarray(growth_data["x"])
y = np.asarray(growth_data["y"])
X, Y = np.meshgrid(x, y)  # rows of the stored fields vary with y
growth = {name: np.asarray(field) for name, field in growth_data["growth"].items()}

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

fig = plt.figure(figsize=(13.0, 7.0))
gs = GridSpec(2, 2, width_ratios=[1.15, 1.0], hspace=0.35, wspace=0.22)
ax_stab = fig.add_subplot(gs[:, 0])
ax_evo = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

# --- Panel 1: measured stability plane -------------------------------------
field = np.ma.masked_less_equal(growth["rk4"], 0.0)
mesh = ax_stab.pcolormesh(
    X, Y, field,
    norm=LogNorm(vmin=1e-2, vmax=float(growth["rk4"].max())),
    cmap="viridis", shading="auto",
)
cbar = fig.colorbar(mesh, ax=ax_stab, pad=0.02)
cbar.set_label(r"measured growth $|y_1|$ per step (library RK4, $h=1$)")

CONTOUR_STYLE = {
    "euler": ("tab:red", "--"),
    "midpoint": ("tab:orange", "-"),
    "rk4": ("white", "-"),
}
for name, (color, ls) in CONTOUR_STYLE.items():
    ax_stab.contour(X, Y, growth[name], levels=[1.0], colors=[color],
                    linestyles=ls, linewidths=1.8)

for dt, marker, color in STEPS:
    z = lambdas * dt
    ax_stab.scatter(z.real, z.imag, s=28, marker=marker, facecolor=color,
                    edgecolor="black", linewidths=0.4, zorder=5)
    # the Nyquist mode is the leftmost, purely real dot
    z_nyq = (-NU * (N // 2) ** 2) * dt
    ax_stab.annotate(
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
ax_stab.legend(handles=legend_items, loc="upper left", fontsize=7.5,
               framealpha=0.9)

ax_stab.axhline(0.0, color="gray", lw=0.5)
ax_stab.axvline(0.0, color="gray", lw=0.5)
ax_stab.set_xlabel(r"$\Re(\lambda h)$")
ax_stab.set_ylabel(r"$\Im(\lambda h)$")
ax_stab.set_aspect("equal")
ax_stab.set_xlim(x[0], x[-1])
ax_stab.set_ylim(y[0], y[-1])
ax_stab.set_title("measured one-step growth (library RK4)\nwith $|R(z)|=1$ of all three integrators + line modes",
                  fontsize=10)

# --- Panels 2-3: Gaussian pulse evolution -----------------------------------
xe = np.asarray(evo_data["x"])
u_max = max(np.abs(np.asarray(run["u"])).max() for run in evo_data["runs"])
# Shared symlog scale clipped at |u| = 1: the smooth pulse (peak ~0.41 after
# diffusion) is visible in both panels; in the unstable run the k=±31
# sawtooth saturates once |u| exceeds 1 (its true maximum is annotated).
CLIP = 1.0
norm = SymLogNorm(linthresh=0.05, linscale=1.0, vmin=-CLIP, vmax=CLIP)
meshes = []
for ax, run, (dt_ref, _, color) in zip(ax_evo, evo_data["runs"], STEPS):
    times = np.asarray(run["times"])
    U = np.asarray(run["u"])
    TT, XX = np.meshgrid(times, xe, indexing="ij")
    meshes.append(ax.pcolormesh(XX, TT, U, norm=norm, cmap="RdBu_r", shading="auto"))
    # analytic centre path of the pulse, x_c(t) = (pi/2 + c t) mod 2pi
    t_line = np.linspace(0.0, run["times"][-1], 200)
    x_line = (np.pi / 2 + C * t_line) % (2 * np.pi)
    breaks = np.where(np.diff(x_line) < -np.pi)[0]
    if breaks.size:
        i = breaks[0]
        ax.plot(np.concatenate([x_line[:i + 1], [np.nan], x_line[i + 1:]]),
                np.concatenate([t_line[:i + 1], [np.nan], t_line[i + 1:]]),
                "k--", lw=1.0, alpha=0.7)
    else:
        ax.plot(x_line, t_line, "k--", lw=1.0, alpha=0.7)
    ax.set_ylabel("t")
    ax.invert_yaxis()  # t grows downward
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    if dt_ref == 0.045:
        ax.set_title("RK4, dt = 0.045: all modes inside RK4 region",
                     fontsize=10, color=color)
    else:
        ax.set_title(f"RK4, dt = 0.056: unstable near Nyquist edge\n"
                     f"(max |u| = {u_max:.2e} at t = 6)", fontsize=9.5, color=color)
        ax.set_xlabel("x")
cbar2 = fig.colorbar(meshes[1], ax=ax_evo, pad=0.02)
cbar2.set_label(f"$u(x,t)$ (symlog, shared, clipped at $\\pm{CLIP:g}$)")

fig.suptitle(
    "Week 4: stability of the line integrators and Gaussian-pulse evolution "
    r"($\nu=0.05$, $c=1$, $n=64$)",
    fontsize=12,
)

out = ROOT / "evidence" / "line_stability.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
