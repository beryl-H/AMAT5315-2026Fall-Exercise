#!/usr/bin/env python3
"""Plot the Lennard-Jones pair field around a single atom.

The radial energy and force values come from the ``md`` crate in
``week2/md/`` (its ``lj_field`` example), so the figure uses exactly the
functions implemented there.  Run from the repository root:

    MPLCONFIGDIR=/tmp/matplotlib python3 week2/plot_lj_field.py

Saves ``week2/field.png``.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "week2" / "md"
OUT = ROOT / "week2" / "field.png"

# ----------------------------------------------------------------------
# 1. Radial data from the md crate (energy and scalar force vs. radius).
# ----------------------------------------------------------------------

raw = subprocess.check_output(
    ["cargo", "run", "--quiet", "--example", "lj_field"],
    cwd=CRATE,
    text=True,
)
rows = np.loadtxt(raw.splitlines())
radii, energies, forces = rows[:, 0], rows[:, 1], rows[:, 2]

# ----------------------------------------------------------------------
# 2. Energy sampled on a 2-D grid (colors) and force vectors (arrows).
# ----------------------------------------------------------------------

extent = 3.0
n = 600
x = np.linspace(-extent, extent, n)
X, Y = np.meshgrid(x, x)
R = np.hypot(X, Y)

energy_field = np.interp(R.ravel(), radii, energies).reshape(R.shape)

# Force arrows on a sparse grid, offset so no arrow sits exactly at the
# minimum where the scalar force changes sign.
xq, yq = np.meshgrid(np.arange(-2.8, 2.81, 0.4), np.arange(-2.8, 2.81, 0.4))
rq = np.hypot(xq, yq)
force_scalar = np.interp(rq.ravel(), radii, forces).reshape(rq.shape)

# Unit vector pointing away from the central atom.
ux = np.divide(xq, rq, out=np.zeros_like(xq), where=rq > 0)
uy = np.divide(yq, rq, out=np.zeros_like(yq), where=rq > 0)

# Arrow direction: outward where the scalar force is repulsive (positive),
# inward where it is attractive (negative).  Length grows logarithmically
# with |force| so both the weak tail and the steep core stay readable.
with np.errstate(divide="ignore"):
    arrow_len = np.log10(1.0 + np.abs(force_scalar))
arrow_len = np.clip(arrow_len / 2.0, 0.06, 1.0)
sign = np.sign(force_scalar)
qx = sign * arrow_len * ux
qy = sign * arrow_len * uy

# ----------------------------------------------------------------------
# 3. Diverging energy colour scale.
#
# Positive energies (repulsive wall) and negative energies (attractive
# well) are both shown on logarithmic branches that meet linearly at
# V = 0, so zero is white, the well (V = -1) is deep blue, and the
# repulsive core is red.
# ----------------------------------------------------------------------

vmax = 1.0  # colour scale saturates for V >= 1 (well below the core wall)
Eclip = np.clip(energy_field, -1.0, vmax)

def branch(E: np.ndarray) -> np.ndarray:
    """Signed branch value: log_2(1 + |V|) for |V| < 1, linear past 1."""
    b = np.log2(1.0 + np.abs(E))
    return np.sign(E) * b

s = branch(Eclip)
snorm = branch(np.asarray([-1.0, 0.0, vmax]))
lo, mid, hi = snorm[0], snorm[1], snorm[2]
t = np.where(s < 0, 0.5 * (s - lo) / (mid - lo), 0.5 + 0.5 * (s - mid) / (hi - mid))
t = np.clip(t, 0.0, 1.0)

# White | light blue | saturated blue (attractive), then white | red
# (repulsive): diverging with white at V = 0.
stop_t = np.array([0.00, 0.30, 0.48, 0.52, 0.68, 1.00])
stop_rgb = np.array(
    [
        [0.035, 0.075, 0.365],  # deep blue: strong attraction (well)
        [0.145, 0.365, 0.635],  # medium blue
        [0.85, 0.90, 0.98],     # pale blue near V = 0
        [0.99, 0.94, 0.90],     # pale red near V = 0
        [0.98, 0.55, 0.29],     # orange: weak repulsion
        [0.55, 0.055, 0.065],   # deep red: strong repulsion (core)
    ]
)

def color_map(u: np.ndarray) -> np.ndarray:
    out = np.empty((*u.shape, 3))
    for ch in range(3):
        out[..., ch] = np.interp(u, stop_t, stop_rgb[:, ch])
    return out

colors = color_map(t)

# ----------------------------------------------------------------------
# 4. Figure.
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.0, 7.4))
ax.imshow(colors, origin="lower", extent=(-extent, extent, -extent, extent))

ax.quiver(
    xq,
    yq,
    qx,
    qy,
    color="k",
    width=0.004,
    headwidth=3.6,
    headlength=4.5,
    angles="xy",
    scale_units="xy",
    scale=0.6,
    zorder=3,
)

# Central atom.
ax.plot(0, 0, "o", ms=13, mfc="0.12", mec="white", mew=1.2, zorder=5)
ax.annotate(
    "atom",
    xy=(0.13, 0.13),
    xytext=(0.55, 0.55),
    fontsize=9,
    color="0.15",
    arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
)

ax.set_xlim(-extent, extent)
ax.set_ylim(-extent, extent)
ax.set_aspect("equal")
ax.set_xlabel("$x/\\sigma$")
ax.set_ylabel("$y/\\sigma$")
ax.set_title(
    "Lennard-Jones pair field around one atom\n"
    "colors: pair energy  |  arrows: pair force"
)

# Manual colour bar along the top.
tick_vals = [-1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0]
tick_t = np.interp(branch(np.asarray(tick_vals)), [lo, hi], [0.0, 1.0])
cb_ax = fig.add_axes([0.145, 0.915, 0.72, 0.028])
cb_img = np.linspace(0.0, 1.0, 512)[None, :]
cb_ax.imshow(color_map(cb_img), aspect="auto", origin="lower")
cb_ax.set_xticks(tick_t)
cb_ax.set_xticklabels([f"{v:g}" for v in tick_vals], fontsize=8)
cb_ax.tick_params(length=0)
for spine in cb_ax.spines.values():
    spine.set_visible(False)
cb_ax.set_title("pair energy $V(r)$ ($\\epsilon$)", fontsize=9, pad=6)

# Legend for arrows.
ax.plot([], [], "v", color="k", markersize=7, label="force points away (repulsive)")
ax.plot([], [], "^", color="k", markersize=7, label="force points inward (attractive)")
ax.plot([], [], "o", color="k", label="arrow length $\\propto \\log_{10}|F|$")
leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
