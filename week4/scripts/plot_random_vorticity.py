#!/usr/bin/env python3
"""Draw the vorticity of the stored random run at t = 0, 2, 5, 10.

Reads week4/artifacts/random/fields.jsonl (simulation snapshots, stored at
6 decimals) and week4/artifacts/random/run.json. Uses the omega field as
stored; nothing is re-run and the simulation is not modified.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_random_vorticity.py

Saves week4/evidence/random.png.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "artifacts" / "random"

with open(RUN / "fields.jsonl") as f:
    frames = [json.loads(line) for line in f if line.strip()]
run = json.loads((RUN / "run.json").read_text())

n = run["n"]
nu = run["nu"]
WANTED = (0.0, 2.0, 5.0, 10.0)
picked = []
for t in WANTED:
    match = [fr for fr in frames if abs(fr["t"] - t) < 1e-9]
    assert match, f"no stored frame at t = {t}"
    picked.append(match[0])
assert picked[-1]["t"] == 10.0, "run does not reach t = 10"

x = np.array([2 * np.pi * i / n for i in range(n)])
y = np.array([2 * np.pi * j / n for j in range(n)])
X, Y = np.meshgrid(x, y)

fields = [np.asarray(fr["omega"]).reshape(n, n) for fr in picked]
lim = max(max(abs(w.min()), abs(w.max())) for w in fields)

fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.9), sharey=True)
for ax, fr, w in zip(axes, picked, fields):
    im = ax.pcolormesh(X, Y, w, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       shading="auto")
    ax.set_title(f"t = {fr['t']:g} (step {fr['step']})")
    ax.set_xlabel("x")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
axes[0].set_ylabel("y")
cbar = fig.colorbar(im, ax=axes, pad=0.02)
cbar.set_label(r"$\omega$ (shared colour scale)")

fig.suptitle(
    f"Random-band flow (k in [{run['k_band'][0]:g}, {run['k_band'][1]:g}]),"
    f" n = {n}, RK4, dt = {run['dt']}, nu = {nu}:"
    " stored vorticity snapshots",
    fontsize=12,
)

out = ROOT / "evidence" / "random.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
print(f"shared colour scale: omega in [-{lim:.4f}, {lim:.4f}]")
for fr, w in zip(picked, fields):
    print(f"  t = {fr['t']:>5g}: min {w.min():+.4f}, max {w.max():+.4f}")
