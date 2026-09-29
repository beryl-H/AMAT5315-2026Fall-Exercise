#!/usr/bin/env python3
"""Compare the last Taylor-Green frame with the course exact field.

Reads week4/artifacts/taylor-green/fields.jsonl (simulation snapshots,
stored at 6 decimals) and week4/artifacts/taylor-green/exact-t1.json (the
course-provided exact velocity field at t = 1). Prints the relative
velocity-field error and draws the vorticity at t = 0 and t = 1 with
velocity arrows on a shared colour scale.

Course convention: u = cos(x) sin(y) exp(-2 nu t),
v = -sin(x) cos(y) exp(-2 nu t), omega = -2 cos(x) cos(y) exp(-2 nu t).
The simulation artifacts are used as stored; nothing is re-run.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python compare_taylor_green.py

Saves week4/evidence/taylor-green.png.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "artifacts" / "taylor-green"

with open(RUN / "fields.jsonl") as f:
    frames = [json.loads(line) for line in f if line.strip()]
with open(RUN / "exact-t1.json") as f:
    exact = json.load(f)

n = exact["n"]
nu = json.loads((RUN / "run.json").read_text())["nu"]

first, last = frames[0], frames[-1]
assert last["t"] == 1.0, f"last frame is t = {last['t']}"

u_c = np.asarray(last["u"]).reshape(n, n)
v_c = np.asarray(last["v"]).reshape(n, n)
u_e = np.asarray(exact["u"]).reshape(n, n)
v_e = np.asarray(exact["v"]).reshape(n, n)

# sanity: the exact field follows the course formula at t = 1
x = np.array([2 * np.pi * i / n for i in range(n)])
y = np.array([2 * np.pi * j / n for j in range(n)])
X, Y = np.meshgrid(x, y)
u_formula = np.cos(X) * np.sin(Y) * np.exp(-2 * nu)
v_formula = -np.sin(X) * np.cos(Y) * np.exp(-2 * nu)
formula_check = np.abs(u_e - u_formula).max() + np.abs(v_e - v_formula).max()

# relative velocity-field error (discrete L2 over both components)
num = np.sqrt(((u_c - u_e) ** 2).sum() + ((v_c - v_e) ** 2).sum())
den = np.sqrt((u_e**2).sum() + (v_e**2).sum())
rel_l2 = num / den
linf = max(np.abs(u_c - u_e).max(), np.abs(v_c - v_e).max())

print(f"frames: {len(frames)}, last t = {last['t']}, step = {last['step']}")
print(f"exact field matches course formula to {formula_check:.3e} (max abs)")
print(f"relative velocity-field error (L2): {rel_l2:.6e}")
print(f"relative velocity-field error (Linf): {linf / den:.6e}"
      f"  (max abs {linf:.6e})")
print("note: stored fields are rounded to 6 decimals in fields.jsonl,"
      " which dominates this error")

# --- figure: vorticity at t = 0 and t = 1 with velocity arrows --------------
w0 = np.asarray(first["omega"]).reshape(n, n)
w1 = np.asarray(last["omega"]).reshape(n, n)
u0 = np.asarray(first["u"]).reshape(n, n)
v0 = np.asarray(first["v"]).reshape(n, n)
lim = max(abs(w0.min()), abs(w0.max()), abs(w1.min()), abs(w1.max()))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
stride = 4  # arrow subsampling for readability
for ax, (frame, w, label) in zip(
    axes, [(first, w0, "t = 0"), (last, w1, "t = 1")]
):
    uu = np.asarray(frame["u"]).reshape(n, n)
    vv = np.asarray(frame["v"]).reshape(n, n)
    im = ax.pcolormesh(X, Y, w.reshape(n, n), cmap="RdBu_r",
                       vmin=-lim, vmax=lim, shading="auto")
    ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
              uu[::stride, ::stride],
              vv[::stride, ::stride],
              color="black", scale=25, width=0.003, alpha=0.85)
    ax.set_title(f"vorticity and velocity, {label} (RK4, nu = {nu})")
    ax.set_xlabel("x")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
axes[0].set_ylabel("y")
cbar = fig.colorbar(im, ax=axes, pad=0.02)
cbar.set_label(r"$\omega$ (shared colour scale)")

fig.suptitle(
    "Taylor-Green vortex, n = 64: stored simulation vs course exact field,"
    f" relative velocity error {rel_l2:.2e}",
    fontsize=12,
)

out = ROOT / "evidence" / "taylor-green.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
