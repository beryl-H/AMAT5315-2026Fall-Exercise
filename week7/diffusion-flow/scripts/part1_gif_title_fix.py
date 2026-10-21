"""Render-only fix of the Part 1 training-movie titles (2-3 readable lines).

Uses ONLY saved raw arrays: artifacts/run1/raw/training_movie_paths.npy
[10, 1001, 10, 2], training_movie_inputs.npz and loss_history.npz.
The prior GIF is preserved under artifacts/failed-attempts/.
"""
import shutil

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

shutil.copy("evidence/training_movie.gif",
            "artifacts/failed-attempts/training_movie-cropped-title.gif")

STAGES = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]
LIM = 12.0
paths = np.load("artifacts/run1/raw/training_movie_paths.npy")  # [10,1001,10,2]
hist = np.load("artifacts/run1/loss_history.npz")
hist_steps, hist_loss = np.asarray(hist["steps"]), np.asarray(hist["loss"])

fig, (ax, axl) = plt.subplots(2, 1, figsize=(6.4, 8.4),
                              height_ratios=[3, 1], constrained_layout=True)
lines = [ax.plot([], [], lw=0.8, alpha=0.8)[0] for _ in range(10)]
ends = ax.scatter([], [], s=40, marker="*", color="crimson", zorder=5)
starts = ax.scatter([], [], s=25, color="white", edgecolor="k", zorder=5)
title = ax.set_title("")
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
for c in [(3, 0), (0, 3), (-3, 0), (0, -3)]:
    ax.add_patch(plt.Circle(c, 0.6, fill=False, ls="--", color="gray"))
(loss_line,) = axl.plot([], [], lw=0.6)
vline = axl.axvline(0, color="crimson", ls=":")
axl.set_xlabel("training update"); axl.set_ylabel("loss")
axl.set_xlim(0, 25000); axl.set_ylim(0, 1.1)

def update(i):
    step = STAGES[i]
    p = paths[i]
    for j, ln in enumerate(lines):
        ln.set_data(p[:, j, 0], p[:, j, 1])
    ends.set_offsets(p[-1]); starts.set_offsets(p[0])
    out = bool((np.abs(p[1:]).max(axis=(1, 2)) > LIM).any())
    max_abs = float(np.abs(p).max())
    title.set_text(f"update {step} / 25000\n"
                   f"10 DDPM paths, shared starts and reverse noise\n"
                   f"unclipped: max|state| = {max_abs:.1f}, "
                   f"leaves view: {'yes' if out else 'no'}")
    mask = hist_steps <= step
    loss_line.set_data(hist_steps[mask], hist_loss[mask])
    vline.set_xdata([step, step])
    return lines + [ends, starts, loss_line, vline, title]

anim = animation.FuncAnimation(fig, update, frames=range(10), blit=False)
anim.save("evidence/training_movie.gif", writer=animation.PillowWriter(fps=2),
          dpi=120)
print("training_movie.gif re-rendered with readable titles")
