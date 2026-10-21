"""Recompute Part 2 trajectory figure (render-only fix; same seeds/keys).

DDPM-B first-10 trajectories are replayed from the ORIGINAL key PRNGKey(2027)
split with the FULL 2000-shaped noise stream, then sliced [:, :10] (full-shape
draws are shape-dependent; never redraw (10,2) innovations). DDIM run 2 is
overlaid dashed. Prior figure preserved under artifacts/failed-attempts/.
"""
import shutil

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import network
from src.checkpoints import load_params
from src.ddim import ddim_sample
from src.diffusion import abar_from_betas, beta_schedule
from src.sample import ddpm_sample

shutil.copy("evidence/part2_trajectories.png",
            "artifacts/failed-attempts/part2_trajectories-blank-panel3.png")

N = 2000
LIM = 5.0
betas = beta_schedule()
abar = abar_from_betas(betas)
params = load_params("artifacts/run1/checkpoints/params_25000.npz")
levels_full = np.arange(1000, -1, -1)

root = jax.random.PRNGKey(2027)          # unchanged predeclared seed
k_xT, k_ddpmA, k_ddpmB, _ = jax.random.split(root, 4)
xT = jax.random.normal(k_xT, (N, 2), dtype=jnp.float32)

ddim_a_states = ddim_sample(network.predict, params, xT, levels_full, abar,
                            record_states=True)[2]
ddim_b_states = ddim_sample(network.predict, params, xT, levels_full, abar,
                            record_states=True)[2]
ddpm_a_states = ddpm_sample(network.predict, params, xT, betas, abar,
                            key=k_ddpmA, record_states=True)[1]
ddpm_b_states = ddpm_sample(network.predict, params, xT, betas, abar,
                            key=k_ddpmB, record_states=True)[1]

def ddim_path(states):  # DDIM scan returns 1000 post-jump states; add xT
    return np.concatenate([np.asarray(xT)[None],
                           np.stack([np.asarray(s) for s in states])], axis=0)

def ddpm_path(states):  # Part 1 DDPM record_states already includes xT
    return np.stack([np.asarray(s) for s in states])

paths = {k: v[:, :10] for k, v in {
    "DDIM run 1": ddim_path(ddim_a_states),
    "DDIM run 2 (dashed overlay)": ddim_path(ddim_b_states),
    "DDPM noise stream A": ddpm_path(ddpm_a_states),
    "DDPM noise stream B": ddpm_path(ddpm_b_states)}.items()}
assert paths["DDIM run 1"].shape == (1001, 10, 2)
assert paths["DDPM noise stream A"].shape == (1001, 10, 2)
np.savez("artifacts/part2/raw/part2_paths_first10_fixed.npz",
         levels=np.arange(1000, -1, -1),
         **{f"{k}_first10": v[:, :10] for k, v in paths.items()})

show = np.arange(0, 1001, 10)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
panels = [("DDIM run 1 (deterministic; run 2 dashed, identical)",
           paths["DDIM run 1"], paths["DDIM run 2 (dashed overlay)"]),
          ("DDPM noise stream A", paths["DDPM noise stream A"], None),
          ("DDPM noise stream B", paths["DDPM noise stream B"], None)]
for ax, (name, path, overlay) in zip(axes, panels):
    for j in range(path.shape[1]):
        ax.plot(path[show, j, 0], path[show, j, 1], lw=0.7, alpha=0.85)
    if overlay is not None:
        for j in range(10):
            ax.plot(overlay[show, j, 0], overlay[show, j, 1], lw=0.7,
                    ls="--", color="gray", alpha=0.7)
    ax.scatter(path[0][:, 0], path[0][:, 1], s=30, color="white",
               edgecolor="k", zorder=5, label="shared starts (t=1000)")
    ax.scatter(path[-1][:, 0], path[-1][:, 1], s=45, marker="*",
               color="crimson", zorder=6, label="endpoints (t=0)")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_title(f"{name}\nunclipped; levels outside [-5,5] not shown")
    ax.set_xlabel("$x_1$"); ax.legend(fontsize=7)
axes[0].set_ylabel("$x_2$")
fig.suptitle("10 shared starts, seed 2027: deterministic DDIM (runs 1/2 "
             "identical) vs stochastic DDPM streams A/B")
fig.tight_layout()
fig.savefig("evidence/part2_trajectories.png", dpi=120)
plt.close(fig)
print("trajectories figure recomputed; DDIM run2 maxdiff:",
      float(np.abs(paths["DDIM run 1"] - paths["DDIM run 2 (dashed overlay)"]).max()))
