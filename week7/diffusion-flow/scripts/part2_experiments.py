"""Part 2 experiments: DDIM determinism, DDPM stochasticity, exact predictor.

Predeclared seeds (pi-part2.md): root PRNGKey(2027) split into
(shared xT, DDPM noise A, DDPM noise B, independent target reference).
N = 2000, all 1000 levels. No seed search, no tuning.
Raw arrays -> artifacts/part2/*.npz; figures + summary -> evidence/.
"""

import json
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import network
from src.checkpoints import load_params
from src.ddim import ddim_sample, exact_predictor
from src.diffusion import abar_from_betas, beta_schedule, sample_target
from src.sample import ddpm_sample

RUN = "artifacts/part2"
EVID = "evidence"
RAW_OUT = f"{RUN}/raw"
N = 2000
LIM = 5.0
CENTRES = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])

os.makedirs(RAW_OUT, exist_ok=True)
os.makedirs(EVID, exist_ok=True)

betas = beta_schedule()
abar = abar_from_betas(betas)
params = load_params("artifacts/run1/checkpoints/params_25000.npz")
levels_full = np.arange(1000, -1, -1)

# predeclared seed split
root = jax.random.PRNGKey(2027)
k_xT, k_ddpmA, k_ddpmB, k_target = jax.random.split(root, 4)
xT = jax.random.normal(k_xT, (N, 2), dtype=jnp.float32)

# 1) learned DDIM, twice from exactly the same xT (all 1000 levels)
ddim_end_a, lv, ddim_states_a = ddim_sample(
    network.predict, params, xT, levels_full, abar, record_states=True)
ddim_end_b, _, ddim_states_b = ddim_sample(
    network.predict, params, xT, levels_full, abar, record_states=True)
maxdiff = float(jnp.max(jnp.abs(ddim_end_a - ddim_end_b)))
maxdiff_states = float(jnp.max(jnp.abs(
    jnp.stack(list(ddim_states_a)) - jnp.stack(list(ddim_states_b)))))
print(f"DDIM repeat max |endpoint diff| = {maxdiff}")
print(f"DDIM repeat max |all-state diff| = {maxdiff_states}")

# 2) DDPM from the same xT with two independent reverse-noise streams
ddpm_a, ddpm_a_states = ddpm_sample(network.predict, params, xT, betas, abar,
                                    key=k_ddpmA, record_states=True)
ddpm_b, ddpm_b_states = ddpm_sample(network.predict, params, xT, betas,
                                    abar, key=k_ddpmB, record_states=True)

# 3) exact-predictor DDIM, same starting xT
exact_pred = exact_predictor(abar)
exact_end = ddim_sample(exact_pred, None, xT, levels_full, abar)

# 4) independent exact target reference
target = sample_target(k_target, N)

np.savez(f"{RAW_OUT}/part2_runs.npz",
         seed=np.int64(2027), n=np.int64(N),
         xT=np.asarray(xT),
         ddim_a=np.asarray(ddim_end_a), ddim_b=np.asarray(ddim_end_b),
         ddpm_a=np.asarray(ddpm_a), ddpm_b=np.asarray(ddpm_b),
         exact_ddim=np.asarray(exact_end), target=np.asarray(target),
         ddim_repeat_maxdiff=np.float32(maxdiff),
         ddim_repeat_maxdiff_states=np.float32(maxdiff_states))
# first 10 full trajectories (shared starts) for audit
np.savez(f"{RAW_OUT}/part2_paths_first10.npz",
         levels=levels_full,
         ddim_a=np.concatenate(
             [np.asarray(xT)[None, :10],
              np.stack([np.asarray(s) for s in ddim_states_a])[:, :10]],
             axis=0),
         ddpm_b=np.stack([np.asarray(s) for s in ddpm_b_states])[:, :10],
         ddpm_a=np.stack([np.asarray(s) for s in ddpm_a_states])[:, :10])


def stats(x):
    d = ((x[:, None, :] - CENTRES[None]) ** 2).sum(-1)
    assign = d.argmin(1)
    out = []
    for k in range(4):
        pts = x[assign == k]
        out.append({"centre": pts.mean(axis=0).tolist(),
                    "fraction": float(len(pts) / len(x)),
                    "std": pts.std(axis=0).tolist()})
    return out, assign


runs = {"learned_ddim": np.asarray(ddim_end_a),
        "ddpm_a": np.asarray(ddpm_a), "ddpm_b": np.asarray(ddpm_b),
        "exact_ddim": np.asarray(exact_end), "target": np.asarray(target)}
all_stats = {}
assigns = {}
for name, x in runs.items():
    st, assign = stats(x)
    all_stats[name] = st
    assigns[name] = assign

# DDPM A/B same-cluster agreement (descriptive, reference ~0.25)
same_cluster = float((assigns["ddpm_a"] == assigns["ddpm_b"]).mean())

summary = {
    "seeds": {"root": 2027, "xT": 0, "ddpm_a": 1, "ddpm_b": 2, "target": 3},
    "n": N, "levels": "all 1000 levels",
    "ddim_repeat_max_endpoint_diff": maxdiff,
    "ddim_repeat_max_all_state_diff": maxdiff_states,
    "ddpm_ab_same_cluster_fraction": same_cluster,
    "cluster_stats": all_stats,
    "gates": {
        "ddim_fraction_abs_error_lt": 0.015,
        "learned_widths_in": [0.29, 0.32],
        "ddim_fraction_errors": {
            name: max(abs(all_stats[name][k]["fraction"] - 0.25)
                      for k in range(4))
            for name in ("learned_ddim", "exact_ddim")},
        "learned_width_ranges": {
            name: [float(min(min(c["std"]) for c in all_stats[name])),
                   float(max(max(c["std"]) for c in all_stats[name]))]
            for name in ("learned_ddim", "ddpm_a", "ddpm_b")},
    },
}
summary["gates"]["pass_ddim_fractions"] = all(
    v < 0.015 for v in summary["gates"]["ddim_fraction_errors"].values())
summary["gates"]["pass_learned_widths"] = {
    name: bool(0.29 <= r[0] and r[1] <= 0.32)
    for name, r in summary["gates"]["learned_width_ranges"].items()}
with open(f"{EVID}/part2_stats.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary["gates"], indent=2))
print(f"DDPM A/B same-cluster fraction: {same_cluster:.4f}")

# ---------------- figures ----------------
# paired trajectories: 10 shared starts through DDIM(A), DDPM(A), DDPM(B)
sel = slice(0, 10)
ddim_path10 = np.concatenate(
    [xT[None], np.stack([np.asarray(s) for s in ddim_states_a])], axis=0)[:, sel]
ddpmA_path10 = np.stack([np.asarray(s) for s in ddpm_a_states])[:, sel]
show = np.arange(0, 1001, 10)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, path, name in zip(
        axes, [ddim_path10, ddpmA_path10,
               np.stack([np.asarray(s) for s in ddpm_b_states])[:, :10]],
        ["DDIM run 1 (deterministic)", "DDPM noise stream A",
         "DDPM noise stream B"]):
    for j in range(path.shape[1]):
        ax.plot(path[show, j, 0], path[show, j, 1], lw=0.7, alpha=0.8)
    ax.scatter(path[0][:, 0], path[0][:, 1], s=25, color="white",
               edgecolor="k", zorder=5, label="shared starts")
    ax.scatter(path[-1][:, 0], path[-1][:, 1], s=40, marker="*",
               color="crimson", zorder=5, label="endpoints")
    ax.set_title(name); ax.set_xlabel("$x_1$")
    ax.set_xlim(-LIM * 2.5, LIM * 2.5); ax.set_ylim(-LIM * 2.5, LIM * 2.5)
    ax.set_aspect("equal"); ax.legend(fontsize=7)
axes[0].set_ylabel("$x_2$")
fig.suptitle("10 shared starts: deterministic DDIM vs stochastic DDPM "
             "(unclipped trajectories; start = original level t=1000, "
             "end = t=0; seed 2027)")
fig.tight_layout(); fig.savefig(f"{EVID}/part2_trajectories.png", dpi=120)
plt.close(fig)

# DDPM B endpoints on the same axes as A for pairing
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(np.asarray(ddpm_a)[:, 0], np.asarray(ddpm_a)[:, 1], s=3, alpha=0.4,
           label="DDPM A")
ax.scatter(np.asarray(ddpm_b)[:, 0], np.asarray(ddpm_b)[:, 1], s=3, alpha=0.4,
           label="DDPM B")
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
ax.legend(); ax.set_title("DDPM A/B endpoints, same 2000 starts (seed 2027)")
fig.tight_layout(); fig.savefig(f"{EVID}/part2_ddpm_ab.png", dpi=120)
plt.close(fig)

# final distributions on fixed square [-5, 5]
panels = [("learned DDIM (1000 levels)", ddim_end_a),
          ("exact-predictor DDIM", exact_end),
          ("DDPM stream A", ddpm_a), ("exact target samples", target)]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
for ax, (name, x) in zip(axes, panels):
    ax.scatter(x[:, 0], x[:, 1], s=3, alpha=0.4)
    ax.set_title(name); ax.set_xlabel("$x_1$")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    for c in CENTRES:
        ax.add_patch(plt.Circle(c, 0.6, fill=False, ls="--", color="gray"))
axes[0].set_ylabel("$x_2$")
fig.suptitle("2000 samples per distribution ([-5,5] square; seed 2027)")
fig.tight_layout(); fig.savefig(f"{EVID}/part2_final_distributions.png",
                                dpi=120)
plt.close(fig)
print("figures written")
