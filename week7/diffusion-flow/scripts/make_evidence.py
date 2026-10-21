"""Part 1 evidence driver (attempt 2): forward movie + stats, training movie,
reverse movie, final scatter and full cluster statistics.

All sampling goes through the production sampler src.sample.ddpm_sample with
network.predict passed directly. Raw streams/states saved under
artifacts/run1/raw. Statistics written to evidence/stats.json.
"""

import json
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from src import network
from src.checkpoints import load_params
from src.diffusion import abar_from_betas, beta_schedule, sample_target
from src.sample import ddpm_sample

EVID = "evidence"
RUN = "artifacts/run1"
RAW = f"{RUN}/raw"
CENTRES = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])
LIM = 12.0  # fixed plotting square

os.makedirs(EVID, exist_ok=True)
os.makedirs(RAW, exist_ok=True)

betas = beta_schedule()
abar = abar_from_betas(betas)


def gif(fig, update_fig, frames, path, fps=5, dpi=120):
    anim = animation.FuncAnimation(fig, update_fig, frames=frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


# ---------------- VERIFY 1: forward diffusion ----------------
N_FORWARD = 2000
root = jax.random.PRNGKey(100)
key_target, key_chain = jax.random.split(root)  # dedicated, independent keys
x0 = sample_target(key_target, N_FORWARD)
# follow the SAME points through the Markov chain, one step at a time
level_steps = [0, 50, 100, 200, 400, 600, 800, 1000]
states = [np.asarray(x0)]
x = x0
k = key_chain
for t in range(1, 1001):
    k, sub = jax.random.split(k)
    z = jax.random.normal(sub, x.shape, dtype=x.dtype)
    beta_t = betas[t - 1]
    x = jnp.sqrt(1.0 - beta_t) * x + jnp.sqrt(beta_t) * z
    if t in level_steps[1:]:
        states.append(np.asarray(x))
np.savez(f"{RAW}/forward_states.npz",
         levels=np.array(level_steps),
         **{f"level_{lv}": s for lv, s in zip(level_steps, states)})

m1000 = states[-1].mean(axis=0)
v1000 = states[-1].var(axis=0)
forward_stats = {
    "n_points": N_FORWARD,
    "levels": level_steps,
    "t1000_coord_means": m1000.tolist(),
    "t1000_coord_variances": v1000.tolist(),
    "threshold_mean_abs": 0.1,
    "threshold_var_within": 0.1,
    "pass_mean": bool(np.all(np.abs(m1000) < 0.1)),
    "pass_var": bool(np.all(np.abs(v1000 - 1.0) < 0.1)),
    "rng_convention": "PRNGKey(100) split into (target key, Markov-chain key)",
}

fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter([], [], s=4, alpha=0.5)
title = ax.set_title("")
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

def update_forward(i):
    sc.set_offsets(states[i])
    title.set_text(f"forward diffusion, level t = {level_steps[i]}")
    return sc, title

gif(fig, update_forward, range(len(level_steps)), f"{EVID}/forward_movie.gif")
fig2, ax2 = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
for a, s, lv in zip(ax2.ravel(), states, level_steps):
    a.scatter(s[:, 0], s[:, 1], s=3, alpha=0.4)
    a.set_title(f"t={lv}"); a.set_xlim(-LIM, LIM); a.set_ylim(-LIM, LIM)
    a.set_aspect("equal")
fig2.suptitle("Forward diffusion: the same 2000 points (fixed scale)")
fig2.tight_layout(); fig2.savefig(f"{EVID}/forward_panels.png", dpi=120)
plt.close(fig2)

# ---------------- VERIFY 2: training stages movie ----------------
STAGES = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]
N_PATHS = 10
key = jax.random.PRNGKey(200)
k_init, k_rev = jax.random.split(key)
xT = jax.random.normal(k_init, (N_PATHS, 2), dtype=jnp.float32)
z_stack = jax.random.normal(k_rev, (1000, N_PATHS, 2), dtype=jnp.float32)
reverse_noise = {t: z_stack[1000 - t] for t in range(1, 1001)}
np.savez(f"{RAW}/training_movie_inputs.npz",
         seed=np.int64(200), xT=np.asarray(xT), z_stack=np.asarray(z_stack),
         levels=np.arange(1000, 0, -1))

hist = np.load(f"{RUN}/loss_history.npz")
hist_steps = np.asarray(hist["steps"])
hist_loss = np.asarray(hist["loss"])

# paths[i] shape (1001, 10, 2): states from t=1000 down to t=0 for stage i
paths = np.zeros((len(STAGES), 1001, N_PATHS, 2), dtype=np.float32)
for i, step in enumerate(STAGES):
    params = load_params(f"{RUN}/checkpoints/params_{step:05d}.npz")
    x0s, rev_states = ddpm_sample(network.predict, params, xT, betas, abar,
                                  reverse_noise=reverse_noise,
                                  record_states=True)
    paths[i] = np.stack([np.asarray(s) for s in rev_states])
np.save(f"{RAW}/training_movie_paths.npy", paths)  # [10, 1001, 10, 2]

fig, (ax, axl) = plt.subplots(2, 1, figsize=(6, 8), height_ratios=[3, 1])
lines = [ax.plot([], [], lw=0.8, alpha=0.8)[0] for _ in range(N_PATHS)]
ends = ax.scatter([], [], s=40, marker="*", color="crimson", zorder=5)
starts = ax.scatter([], [], s=25, color="white", edgecolor="k", zorder=5)
title = ax.set_title("")
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
for c in CENTRES:
    ax.add_patch(plt.Circle(c, 0.6, fill=False, ls="--", color="gray"))
(loss_line,) = axl.plot([], [], lw=0.6)
vline = axl.axvline(0, color="crimson", ls=":")
axl.set_xlabel("training update"); axl.set_ylabel("loss")
axl.set_xlim(0, 25000); axl.set_ylim(0, 1.1)

def update_train(i):
    step = STAGES[i]
    p = paths[i]  # (1001, 10, 2)
    for j, ln in enumerate(lines):
        ln.set_data(p[:, j, 0], p[:, j, 1])
    ends.set_offsets(p[-1])
    starts.set_offsets(p[0])
    # honesty: trajectories (not the shared starts) leaving the square
    out_traj = float((np.abs(p[1:]).max(axis=(1, 2)) > LIM).any())
    max_abs = float(np.abs(p).max())
    title.set_text(
        f"update {step}: 10 DDPM paths (shared starts/noise) | "
        f"trajectories leave view: {'yes' if out_traj else 'no'} | "
        f"max|state| = {max_abs:.1f} (unclipped)")
    mask = hist_steps <= step
    loss_line.set_data(hist_steps[mask], hist_loss[mask])
    vline.set_xdata([step, step])
    return lines + [ends, starts, loss_line, vline, title]

gif(fig, update_train, range(len(STAGES)), f"{EVID}/training_movie.gif", fps=2)
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(hist_steps, hist_loss, lw=0.7)
for s in STAGES:
    ax3.axvline(s, color="crimson", ls=":", lw=0.8)
ax3.set_xlabel("training update"); ax3.set_ylabel("noise-prediction loss")
ax3.set_title("Loss history; dotted lines mark the 10 saved stages")
fig3.tight_layout(); fig3.savefig(f"{EVID}/loss_history.png", dpi=120)
plt.close(fig3)

# ---------------- VERIFY 3: 2000 fresh samples, reverse movie + stats --------
N_SAMPLES = 2000
key = jax.random.PRNGKey(300)
k_init, k_rev = jax.random.split(key)
xT2 = jax.random.normal(k_init, (N_SAMPLES, 2), dtype=jnp.float32)
rev = jax.random.normal(k_rev, (1000, N_SAMPLES, 2), dtype=jnp.float32)
reverse_noise2 = {t: rev[1000 - t] for t in range(1, 1001)}
final_params = load_params(f"{RUN}/checkpoints/params_25000.npz")
x0s, rev_states = ddpm_sample(network.predict, final_params, xT2, betas, abar,
                              reverse_noise=reverse_noise2, record_states=True)
rev_states = np.stack([np.asarray(s) for s in rev_states])  # (1001, N, 2)
np.save(f"{RAW}/reverse_states.npy", rev_states)
np.savez(f"{RAW}/reverse_inputs.npz", seed=np.int64(300),
         xT=np.asarray(xT2), z_stack=np.asarray(rev),
         levels=np.arange(1000, 0, -1))
X = rev_states[-1]
tgt = np.asarray(sample_target(jax.random.PRNGKey(400), N_SAMPLES))
np.savez(f"{RAW}/final_samples.npz", generated=X, target=tgt)

movie_levels = [1000, 800, 600, 400, 200, 100, 50, 20, 10, 5, 2, 0]
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter([], [], s=3, alpha=0.5)
title = ax.set_title("")
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")

def update_rev(i):
    lv = movie_levels[i]
    sc.set_offsets(rev_states[1000 - lv])
    title.set_text(f"DDPM reverse, level t = {lv}")
    return sc, title

gif(fig, update_rev, range(len(movie_levels)), f"{EVID}/reverse_movie.gif")
fig4, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
axes[0].scatter(X[:, 0], X[:, 1], s=3, alpha=0.4)
axes[0].set_title(f"DDPM generated ({N_SAMPLES} samples, 1000 steps)")
axes[1].scatter(tgt[:, 0], tgt[:, 1], s=3, alpha=0.4)
axes[1].set_title("Exact target samples")
for a in axes:
    a.set_xlim(-LIM, LIM); a.set_ylim(-LIM, LIM); a.set_aspect("equal")
    for c in CENTRES:
        a.add_patch(plt.Circle(c, 0.6, fill=False, ls="--", color="gray"))
fig4.tight_layout(); fig4.savefig(f"{EVID}/final_scatter.png", dpi=120)
plt.close(fig4)

# cluster statistics on the generated samples (every statistic explicit)
d = ((X[:, None, :] - CENTRES[None]) ** 2).sum(-1)
assign = d.argmin(1)
cluster_stats = []
for k_i, c in enumerate(CENTRES):
    pts = X[assign == k_i]
    cm = pts.mean(axis=0)
    sd = pts.std(axis=0)
    cluster_stats.append({
        "centre_target": c.tolist(),
        "centre_measured": cm.tolist(),
        "centre_error_abs": np.abs(cm - c).tolist(),
        "fraction": float(len(pts) / len(X)),
        "std_per_coordinate": sd.tolist(),
        "std_error_abs": np.abs(sd - 0.30).tolist(),
    })
reverse_stats = {
    "n_samples": N_SAMPLES, "steps": 1000,
    "clusters": cluster_stats,
    "threshold_fraction": 0.04, "threshold_centre": 0.05,
    "threshold_std": 0.03,
    "pass_fractions": all(abs(c["fraction"] - 0.25) < 0.04
                          for c in cluster_stats),
    "pass_centres": all(np.array(c["centre_error_abs"]).max() < 0.05
                        for c in cluster_stats),
    "pass_stds": all(np.array(c["std_error_abs"]).max() < 0.03
                     for c in cluster_stats),
}

summary = {
    "forward": forward_stats,
    "reverse": reverse_stats,
    "training": {"stages": STAGES,
                 "final_loss": float(hist_loss[-1]),
                 "loss_records": int(len(hist_loss)),
                 "loss_last1000_mean": float(hist_loss[-1000:].mean())},
    "seeds": {"forward": 100, "training_movie": 200, "reverse": 300,
              "target_reference": 400, "training": 2026},
}
with open(f"{EVID}/stats.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
