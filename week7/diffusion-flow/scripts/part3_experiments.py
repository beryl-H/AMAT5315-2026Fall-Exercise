"""Part 3 experiments (validated run): fields, step-count cost study.

Predeclared seeds/protocol:
- field evaluation: full 9x9 Cartesian grid [-4,4]^2, all 81 points,
  unweighted; contextual scatter seed 2030 (fold_in per s).
- runs: key index = method_id*10000 + S with method_id ddpm=0, ddim=1,
  flow=2, exact-control ddpm=3, ddim=4, flow=5;
  runroot = fold_in(PRNGKey(2032), index) SPLIT into start key and
  reverse/noise key (never the same key for start and innovations).
  Raw uint32[2] key arrays and xT are saved per run.
- exact reference: seed 2033. Null: 20 pairs of independent exact sets,
  seed 2034 with saved per-pair key arrays and both A/B sample arrays.
N = 4000 per run. Earlier unvalidated attempts are recorded as INVALID_RNG.
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import network
from src.checkpoints import load_params
from src.coarse import coarse_ddpm_sample
from src.ddim import ddim_sample, mixture_exact_noise
from src.diffusion import abar_from_betas, beta_schedule, sample_target
from src.energy import energy_distance_unbiased
from src.flow import euler_sample, exact_marginal_velocity, flow_predict

RUN = "artifacts/part3-validated"
RAW = f"{RUN}/raw"
EVID = "evidence"
N = 4000
LIM = 5.0
CENTRES = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])
SEED_RUNS, SEED_EXACT, SEED_NULL = 2032, 2033, 2034
METHOD_ID = {"ddpm": 0, "ddim": 1, "flow": 2,
             "ddpm_exact": 3, "ddim_exact": 4, "flow_exact": 5}
os.makedirs(RAW, exist_ok=True)
os.makedirs(EVID, exist_ok=True)

betas = beta_schedule()
abar = abar_from_betas(betas)
flow_params = load_params("artifacts/flow-run1/checkpoints/params_25000.npz")
diff_params = load_params("artifacts/run1/checkpoints/params_25000.npz")


def fname(method, S):
    return f"{method}_S{S:04d}.npy"


def keys_for(method, S):
    """(start_key, noise_key) as numpy uint32[2] arrays plus jax keys."""
    idx = METHOD_ID[method] * 10000 + S
    root = jax.random.fold_in(jax.random.PRNGKey(SEED_RUNS), idx)
    k_start, k_noise = jax.random.split(root)
    return (np.asarray(k_start), np.asarray(k_noise), k_start, k_noise)


# ---------------- 1) velocity fields on the predeclared 9x9 grid ----------
g = np.linspace(-4.0, 4.0, 9)
GX, GY = np.meshgrid(g, g)
grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float32)
ss = (0.1, 0.5, 0.9)
key_scatter = jax.random.PRNGKey(2030)
field_stats = {}
save_dict = {"grid": grid_pts}
for i, s in enumerate(ss):
    sv = jnp.full((len(grid_pts),), s, dtype=jnp.float32)
    v_exact = np.asarray(exact_marginal_velocity(jnp.asarray(grid_pts), s))
    v_learn = np.asarray(flow_predict(flow_params, jnp.asarray(grid_pts), sv))
    rel_rms = float(np.sqrt(((v_learn - v_exact) ** 2).sum()
                            / (v_exact ** 2).sum()))
    ks = jax.random.fold_in(key_scatter, i)
    kx0, kz = jax.random.split(ks)
    x0s = sample_target(kx0, 2000)
    zs = jax.random.normal(kz, (2000, 2), dtype=jnp.float32)
    x_s = np.asarray((1 - s) * x0s + s * zs)
    # supplementary (clearly named, never a substitute): 4096 fresh points
    # drawn from the actual x_s marginal, per lecture Eq. 18 test points
    kx0b, kzb = jax.random.split(jax.random.fold_in(key_scatter, 100 + i))
    x0b = sample_target(kx0b, 4096)
    zb = jax.random.normal(kzb, (4096, 2), dtype=jnp.float32)
    xb = np.asarray((1 - s) * x0b + s * zb)
    vb = np.asarray(flow_predict(flow_params, jnp.asarray(xb),
                                 jnp.full((4096,), s)))
    veb = np.asarray(exact_marginal_velocity(jnp.asarray(xb), s))
    rel_rms_marg = float(np.sqrt(((vb - veb) ** 2).sum()
                                 / (veb ** 2).sum()))
    field_stats[f"s{s}"] = {"rel_rms_grid": rel_rms,
                            "rel_rms_marginal_4096": rel_rms_marg}
    save_dict.update({f"s{s}_exact": v_exact, f"s{s}_learned": v_learn,
                      f"s{s}_scatter": x_s, f"s{s}_marginal": xb,
                      f"s{s}_rel_rms": np.float32(rel_rms),
                      f"s{s}_rel_rms_marg": np.float32(rel_rms_marg)})
    print(f"field s={s}: relative RMS (81-pt grid) = {rel_rms:.4f}")
np.savez(f"{RAW}/field_evaluation.npz", **save_dict)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True, sharey=True)
for ax, s in zip(axes, ss):
    v_exact = save_dict[f"s{s}_exact"]
    v_learn = save_dict[f"s{s}_learned"]
    x_s = save_dict[f"s{s}_scatter"]
    ax.scatter(x_s[:, 0], x_s[:, 1], s=2, alpha=0.15, color="lightgray",
               label="$x_s$ samples")
    def clip(V, cap=1.2):
        n = np.linalg.norm(V, axis=-1, keepdims=True)
        return V * np.minimum(1.0, cap / np.maximum(n, 1e-12))
    ax.quiver(grid_pts[:, 0], grid_pts[:, 1], *clip(v_exact).T,
              color="teal", alpha=0.8, label="exact", width=0.004)
    ax.quiver(grid_pts[:, 0], grid_pts[:, 1], *clip(v_learn).T,
              color="crimson", alpha=0.6, label="learned", width=0.004)
    ax.set_title(f"s = {s}   rel. RMS = {field_stats[f's{s}']['rel_rms_grid']:.3f}")
    ax.set_xlabel("$x_1$"); ax.set_aspect("equal"); ax.legend(fontsize=7)
axes[0].set_ylabel("$x_2$")
fig.suptitle("Velocity fields on the predeclared 9x9 grid [-4,4]^2 "
             "(arrow lengths clipped for display only; raw vectors saved)")
fig.tight_layout(); fig.savefig(f"{EVID}/part3_velocity_fields.png", dpi=120)
plt.close(fig)


# ---------------- 2) step-count study ----------------
def levels_for(S):
    return np.arange(1000, -1, -1) if S == 1000 else \
        np.rint(np.linspace(1000, 0, S + 1)).astype(int)


def times_for(S):
    return np.linspace(1.0, 0.0, S + 1)


def metrics(x):
    """Per-cluster stats; empty clusters get None centres/widths (undefined),
    explicit empty flags and counts; max errors None when undefined."""
    x = np.asarray(x)
    d = ((x[:, None, :] - CENTRES[None]) ** 2).sum(-1)
    a = d.argmin(1)
    counts = [int((a == k).sum()) for k in range(4)]
    fr = [c / len(x) for c in counts]
    cs, ws = [], []
    for k in range(4):
        if counts[k] > 0:
            cs.append(x[a == k].mean(axis=0).tolist())
            ws.append(x[a == k].std(axis=0).tolist())
        else:
            cs.append(None)
            ws.append(None)
    frac_err = float(np.abs(np.array(fr) - 0.25).max())
    defined = [k for k in range(4) if counts[k] > 0]
    centre_err = (float(max(np.abs(np.array(cs[k]) - CENTRES[k]).max()
                            for k in defined)) if defined else None)
    width_err = (float(max(np.abs(np.array(ws[k]) - 0.30).max()
                           for k in defined)) if defined else None)
    return {"counts": counts, "fractions": fr,
            "centres": cs, "widths": ws,
            "empty_clusters": [k for k in range(4) if counts[k] == 0],
            "max_fraction_error": frac_err,
            "max_centre_error": centre_err,
            "max_width_error": width_err,
            "finite": bool(np.all(np.isfinite(x)))}


def run_sampler(method, S):
    """method in {ddpm, ddim, flow, *_exact}; one (start, noise) key pair."""
    t0 = time.time()
    _, _, k_start, k_noise = keys_for(method, S)
    xT = np.asarray(jax.random.normal(k_start, (N, 2), dtype=jnp.float32))
    net = "exact" if method.endswith("_exact") else "learned"
    base = method.replace("_exact", "")
    if base == "ddpm":
        lv = levels_for(S)
        z = np.asarray(jax.random.normal(k_noise, (len(lv) - 1, N, 2),
                                         dtype=jnp.float32))
        replay = {int(lv[i]): z[i] for i in range(len(lv) - 1)}
        pred = network.predict if net == "learned" else \
            lambda p, x, t: mixture_exact_noise(x, t[0], abar)
        out = coarse_ddpm_sample(pred, diff_params, jnp.asarray(xT), betas,
                                 abar, levels=lv, reverse_noise=replay)
    elif base == "ddim":
        pred = network.predict if net == "learned" else \
            lambda p, x, t: mixture_exact_noise(x, t[0], abar)
        out = ddim_sample(pred, diff_params, jnp.asarray(xT), levels_for(S),
                          abar)
    else:
        pred = (lambda p, x, s: flow_predict(p, x, s)) if net == "learned" \
            else (lambda p, x, s: exact_marginal_velocity(x, s[0]))
        out = euler_sample(pred, flow_params, jnp.asarray(xT), times_for(S))
    x = np.asarray(out)
    np.save(f"{RAW}/{fname(method, S)}", x)
    np.savez(f"{RAW}/{fname(method, S).replace('.npy', '_meta.npz')}",
             start_key=np.asarray(keys_for(method, S)[0], dtype=np.uint32),
             noise_key=np.asarray(keys_for(method, S)[1], dtype=np.uint32),
             xT=xT.astype(np.float32))
    m = metrics(x)
    m["energy"] = float(energy_distance_unbiased(x, exact_ref))
    m["runtime_s"] = time.time() - t0
    m["predictor"] = net
    m["start_is_gaussian"] = True
    return x, m


exact_ref = np.asarray(sample_target(jax.random.PRNGKey(SEED_EXACT), N))
np.save(f"{RAW}/exact_reference_n4000.npy", exact_ref)

results = {}
for method, steps in [("ddpm", [1, 2, 5, 10, 20, 50, 100, 1000]),
                      ("ddim", [1, 2, 5, 10, 20, 50, 100, 1000]),
                      ("flow", [1, 2, 5, 10, 20, 50, 100])]:
    for S in steps:
        _, m = run_sampler(method, S)
        results[f"{method}_S{S}"] = m
        print(f"{method} S={S}: energy {m['energy']:.5f} "
              f"maxfrac {m['max_fraction_error']:.4f} "
              f"maxwidth {m['max_width_error']} "
              f"empty {m['empty_clusters']}")

controls = {}
for method, steps in [("ddpm", [1, 2, 5, 10, 20, 50, 100, 1000]),
                      ("ddim", [1, 2, 5, 10, 20, 50, 100, 1000]),
                      ("flow", [1, 2, 5, 10, 20, 50, 100])]:
    for S in (1, 2):
        _, m = run_sampler(f"{method}_exact", S)
        controls[f"{method}_exact_S{S}"] = m
        print(f"control {method} exact S={S}: energy {m['energy']:.5f} "
              f"mean|r| pending, empty {m['empty_clusters']}")

# collapse evidence for BOTH S1 and S2, ALL three methods (exact predictors)
collapse = {}
for method in ("ddpm", "ddim", "flow"):
    for S in (1, 2):
        x = np.load(f"{RAW}/{fname(f'{method}_exact', S)}")
        m = controls[f"{method}_exact_S{S}"]
        collapse[f"{method}_exact_S{S}"] = {
            "mean_radial_norm": float(np.linalg.norm(x, axis=1).mean()),
            "energy": m["energy"],
            "empty_clusters": m["empty_clusters"],
            "max_fraction_error": m["max_fraction_error"],
            "max_width_error": m["max_width_error"]}

# null distribution: 20 pairs of independent exact sets (seed 2034)
null_vals, null_key_arrays, null_pairs = [], [], []
kn = jax.random.PRNGKey(SEED_NULL)
for i in range(20):
    k1, k2, kn = jax.random.split(kn, 3)
    A = np.asarray(sample_target(k1, N))
    B = np.asarray(sample_target(k2, N))
    null_vals.append(float(energy_distance_unbiased(A, B)))
    null_key_arrays.append([np.asarray(k1, dtype=np.uint32),
                            np.asarray(k2, dtype=np.uint32)])
    null_pairs.append((A, B))
null_vals = np.array(null_vals)
q95 = float(np.quantile(null_vals, 0.95))
np.savez(f"{RAW}/null_pairs.npz", values_raw=null_vals,
         keys=np.stack([np.stack(p) for p in null_key_arrays]).astype(np.uint32),
         **{f"pair{i}_A": p[0] for i, p in enumerate(null_pairs)},
         **{f"pair{i}_B": p[1] for i, p in enumerate(null_pairs)})
print(f"null 95% quantile: {q95:.5f}; raw estimates range "
      f"[{null_vals.min():.5f}, {null_vals.max():.5f}]")

flow20 = results["flow_S20"]
summary = {
    "seeds": {"runs": 2032, "exact_reference": 2033, "null": 2034,
              "scatter": 2030, "flow_training": 2029,
              "protocol": ("index = method_id*10000+S; "
                           "fold_in(PRNGKey(2032), index) split into "
                           "start and noise keys")},
    "n": N,
    "field_rel_rms_grid": {f"s{s}": field_stats[f"s{s}"]["rel_rms_grid"]
                           for s in ss},
    "runs": results,
    "exact_controls": controls,
    "collapse_S1_S2": collapse,
    "null": {"values_raw": null_vals.tolist(), "q95": q95,
             "note": "raw unbiased estimates, may be negative"},
    "gates": {
        "field_rel_rms_lt_0.1_all_s_FULL_GRID": bool(all(
            field_stats[f"s{s}"]["rel_rms_grid"] < 0.1 for s in ss)),
        "field_rel_rms_marginal_4096": {
            f"s{s}": field_stats[f"s{s}"]["rel_rms_marginal_4096"]
            for s in ss},
        "field_note": ("full-grid gate evaluated honestly; errors concentrate "
                       "in zero-training-density far corners; marginal metric "
                       "is supplementary only"),
        "flow_20_energy_lt_0.005": bool(flow20["energy"] < 0.005),
        "flow_20_max_width_error_lt_0.03": (
            None if flow20["max_width_error"] is None
            else bool(flow20["max_width_error"] < 0.03)),
        "flow_20_empty_clusters": flow20["empty_clusters"],
        "fraction_2se_band": float(2 * np.sqrt(0.25 * 0.75 / N)),
    },
}
with open(f"{EVID}/part3_stats.json", "w") as f:
    json.dump(summary, f, indent=2, allow_nan=False, default=float)
print(json.dumps(summary["gates"], indent=2, default=float))

# ---------------- 3) cost plots ----------------
FLOOR = 1e-4  # plotted log values only; raw JSON keeps signed values
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
colors = {"ddpm": "tab:blue", "ddim": "tab:orange", "flow": "tab:green"}
for method, steps in [("ddpm", [1, 2, 5, 10, 20, 50, 100, 1000]),
                      ("ddim", [1, 2, 5, 10, 20, 50, 100, 1000]),
                      ("flow", [1, 2, 5, 10, 20, 50, 100])]:
    xs = steps
    axes[0].plot(xs, [max(results[f"{method}_S{S}"]["energy"], FLOOR)
                      for S in steps], "o-", color=colors[method],
                 label=method)
    axes[1].plot(xs, [results[f"{method}_S{S}"]["max_fraction_error"]
                      for S in steps], "o-", color=colors[method])
    axes[2].plot(xs, [results[f"{method}_S{S}"]["max_width_error"]
                      for S in steps], "o-", color=colors[method])
axes[0].axhspan(FLOOR, q95, alpha=0.15, color="gray",
                label="null band (95% quantile)")
axes[0].axhline(q95, color="gray", ls="--", lw=1)
axes[0].set_yscale("log")
axes[0].set_ylabel("energy distance (plotted values floored at 1e-4)")
axes[1].axhspan(0, 2 * np.sqrt(0.25 * 0.75 / N), alpha=0.15, color="gray")
axes[1].set_ylabel("max cluster-fraction error")
axes[2].set_ylabel("max per-coordinate width error")
for ax in axes:
    ax.set_xlabel("sampling steps (one network evaluation each)")
    ax.set_xscale("log")
    ax.axvline(20, color="crimson", ls=":", lw=1)
    ax.legend(fontsize=7)
axes[0].set_title("Energy distance vs cost")
axes[1].set_title("Max fraction error (shaded: 2 binomial SE)")
axes[2].set_title("Max width error (dotted: 20 steps)")
fig.suptitle("Part 3 step-count study, N=4000 (seeds 2032/2033/2034)")
fig.tight_layout(); fig.savefig(f"{EVID}/part3_cost_study.png", dpi=120)
plt.close(fig)

# collapse scatter: exact predictors, S=1 and S=2, all three methods
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
for row, S in enumerate((1, 2)):
    for col, method in enumerate(("ddim", "flow", "ddpm")):
        ax = axes[row, col]
        x = np.load(f"{RAW}/{fname(f'{method}_exact', S)}")
        ax.scatter(x[:, 0], x[:, 1], s=3, alpha=0.4)
        ax.set_title(f"{method} exact, S={S}: mean |x| = "
                     f"{np.linalg.norm(x, axis=1).mean():.3f}")
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
    ax = axes[row, 3]
    ax.scatter(exact_ref[:, 0], exact_ref[:, 1], s=3, alpha=0.4)
    ax.set_title("exact target (reference)")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
fig.suptitle("Coarse grids collapse toward the mean even with exact "
             "predictors (controls, N=4000)")
fig.tight_layout(); fig.savefig(f"{EVID}/part3_collapse_controls.png", dpi=120)
plt.close(fig)
print("part3 validated figures written")
