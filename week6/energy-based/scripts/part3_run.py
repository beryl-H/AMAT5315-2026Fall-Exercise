"""Part 3 study driver: Hopfield capacity sweep (N, alpha) x 20 sets.

Key derivation (documented, fixed before outcomes):
  master = PRNGKey(2026)
  key_N        = fold_in(master, N)
  key_set      = fold_in(key_N, set_index)          # patterns of that set
  key_recall   = fold_in(key_set, 7777)             # update-order stream
Per-sweep permutations are drawn inside recall from key_recall (fresh split
per sweep), so every chain and sweep has an independent random order.

Raw per-set arrays are saved under artifacts/part3/evaluation-<timestamp>/
(git-ignored); compact aggregate metrics go to evidence/part3_metrics.json.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ebm.hopfield import (hebb_k, overlap, random_patterns, recall,
                          unstable_fraction)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts/part3"
EV = ROOT / "evidence"
NS = [100, 200, 400, 800]
ALPHAS = [round(a, 2) for a in np.arange(0.05, 0.2501, 0.01)]
N_SETS = 20
MAX_SWEEPS = 200
MAX_P = int(round(0.25 * max(NS)))  # 200: padding bound
SEED = 2026


def keys_for(N, set_idx):
    key_N = jax.random.fold_in(jax.random.PRNGKey(SEED), N)
    key_set = jax.random.fold_in(key_N, set_idx)
    key_recall = jax.random.fold_in(key_set, 7777)
    return key_set, key_recall


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ART / f"evaluation-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    print(f"keys: master=PRNGKey({SEED}), key_N=fold_in(master,N), "
          f"key_set=fold_in(key_N,set), key_recall=fold_in(key_set,7777)",
          flush=True)

    results = []
    unstable_all = {}
    recall_all = {}
    sweeps_all = {}
    incomplete = []
    for N in NS:
        tN = time.perf_counter()
        for alpha in ALPHAS:
            P = int(round(alpha * N))
            # ---- patterns for all 20 sets, (S, P, N)
            Xis = np.stack([
                np.asarray(random_patterns(keys_for(N, s)[0], P=P, N=N))
                for s in range(N_SETS)])
            # ---- unstable fraction (batched over sets), no padding needed
            f_vmap = jax.jit(jax.vmap(unstable_fraction, in_axes=(0, 0)))
            t0 = time.perf_counter()
            frac = np.asarray(f_vmap(jnp.stack([hebb_k(jnp.asarray(x)) for x in Xis]),
                                     jnp.asarray(Xis)))
            t_unstable = time.perf_counter() - t0
            # ---- recall: first min(10, P) stored patterns, exact starts
            R = min(10, P)
            starts = Xis[:, :R, :]                     # (S, R, N)
            Ks = jnp.stack([hebb_k(jnp.asarray(x)) for x in Xis])
            recall_vmap = jax.jit(jax.vmap(
                lambda K, st, k: recall(K, st, k, max_sweeps=MAX_SWEEPS),
                in_axes=(0, 0, 0)))
            t0 = time.perf_counter()
            out = recall_vmap(Ks, jnp.asarray(starts),
                              jnp.stack([keys_for(N, s)[1]
                                         for s in range(N_SETS)]))
            jax.block_until_ready(out["states"])
            t_recall = time.perf_counter() - t0
            # final overlap of each chain with its TARGET pattern (diagonal
            # of the pattern x chain overlap matrix)
            ov_all = np.asarray(jax.vmap(
                lambda Xi, st: overlap(Xi, st.T))(jnp.asarray(Xis),
                                                  out["states"]))  # (S,P,R)
            ov = np.stack([np.diagonal(ov_all[s]) for s in range(N_SETS)])
            sweeps = np.asarray(out["sweeps_used"])
            conv = np.asarray(out["converged"])
            if not conv.all():
                incomplete.append({"N": N, "alpha": alpha,
                                   "n_not_converged": int((~conv).sum())})
            # aggregate: per set first, then over sets
            per_set_mean = ov.mean(axis=1)             # (S,)
            per_set_success = (ov >= 0.9).mean(axis=1)  # (S,)
            rec = {
                "N": N, "alpha": alpha, "P": P, "R": R,
                "unstable_mean": float(frac.mean()),
                "unstable_sem": float(frac.std(ddof=1) / np.sqrt(N_SETS)),
                "recall_mean": float(per_set_mean.mean()),
                "recall_sem": float(per_set_mean.std(ddof=1) / np.sqrt(N_SETS)),
                "success_mean": float(per_set_success.mean()),
                "success_sem": float(per_set_success.std(ddof=1) / np.sqrt(N_SETS)),
                "sweeps_mean": float(sweeps.mean()),
                "t_unstable_s": t_unstable, "t_recall_s": t_recall,
            }
            results.append(rec)
            unstable_all[(N, alpha)] = frac
            recall_all[(N, alpha)] = ov
            sweeps_all[(N, alpha)] = sweeps
            print(f"N={N:4d} alpha={alpha:.2f} P={P:4d}: unstable="
                  f"{rec['unstable_mean']:.2e}+-{rec['unstable_sem']:.1e} "
                  f"recall={rec['recall_mean']:.4f} "
                  f"success={rec['success_mean']:.3f} "
                  f"({t_unstable:.2f}s+{t_recall:.2f}s)", flush=True)
        print(f"N={N} done in {time.perf_counter()-tN:.1f}s", flush=True)

    # ---- crossings: first downward .9 crossing, linear interpolation
    crossings = {}
    for N in NS:
        ms = [r["recall_mean"] for r in results if r["N"] == N]
        als = [r["alpha"] for r in results if r["N"] == N]
        crossing = None
        for i in range(len(als) - 1):
            if ms[i] >= 0.9 and ms[i + 1] < 0.9:
                t = (ms[i] - 0.9) / (ms[i] - ms[i + 1])
                crossing = als[i] + t * (als[i + 1] - als[i])
                break
        crossings[str(N)] = crossing
    min_low = {}
    for N in NS:
        ms = [r["recall_mean"] for r in results
              if r["N"] == N and r["alpha"] <= 0.12]
        min_low[str(N)] = min(ms)

    summary = {
        "settings": {"NS": NS, "alphas": ALPHAS, "n_sets": N_SETS,
                     "max_sweeps": MAX_SWEEPS, "seed": SEED,
                     "P_rule": "P = round(alpha*N)",
                     "starts": "first min(10,P) exact stored patterns",
                     "keys": "master=PRNGKey(2026); key_N=fold_in(master,N); "
                             "key_set=fold_in(key_N,set); "
                             "key_recall=fold_in(key_set,7777)"},
        "points": results,
        "crossings_at_0.9_linear_interp": crossings,
        "crossing_method": "first adjacent alpha pair with mean>=0.9 then "
                           "<0.9; crossing = a_i + (m_i-0.9)/(m_i-m_{i+1})"
                           "*(a_{i+1}-a_i)",
        "min_recall_mean_alpha_le_0.12": min_low,
        "incomplete_runs": incomplete,
        "total_time_s": time.perf_counter() - t_start,
    }
    (EV / "part3_metrics.json").write_text(json.dumps(summary, indent=2))
    R_MAX = 10
    ov_pad = np.full((len(NS) * len(ALPHAS), N_SETS, R_MAX), np.nan)
    sw_pad = np.full_like(ov_pad, np.nan, dtype=np.float64)
    for i, (N, a) in enumerate([(N, a) for N in NS for a in ALPHAS]):
        R = recall_all[(N, a)].shape[1]
        ov_pad[i, :, :R] = recall_all[(N, a)]
        sw_pad[i, :, :R] = sweeps_all[(N, a)]
    np.savez_compressed(
        out_dir / "part3_raw.npz",
        alphas=np.array(ALPHAS), ns=np.array(NS),
        unstable=np.stack([unstable_all[(N, a)] for N in NS for a in ALPHAS]),
        overlaps=ov_pad, sweeps=sw_pad)
    print(json.dumps({"crossings": crossings, "min_low": min_low,
                      "incomplete": incomplete,
                      "total_min": summary["total_time_s"] / 60}, indent=2))
    print("PART3_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
