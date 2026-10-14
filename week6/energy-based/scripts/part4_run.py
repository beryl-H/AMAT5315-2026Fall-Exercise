"""Part 4 driver: modern Hopfield retrieval vs classical recall on MNIST.

Key derivation (fixed before outcomes):
  master = PRNGKey(2026)
  key_P      = fold_in(master, P)                # per-P stream
  key_query  = fold_in(key_P, 1)                 # corrupted-query spins
  (queries: target index q % P; lower 14 rows independent random +/-1)
Queries are computed once per P and reused for EVERY beta and for classical
recall. Raw results are saved incrementally per point (a packaging failure
cannot lose a completed scan) under artifacts/part4/evaluation-<ts>/.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ebm.data import load_mnist_pair
from ebm.hopfield import hebb_k, recall
from ebm.modern_hopfield import (LOWER_ROWS, attention_update,
                                 nearest_pattern, retrieval_success,
                                 sign_overlap, modern_update, spinize)

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
ART = ROOT / "artifacts/part4"
PS = [3, 10, 30, 100, 300, 1000]
BETAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
NQ = 100
SEED = 2026
MAX_SWEEPS = 200


def queries_for_P(X, P):
    """(100, 784) corrupted queries + target indices, seed 2026."""
    key_P = jax.random.fold_in(jax.random.PRNGKey(SEED), P)
    key_query = jax.random.fold_in(key_P, 1)
    targets = np.arange(NQ) % P
    ks = jax.random.split(key_query, NQ)
    spins = jax.vmap(lambda k: jax.random.bernoulli(
        k, 0.5, (784,)).astype(jnp.float32) * 2 - 1)(ks)
    lower_mask = np.zeros(784, dtype=bool)
    lower_mask[LOWER_ROWS * 28:] = True
    Q = np.where(lower_mask[None, :], np.asarray(spins),
                 np.asarray(X)[:, targets].T)
    return Q, targets, bool(lower_mask.sum() == 392)


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ART / f"evaluation-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    train_images, _ = load_mnist_pair(
        ROOT / "inputs/mnist/train-images-idx3-ubyte.gz",
        ROOT / "inputs/mnist/train-labels-idx1-ubyte.gz")
    settings = {"PS": PS, "betas": BETAS, "n_queries": NQ, "seed": SEED,
                "max_sweeps": MAX_SWEEPS,
                "keys": "master=PRNGKey(2026); key_P=fold_in(master,P); "
                        "key_query=fold_in(key_P,1)",
                "policy": "sign(0)=0 for overlap; nearest by spin overlap, "
                          "first-index argmax; success m>=0.95 AND target nearest",
                "jax_version": jax.__version__}
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps(settings), flush=True)

    summary = {"settings": settings, "points": [], "global_max_attention_diff": 0.0}
    t_start = time.perf_counter()
    for P in PS:
        t0 = time.perf_counter()
        X = spinize(train_images[:P])             # (784, P)
        Q, targets, mask_ok = queries_for_P(X, P)
        assert mask_ok
        Qt = jnp.asarray(Q, dtype=jnp.float32)    # (100, 784)
        # ---- modern updates (vectorized over queries) + attention equality
        max_diff = 0.0
        modern_success = {}
        modern_out = {}
        for beta in BETAS:
            outs = jax.vmap(lambda q: modern_update(X, q, beta))(Qt)
            att = jax.vmap(lambda q: attention_update(
                jnp.asarray(X), q, beta))(Qt)
            d = float(jnp.max(jnp.abs(outs - att)))
            max_diff = max(max_diff, d)
            sgn = jnp.sign(outs)
            succ = []
            for q in range(NQ):
                succ.append(retrieval_success(jnp.asarray(X),
                                              int(targets[q]), sgn[q]))
            modern_success[beta] = succ
            modern_out[beta] = np.asarray(outs)
        # ---- classical recall (Part 3 production implementation)
        K = hebb_k(jnp.asarray(X).T.astype(jnp.int32))          # (784,784)
        rec = recall(K, jnp.asarray(Q).astype(jnp.int32),
                     jax.random.fold_in(
                         jax.random.fold_in(jax.random.PRNGKey(SEED), P), 2),
                     max_sweeps=MAX_SWEEPS)
        conv = np.asarray(rec["converged"])
        assert conv.all(), f"classical recall hit safety cap at P={P}"
        classical_out = np.asarray(rec["states"]).astype(np.float32)
        classical_success = [retrieval_success(jnp.asarray(X),
                                               int(targets[q]),
                                               classical_out[q])
                             for q in range(NQ)]
        # ---- overlaps and nearest indices saved raw
        t_point = time.perf_counter() - t0
        np.savez_compressed(
            out / f"point_P{P}.npz",
            queries=Q, targets=targets,
            **{f"modern_out_beta{b}": modern_out[b] for b in BETAS},
            classical_out=classical_out,
            classical_sweeps=np.asarray(rec["sweeps_used"]),
            classical_converged=conv)
        point = {
            "P": P,
            "modern_success_frac": {str(b): float(np.mean(modern_success[b]))
                                    for b in BETAS},
            "classical_success_frac": float(np.mean(classical_success)),
            "max_attention_diff": max_diff,
            "classical_sweeps_mean": float(np.asarray(rec["sweeps_used"]).mean()),
            "time_s": t_point,
            # weights of the FIRST query (predeclared) at P=100 for the plot
        }
        if P == 100:
            point["weights_q0"] = {
                str(b): np.asarray(jax.nn.softmax(
                    b * (Qt[0] @ jnp.asarray(X)))).tolist()
                for b in (0.003, 0.03, 0.3)}
        summary["points"].append(point)
        summary["global_max_attention_diff"] = max(
            summary["global_max_attention_diff"], max_diff)
        (EV / "part4_metrics.json").write_text(json.dumps(summary, indent=2))
        print(f"P={P:5d}: modern β=.3 success={point['modern_success_frac']['0.3']:.2f} "
              f"classical={point['classical_success_frac']:.2f} "
              f"maxdiff={max_diff:.2e} ({t_point:.1f}s)", flush=True)

    print(f"GLOBAL max attention diff: {summary['global_max_attention_diff']:.3e} "
          f"({'< 1e-5 OK' if summary['global_max_attention_diff'] < 1e-5 else 'FAIL'})",
          flush=True)
    summary["total_time_s"] = time.perf_counter() - t_start
    (EV / "part4_metrics.json").write_text(json.dumps(summary, indent=2))
    print("PART4_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
