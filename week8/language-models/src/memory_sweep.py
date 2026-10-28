"""Part 3 memory sweep driver: exact all-queries ISR curves for the five
stores (outer, least-squares, ReLU^1/2/3).

Frozen protocol (SPEC): keys/values unit vectors from one seeded RNG generated
as largest-65536 arrays per d (seeds 101/102/103), smaller M are prefixes;
draws independent; EVERY stored key is queried with every store (no
subsampling, no analytic substitutes); R own-normalized (own coefficient 1 for
outer/ReLU^p, h_jj for LS); per-draw aggregate = median over keys; final curve
= median of the 3 draws; capacity = log-log crossing of R = 1 (None if never
crossed). ReLU powers share the query tiles. No M x M allocation at any M.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

STOKES = ["outer", "ls", "relu1", "relu2", "relu3"]
M_GRID = [2 ** e for e in range(3, 17)]  # 8 .. 65536
D_GRID = [16, 32, 64]
DRAW_SEEDS = [101, 102, 103]
TILE = 256
BLAS_THREADS = 8  # predeclared; set before numpy import via env in the CLI


FULL_M = 65536


def generate_unit(M_max: int, d: int, seed: int):
    """Unit keys/values, each generated as the FULL 65536-row array once per
    (d, seed) so every smaller M is an exact prefix of the same draw (frozen
    protocol). Keys and values are independent (sequential draws from one
    seeded generator, both at full size)."""
    assert M_max <= FULL_M
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(FULL_M, d))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.normal(size=(FULL_M, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return K[:M_max], V[:M_max]


def sweep_case(K: np.ndarray, V: np.ndarray, tile: int = TILE) -> dict:
    """ISR vector per store for M queries (K, V: M x d unit rows)."""
    M, d = K.shape
    Vn = np.linalg.norm(V, axis=1)
    out = {}
    # outer product store: one d x d aggregate, own coeff = 1
    F = K @ (K.T @ V)
    out["outer"] = np.linalg.norm(F - V, axis=1) / Vn
    # least-squares store: SVD pseudoinverse, no M x M
    Kpinv = np.linalg.pinv(K)  # d x M
    F = K @ (Kpinv @ V)
    h = np.einsum("ja,aj->j", K, Kpinv)
    out["ls"] = np.linalg.norm(F - h[:, None] * V, axis=1) / (np.abs(h) * Vn)
    # ReLU^p stores share query tiles: scores tile x M, own coeff = 1
    F = {p: np.zeros_like(V) for p in (1, 2, 3)}
    for s in range(0, M, tile):
        e = min(s + tile, M)
        S = K[s:e] @ K.T  # (e-s) x M
        W = np.maximum(S, 0.0)
        for p in (1, 2, 3):
            F[p][s:e] = (W ** p) @ V
    for p in (1, 2, 3):
        out[f"relu{p}"] = np.linalg.norm(F[p] - V, axis=1) / Vn
    assert all(len(r) == M for r in out.values())  # query count == M per store
    return out


def aggregate_R(R: np.ndarray) -> dict:
    """Per-draw aggregate over ALL queried keys: median primary; the rest
    descriptive. No post-hoc selection."""
    return {
        "median": float(np.median(R)),
        "mean": float(R.mean()),
        "rms": float(np.sqrt((R ** 2).mean())),
        "q25": float(np.quantile(R, 0.25)),
        "q75": float(np.quantile(R, 0.75)),
    }


def capacity_loglog(M_grid, med_curve) -> float | None:
    """First crossing of R = 1, linear interpolation on log-log axes.
    Returns None when no crossing is reached (never misclassified)."""
    pts = [(M, med_curve[M]) for M in sorted(M_grid)]
    for (m0, r0), (m1, r1) in zip(pts[:-1], pts[1:]):
        if r0 <= 1.0 <= r1:
            if r1 == r0:
                return float(m1)
            t = (np.log(1.0) - np.log(r0)) / (np.log(r1) - np.log(r0))
            return float(np.exp(np.log(m0) + t * (np.log(m1) - np.log(m0))))
    return None


def run_sweep(d_values, M_values, seeds, art_dir, log=print, formal=False):
    """Drive cases; JSONL progress per case with source hash and settings."""
    os.makedirs(art_dir, exist_ok=True)
    src_hash = {f: hashlib.sha256(open(os.path.join(HERE, "src", f), "rb").read()).hexdigest()
                for f in ["memory_sweep.py"]}
    jsonl = open(os.path.join(art_dir, "progress.jsonl"), "a")
    curves = {d: {s: {} for s in STOKES} for d in d_values}
    for d in d_values:
        Kfull, Vfull = {}, {}
        for seed in seeds:
            Kfull[seed], Vfull[seed] = generate_unit(max(M_values), d, seed)
        for M in M_values:
            per_store = {s: [] for s in STOKES}
            for seed in seeds:
                out = sweep_case(Kfull[seed][:M], Vfull[seed][:M])
                for s in STOKES:
                    per_store[s].append(aggregate_R(out[s]))
                    jsonl.write(json.dumps({
                        "case": {"d": d, "M": M, "seed": seed, "store": s},
                        "aggregate": per_store[s][-1], "tile": TILE,
                        "source_hash": src_hash, "formal": formal}) + "\n")
            for s in STOKES:
                curves[d][s][M] = float(np.median([a["median"] for a in per_store[s]]))
            jsonl.flush()
            log(f"d={d} M={M}: " + " ".join(f"{s}={curves[d][s][M]:.4f}" for s in STOKES))
    jsonl.close()
    result = {"d_values": d_values, "M_values": M_values, "seeds": seeds,
              "curves_median_of_3": {str(d): {s: {str(M): curves[d][s][M] for M in M_values}
                                              for s in STOKES} for d in d_values},
              "capacity": {str(d): {s: capacity_loglog(M_values, curves[d][s])
                                    for s in STOKES} for d in d_values},
              "source_hash": src_hash, "formal": formal}
    with open(os.path.join(art_dir, "sweep-summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--attempt", default=None)
    args = ap.parse_args(argv)
    if args.smoke:
        art = os.path.join(HERE, "artifacts", "part3", "smoke-01")
        res = run_sweep([16], [8, 32, 128], [101], art, formal=False)
        # smoke checks: chunked path already covered by tests; report only
        print("SMOKE OK:", json.dumps(res["curves_median_of_3"]["16"], indent=1))
        return 0
    if args.formal:
        if not args.attempt:
            print("--attempt required for --formal"); return 2
        art_root = os.path.join(HERE, "artifacts", "part3")
        d = os.path.join(art_root, args.attempt)
        if os.path.exists(d):
            print(f"attempt exists: {d}"); return 2
        os.makedirs(d)
        # full sweep is dispatched by the root, not by this agent
        res = run_sweep(D_GRID, M_GRID, DRAW_SEEDS, d, formal=True)
        print("FORMAL SWEEP DONE; gates checked against SPEC by evidence step")
        return 0
    print("nothing to do: pass --smoke (agent) or --formal --attempt (root dispatch)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
