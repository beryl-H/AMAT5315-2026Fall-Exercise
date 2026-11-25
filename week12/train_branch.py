#!/usr/bin/env python3
"""Train the shared 12 -> 48 tanh -> 1 branching scorer (AMAT5315 Week 12, Part 3).

Reads the branching-label dataset from Part 2, trains a per-candidate JAX
network with masked cross entropy against equal-mass tie targets, retains the
best validation model, and writes model.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# JAX platform and x64 must be configured before any arrays are created.
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.nn
import jax.numpy as jnp
import numpy as np

FEATURES = 12
HIDDEN = 48
TIE_TOL = 1e-9          # tight tolerance defining equal-score ties (reported in settings)
SEED = 2026
EPOCHS = 120
BATCH = 64
LR = 0.004
BETA1, BETA2, EPS_ADAM = 0.9, 0.999, 1e-8
L2 = 2e-4               # applied only to w1 and w2
N_PROBES = 6


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def load_dataset(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") != "amat5315.week12.spin-glass.labels.v1":
        sys.exit("expected revised spin-glass teacher labels")
    for key in ("train", "validation", "feature_names", "train_instance_ids", "validation_instance_ids"):
        if key not in data:
            sys.exit(f"dataset missing key {key!r}")
    train_ids = list(data.get("train_instance_ids", []))
    val_ids = list(data.get("validation_instance_ids", []))
    if set(train_ids) & set(val_ids):
        sys.exit("train and validation instance id sets overlap")
    # Verify actual record memberships against the declared splits.
    for split, ids, name in ((data["train"], train_ids, "train"),
                             (data["validation"], val_ids, "validation")):
        seen = {rec["instance_id"] for rec in split}
        if len(ids) != len(set(ids)) or not seen <= set(ids):
            sys.exit(f"{name} records do not match {name}_instance_ids")
    return data


def collect_nodes(split: list[dict]) -> list[dict]:
    """Keep only nodes with >= 2 candidates; attach instance ids."""
    nodes = []
    for rec in split:
        cands = rec["candidates"]
        if len(cands) < 2:
            continue
        feats = np.asarray([c["features"] for c in cands], dtype=np.float64)
        if feats.shape[1] != FEATURES:
            sys.exit("candidate feature length is not 12")
        if not np.all(np.isfinite(feats)):
            sys.exit("non-finite features in dataset")
        scores = np.asarray([c["score"] for c in cands], dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            sys.exit("non-finite teacher scores in dataset")
        nodes.append({
            "instance_id": rec["instance_id"],
            "node": rec["node"],
            "columns": [c["spin"] for c in cands],
            "scores": scores,
            "features": feats,
        })
    return nodes


def tie_targets(scores: np.ndarray) -> np.ndarray:
    """Equal-mass target over candidates tied for the max score (within TIE_TOL)."""
    best = scores.max()
    ties = np.abs(scores - best) <= TIE_TOL * max(1.0, abs(best))
    return ties.astype(np.float64) / ties.sum()


class Padded:
    """Fixed-shape padded node arrays: stable shapes, very few JIT compilations."""

    def __init__(self, nodes: list[dict], max_k: int):
        self.max_k = max_k
        self.n = len(nodes)
        self.feats = np.zeros((self.n, max_k, FEATURES), dtype=np.float64)
        self.cand_mask = np.zeros((self.n, max_k), dtype=bool)
        self.scores = np.zeros((self.n, max_k), dtype=np.float64)
        for i, nd in enumerate(nodes):
            k = len(nd["columns"])
            self.feats[i, :k] = nd["features"]
            self.cand_mask[i, :k] = True
            self.scores[i, :k] = nd["scores"]
        self.node_mask = np.ones(self.n, dtype=bool)

    @staticmethod
    def stack(nodes: list[dict], n_slots: int, max_k: int) -> "Padded":
        base = Padded(nodes, max_k)
        if base.n > n_slots:
            raise ValueError("too many nodes for slot count")
        if base.n == n_slots:
            return base
        feats = np.zeros((n_slots, max_k, FEATURES), dtype=np.float64)
        cand = np.zeros((n_slots, max_k), dtype=bool)
        scores = np.zeros((n_slots, max_k), dtype=np.float64)
        feats[:base.n], cand[:base.n], scores[:base.n] = base.feats, base.cand_mask, base.scores
        out = Padded.__new__(Padded)
        out.max_k, out.n = max_k, n_slots
        out.feats, out.cand_mask, out.scores = feats, cand, scores
        out.node_mask = np.arange(n_slots) < base.n
        return out


def init_params(key: jax.Array) -> dict:
    """Xavier (tanh-appropriate) initialisation."""
    k1, k2 = jax.random.split(key)
    lim1 = math.sqrt(6.0 / (FEATURES + HIDDEN))
    lim2 = math.sqrt(6.0 / (HIDDEN + 1))
    return {
        "w1": jax.random.uniform(k1, (FEATURES, HIDDEN), minval=-lim1, maxval=lim1),
        "b1": jnp.zeros(HIDDEN),
        "w2": jax.random.uniform(k2, (HIDDEN,), minval=-lim2, maxval=lim2),
        "b2": jnp.zeros(()),
    }


def logits(params: dict, feats: jax.Array) -> jax.Array:
    """Per-candidate score; independent per candidate, hence permutation-equivariant."""
    h = jnp.tanh(feats @ params["w1"] + params["b1"])          # (..., HIDDEN)
    return jnp.sum(h * params["w2"], axis=-1) + params["b2"]   # (...)


def cross_entropy(params: dict, feats, cand_mask, scores, node_mask) -> jax.Array:
    """Cross-entropy for rows with candidates, excluding padded minibatch rows.

    Targets put equal mass on candidates whose *scores* tie for the maximum
    within TIE_TOL; logits are shifted by the row max for stable logsumexp.
    """
    lg = logits(params, feats)                                 # (N, K)
    very_neg = jnp.asarray(-1e30)
    lg = jnp.where(cand_mask, lg, very_neg)
    best = jnp.max(jnp.where(cand_mask, scores, -jnp.inf), axis=1, keepdims=True)
    best = jnp.where(node_mask[:, None], best, 0.0)
    tolerance = TIE_TOL * jnp.maximum(1.0, jnp.abs(best))
    ties = cand_mask & (jnp.abs(scores - best) <= tolerance)
    target = ties / jnp.maximum(jnp.sum(ties, axis=1, keepdims=True), 1)
    logZ = jax.nn.logsumexp(lg, axis=1)
    ce = logZ - jnp.sum(target * lg, axis=1)                   # = logZ - E_target[lg]
    per_node = jnp.where(node_mask, ce, 0.0)
    return jnp.sum(per_node) / jnp.maximum(jnp.sum(node_mask), 1)


def loss_fn(params: dict, feats, cand_mask, scores, node_mask) -> jax.Array:
    l2 = jnp.sum(params["w1"] ** 2) + jnp.sum(params["w2"] ** 2)
    return cross_entropy(params, feats, cand_mask, scores, node_mask) + L2 * l2


def eval_nodes(params, feats, cand_mask, scores, node_mask):
    """(mean_ce, top1, top3, chance) on a padded node array (NumPy side)."""
    lg = np.asarray(logits(params, jnp.asarray(feats)))        # (N, K) float64
    ces, top1s, top3s, chances = [], [], [], []
    for i in range(feats.shape[0]):
        if not node_mask[i]:
            continue
        k = int(cand_mask[i].sum())
        if k < 2:
            continue
        l = lg[i, :k]
        logZ = np.logaddexp.reduce(l)
        targets = tie_targets(scores[i, :k])
        ces.append(logZ - float(np.sum(targets * l)))
        order = np.argsort(-l, kind="stable")                  # stable order breaks ties
        correct = targets > 0
        top1s.append(correct[order[0]])
        top3s.append(np.any(correct[order[:3]]))
        chances.append(np.count_nonzero(correct) / k)
    if not ces:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (float(np.mean(ces)), float(np.mean(top1s)),
            float(np.mean(top3s)), float(np.mean(chances)))


def main() -> None:
    args = parse_args()
    data = load_dataset(args.dataset)

    train_nodes = collect_nodes(data["train"])
    val_nodes = collect_nodes(data["validation"])
    if not train_nodes or not val_nodes:
        sys.exit("need nodes with >= 2 candidates in both splits")

    # Normalisation statistics from real train candidates only.
    all_train = np.concatenate([nd["features"] for nd in train_nodes], axis=0)
    mean = all_train.mean(axis=0)
    std = np.maximum(all_train.std(axis=0), 1e-6)
    for nd in list(train_nodes) + list(val_nodes):
        nd["raw_features"] = nd["features"]
        nd["features"] = (nd["features"] - mean) / std

    max_k = max(max(len(nd["columns"]) for nd in train_nodes),
                max(len(nd["columns"]) for nd in val_nodes))

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}

    @jax.jit
    def update(p, m, v, step, f, cm, s, nm):
        g = jax.grad(loss_fn)(p, f, cm, s, nm)
        m2 = {k: BETA1 * m[k] + (1 - BETA1) * g[k] for k in g}
        v2 = {k: BETA2 * v[k] + (1 - BETA2) * g[k] ** 2 for k in g}
        mhat = {k: m2[k] / (1 - BETA1 ** step) for k in g}
        vhat = {k: v2[k] / (1 - BETA2 ** step) for k in g}
        p2 = {k: p[k] - LR * mhat[k] / (jnp.sqrt(vhat[k]) + EPS_ADAM) for k in g}
        return p2, m2, v2

    train_pad = Padded(train_nodes, max_k)
    val_pad = Padded(val_nodes, max_k)
    measure = jax.jit(cross_entropy)
    train_arrays = tuple(jnp.asarray(x) for x in (train_pad.feats, train_pad.cand_mask, train_pad.scores, train_pad.node_mask))

    rng = np.random.default_rng(SEED)
    history = []
    best_ce, best_params, best_epoch = float("inf"), None, -1
    t0 = time.perf_counter()
    optimizer_step = 0

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_nodes))
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            b = Padded.stack([train_nodes[i] for i in idx], BATCH, max_k)
            f = jnp.asarray(b.feats)
            cm = jnp.asarray(b.cand_mask)
            s = jnp.asarray(b.scores)
            nm = jnp.asarray(b.node_mask)
            optimizer_step += 1
            params, m, v = update(params, m, v, jnp.asarray(optimizer_step), f, cm, s, nm)
        val_ce, top1, top3, chance = eval_nodes(
            params, val_pad.feats, val_pad.cand_mask, val_pad.scores, val_pad.node_mask)
        train_ce = float(measure(params, *train_arrays))
        if not (math.isfinite(train_ce) and math.isfinite(val_ce)):
            sys.exit(f"non-finite cross-entropy at epoch {epoch}")
        history.append({"epoch": epoch, "train_ce": train_ce,
                        "val_ce": val_ce, "top1": top1, "top3": top3,
                        "chance": chance})
        if (epoch == 1 or epoch % 5 == 0) and val_ce < best_ce:
            best_ce, best_params, best_epoch = val_ce, jax.device_get(params), epoch

    if best_params is None:
        sys.exit("no validation improvement recorded")
    params = best_params
    val_ce, top1, top3, chance = eval_nodes(
        params, val_pad.feats, val_pad.cand_mask, val_pad.scores, val_pad.node_mask)
    one_over_k = float(np.mean([1.0 / len(nd["columns"]) for nd in val_nodes]))
    print(f"chosen epoch {best_epoch}: val_ce={val_ce:.6f} top1={top1:.4f} "
          f"top3={top3:.4f} chance={chance:.4f} (uniform 1/K={one_over_k:.4f})")
    train_ids = {nd["instance_id"] for nd in train_nodes}
    val_ids = {nd["instance_id"] for nd in val_nodes}
    print(f"train: {len(train_nodes)} nodes / {len(train_ids)} graphs, "
          f"validation: {len(val_nodes)} nodes / {len(val_ids)} graphs, "
          f"overlap: {len(train_ids & val_ids)}")

    # Six deterministic validation probes: prefer one node per validation
    # instance (ascending instance_id), then fill remaining slots in dataset order.
    probe_nodes, used = [], set()
    for iid in sorted({nd["instance_id"] for nd in val_nodes}):
        first = next(nd for nd in val_nodes if nd["instance_id"] == iid)
        probe_nodes.append(first)
        used.add(id(first))
        if len(probe_nodes) == N_PROBES:
            break
    for nd in val_nodes:
        if len(probe_nodes) == N_PROBES:
            break
        if id(nd) not in used:
            probe_nodes.append(nd)
            used.add(id(nd))

    probes = []
    if len(probe_nodes) != N_PROBES:
        sys.exit("need six validation nodes to export parity probes")
    for nd in probe_nodes:
        raw = nd["raw_features"]
        feats_j = jnp.asarray(nd["features"])[None, ...]
        lg = np.asarray(logits(params, feats_j))[0]         # float64 forward
        pr = np.asarray(jax.nn.softmax(jnp.asarray(lg)))
        if not (np.all(np.isfinite(lg)) and np.all(np.isfinite(pr))
                and abs(pr.sum() - 1.0) < 1e-9):
            sys.exit("probe produced non-finite scores/probabilities")
        # Independent joint candidate permutation-equivariance check.
        perm = rng.permutation(len(lg))
        lg_p = np.asarray(logits(params, feats_j[:, perm]))[0]
        if not np.allclose(lg[perm], lg_p, atol=1e-10, rtol=0.0):
            sys.exit("permutation equivariance check failed")
        probes.append({
            "instance_id": nd["instance_id"],
            "node": nd["node"],
            "columns": nd["columns"],
            "features": raw.tolist(),
            "logits": lg.tolist(),
            "probabilities": pr.tolist(),
        })

    out = {
        "schema": "amat5315.week12.spin-glass.model.v1",
        "feature_names": data["feature_names"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "w1": np.asarray(params["w1"]).tolist(),
        "b1": np.asarray(params["b1"]).tolist(),
        "w2": np.asarray(params["w2"]).tolist(),
        "b2": float(params["b2"]),
        "history": history,
        "best_epoch": best_epoch,
        "probes": probes,
        "train_instance_ids": data["train_instance_ids"],
        "validation_instance_ids": data["validation_instance_ids"],
        "settings": {
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH, "lr": LR,
            "beta1": BETA1, "beta2": BETA2, "adam_epsilon": EPS_ADAM,
            "l2": L2, "l2_on": ["w1", "w2"], "hidden": HIDDEN,
            "activation": "tanh", "tie_tolerance": TIE_TOL,
            "max_candidates_padded": max_k,
            "platform": jax.default_backend(), "jax_version": jax.__version__,
            "numpy_version": np.__version__, "uniform_one_over_k": one_over_k,
            "runtime_seconds": time.perf_counter() - t0,
        },
    }
    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
