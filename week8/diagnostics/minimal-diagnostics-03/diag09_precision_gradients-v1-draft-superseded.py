"""diag09 (minimal-diagnostics-03): H1 — float32 vs float64 autodiff on identical
saved checkpoint values.

Pre-registered in PLAN.md / PLAN.json of this directory. Single-shot, read-only
with respect to the repository except for writing its own result file in the
exclusive namespace results/diag09_precision_gradients.json. No training, no
optimizer, no weight regeneration: arrays are loaded from npz and explicitly
cast. Do NOT run before Codex approves the plan.

Engineering decision rule (NOT a course gate), fixed in PLAN.json:
  material defect: whole relL2 > 1e-3 OR any block relL2 > 1e-3 OR cosine < 0.999
  (when defined) OR zero_loss_count > 0 with ||g64|| >= floor(=1e-30).
  H1 supported (engineering level) iff material defect at seed0 mem and/or final,
  none at seed0 init, none at any seeds 1-4 snapshot.
Positive control: bfloat16-quantized transient copy of main-s4 final must be
flagged (relL2 > 0.01) or the diagnostic is invalid (recorded, retained).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                 # week8/
sys.path.insert(0, str(REPO))

GROK_SHA = "d0501c3baf88a1194665123e47abc0a5b0035d0b203a56ca0dcc92853bc1d447"
ACTUAL_SHA = hashlib.sha256((REPO / "grok.py").read_bytes()).hexdigest()
if ACTUAL_SHA != GROK_SHA:
    sys.exit(f"ABORT: grok.py hash mismatch: {ACTUAL_SHA}")

import jax                              # noqa: E402
jax.config.update("jax_enable_x64", False)   # native float32 regime first
import jax.numpy as jnp                 # noqa: E402

import grok                             # frozen, read-only  # noqa: E402

RESULT = HERE / "results" / "diag09_precision_gradients.json"
if RESULT.exists():
    sys.exit(f"ABORT: result file already exists (exclusive namespace): {RESULT}")

ART = REPO / "artifacts" / "part4" / "formal-decoder-02"
SNAPS = [(s, c) for s in range(5) for c in ("init", "memorization", "final")]
FLOOR = 1e-30
TOL_REL = 1e-3          # material-defect threshold (PLAN.json)
COS_MIN = 0.999
CTRL_TOL = 1e-2         # positive-control flag threshold
LOG_TIE_TOL = 1e-5      # baseline tie-in to logged train_loss

BLOCKS = {
    "embedding": ["emb", "pos"],
    "attention": ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"],
    "mlp": ["mlp1.w", "mlp1.b", "mlp2.w", "mlp2.b"],
    "head": ["head.w", "head.b"],
}


def leaf_name(path) -> str:
    return jax.tree_util.keystr(path).replace("'", "").replace("[", ".") \
        .replace("]", "")


def load_params(seed: int, ckpt: str) -> dict:
    p = grok.load_params(ART / f"main-s{seed}" / f"checkpoint-{ckpt}.npz")
    for x in jax.tree_util.tree_leaves(p):
        assert np.asarray(x).dtype == np.float32, "checkpoint arrays must be float32"
    return p


def loss32_and_grad(params, tok, tgt):
    loss, g = jax.value_and_grad(lambda p: grok.loss_acc(p, tok, tgt)[0])(params)
    return float(loss), g


def stats(g32: np.ndarray, g64: np.ndarray) -> dict:
    d = g32.astype(np.float64) - g64
    n32, n64 = np.linalg.norm(g32.astype(np.float64)), np.linalg.norm(g64)
    zero = int(np.count_nonzero((g32 == 0) & (np.abs(g64) > 0)))
    spur = int(np.count_nonzero((g32 != 0) & (g64 == 0)))
    out = {"abs_diff": float(np.linalg.norm(d)), "norm64": float(n64),
           "relL2": float(np.linalg.norm(d) / max(n64, FLOOR)),
           "zero_loss_count": zero, "spurious_count": spur,
           "nonfinite32": int(np.count_nonzero(~np.isfinite(g32))),
           "nonfinite64": int(np.count_nonzero(~np.isfinite(g64))),
           "g64_degenerate": bool(n64 < FLOOR)}
    if n32 < FLOOR or n64 < FLOOR:
        out["cosine"] = None
    else:
        out["cosine"] = float(np.dot(g32.astype(np.float64), g64) / (n32 * n64))
    return out


def material(defs: dict) -> bool:
    for d in defs.values():
        if d["relL2"] > TOL_REL or d["zero_loss_count"] > 0 \
                or d["nonfinite32"] > 0:
            return True
        if d["cosine"] is not None and d["cosine"] < COS_MIN:
            return True
    return False


def main() -> None:
    data = grok.build_dataset()
    split = grok.fixed_split(0.30)
    tok = jnp.asarray(data["inputs"][split["train"]])
    tgt = jnp.asarray(data["targets"][split["train"]])

    per_snapshot, all_defects = {}, {}
    # ---- pass 1: float32 (x64 OFF) --------------------------------------
    assert jax.config.jax_enable_x64 is False
    for seed, ckpt in SNAPS:
        params = load_params(seed, ckpt)
        logged = json.loads(
            (ART / f"main-s{seed}" / "train-log.json").read_text())
        L32, g32 = loss32_and_grad(params, tok, tgt)
        tie = abs(L32 - logged[-1]["train_loss"]) / max(
            abs(logged[-1]["train_loss"]), FLOOR)
        flat32 = np.concatenate([np.asarray(x).ravel()
                                 for x in jax.tree_util.tree_leaves(g32)])
        per_snapshot[f"main-s{seed}/{ckpt}"] = {
            "loss32": L32, "logged_train_loss": logged[-1]["train_loss"],
            "log_tie_rel": float(tie), "_g32_flat": flat32}
    # ---- pass 2: float64 (same process, arrays cast explicitly) ----------
    jax.config.update("jax_enable_x64", True)
    assert jnp.asarray(1.0).dtype == jnp.float64, "x64 toggle failed -> ABORT"
    assert jnp.asarray(np.float32(1.0)).dtype == jnp.float32, \
        "float32 retention failed -> ABORT"
    for seed, ckpt in SNAPS:
        key = f"main-s{seed}/{ckpt}"
        params32 = load_params(seed, ckpt)
        params64 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64),
                                          params32)
        L64, g64 = loss32_and_grad(params64, tok, tgt)
        leaves, _ = jax.tree_util.tree_flatten_with_path(g64)
        defs = {}
        for path, leaf in leaves:
            name = leaf_name(path)
            idx = [i for i, (s, _) in enumerate(
                [(p, 0) for p in []])]  # placeholder no-op
            defs[name] = None
        # align float32 leaves by same tree order
        leaves32, _ = jax.tree_util.tree_flatten_with_path(
            jax.tree_util.tree_map(lambda x: x, per_snapshot[key].pop("_tree")))
        del leaves32
        per_snapshot[key]["loss64"] = L64
    # NOTE: the two-pass alignment above is intentionally re-done cleanly
    # below; the definitive computation follows.
    per_snapshot = {}

    jax.config.update("jax_enable_x64", False)
    store = {}
    for seed, ckpt in SNAPS:
        key = f"main-s{seed}/{ckpt}"
        params = load_params(seed, ckpt)
        logged = json.loads(
            (ART / f"main-s{seed}" / "train-log.json").read_text())
        L32, g32 = loss32_and_grad(params, tok, tgt)
        tie = abs(L32 - logged[-1]["train_loss"]) / max(
            abs(logged[-1]["train_loss"]), FLOOR)
        assert tie <= LOG_TIE_TOL, \
            f"baseline tie-in failed at {key}: rel {tie} -> harness defect"
        store[key] = {"g32": [np.asarray(x) for x in
                              jax.tree_util.tree_leaves(g32)],
                      "L32": L32, "tie": float(tie),
                      "logged": logged[-1]["train_loss"]}

    jax.config.update("jax_enable_x64", True)
    assert jnp.asarray(1.0).dtype == jnp.float64
    for seed, ckpt in SNAPS:
        key = f"main-s{seed}/{ckpt}"
        params32 = load_params(seed, ckpt)
        params64 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64),
                                          params32)
        L64, g64 = loss32_and_grad(params64, tok, tgt)
        g64_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(g64)]
        names = [leaf_name(p) for p, _ in
                 jax.tree_util.tree_flatten_with_path(g64)[0]]
        per_leaf = {n: stats(a, b) for n, a, b in
                    zip(names, store[key]["g32"], g64_leaves)}
        blocks = {}
        for b, members in BLOCKS.items():
            v32 = np.concatenate([store[key]["g32"][names.index(m)].ravel()
                                  for m in members if m in names])
            v64 = np.concatenate([g64_leaves[names.index(m)].ravel()
                                  for m in members if m in names])
            blocks[b] = stats(v32, v64)
        whole = stats(np.concatenate([x.ravel() for x in store[key]["g32"]]),
                      np.concatenate([x.ravel() for x in g64_leaves]))
        rec = {"loss32": store[key]["L32"], "loss64": float(L64),
               "loss_abs_diff": float(store[key]["L32"] - L64),
               "loss_rel": float(abs(store[key]["L32"] - L64)
                                 / max(abs(L64), FLOOR)),
               "log_tie_rel": store[key]["tie"],
               "logged_train_loss": store[key]["logged"],
               "per_leaf": per_leaf, "per_block": blocks, "whole": whole,
               "material_defect": material({**blocks, "whole": whole})}
        per_snapshot[key] = rec
        all_defects[key] = rec["material_defect"]

    # ---- positive defect control (transient only) ------------------------
    p64 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64),
        grok.load_params(ART / "main-s4" / "checkpoint-final.npz"))
    p_bf16 = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16)
                                    .astype(jnp.float64), p64)
    _, g_ref = loss32_and_grad(p64, tok, tgt)
    _, g_def = loss32_and_grad(p_bf16, tok, tgt)
    r = stats(np.concatenate([np.asarray(x).ravel()
                              for x in jax.tree_util.tree_leaves(g_def)]),
              np.concatenate([np.asarray(x).ravel()
                              for x in jax.tree_util.tree_leaves(g_ref)]))
    control_ok = r["relL2"] > CTRL_TOL

    seeds14_clean = not any(all_defects[f"main-s{s}/{c}"]
                            for s in range(1, 5) for c in SNAPS[0:3][0:3]
                            for c in ("init", "memorization", "final"))
    s0_defect = any(all_defects[f"main-s0/{c}"]
                    for c in ("memorization", "final"))
    s0_init_clean = not all_defects["main-s0/init"]
    h1 = "supported_engineering_level" if (s0_defect and s0_init_clean
                                           and seeds14_clean) else (
        "rejected_not_discriminating")

    out = {"diagnostic": "diag09_precision_gradients",
           "hypothesis": "H1",
           "grok_py_sha256": ACTUAL_SHA,
           "x64_flag_pass1": False, "x64_flag_pass2": True,
           "thresholds": {"relL2": TOL_REL, "cosine_min": COS_MIN,
                          "floor": FLOOR, "log_tie_tol": LOG_TIE_TOL,
                          "positive_control_min_relL2": CTRL_TOL},
           "positive_control": {**r, "ok": bool(control_ok)},
           "material_defects": all_defects,
           "h1_verdict_engineering": h1,
           "snapshots": {k: {kk: vv for kk, vv in v.items()}
                         for k, v in per_snapshot.items()}}
    RESULT.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"diag09: positive_control={'PASS' if control_ok else 'FAIL'} "
          f"H1={h1} -> {RESULT.name}")


if __name__ == "__main__":
    main()
