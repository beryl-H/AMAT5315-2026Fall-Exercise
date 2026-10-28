"""diag09 (minimal-diagnostics-03, v2): H1 — float32 vs float64 autodiff on
identical saved checkpoint values.

Pre-registered in PLAN.md / PLAN.json v2 of this directory (Codex-reviewed
before execution). Single-shot, read-only with respect to the repository
except for writing its own result file results/diag09/diag09_precision_gradients.json
(O_EXCL semantics; never overwrites). No training, no optimizer, no weight
regeneration: arrays are loaded from npz, kept in memory, and explicitly cast.
ALL float32 passes run before the single x64 toggle, then ALL float64 passes.

Engineering decision rule (NOT a course gate), fixed in PLAN.json v2:
  material defect: on whole or any non-degenerate block (norm64 >= 1e-12):
  relL2 > 1e-3 OR cosine < 0.999 (when defined) OR meaningful_zero_count > 0
  OR nonfinite32 > 0. Meaningful zero: g32==0 AND |g64| > max(1e-12,
  1e-6*||g64_whole||). g32-zero vs tiny g64 is never classified material.
Verdicts (three-way): supported_seed0_specific_engineering /
defect_present_not_seed0_specific / no_material_defect_at_snapshots.
None of these eliminates float32 training-dynamics effects in general or
supports a causal claim.

Positive control = deterministic METRIC-DETECTOR validation on one snapshot's
g64 (validates the metric implementation, NOT the float32 backend):
  (a) self: relL2 == 0, cosine 1;
  (b) g64*1.1: relL2 == 0.1 (float64 roundoff) and flagged;
  (c) zero largest-|g64| coordinate of a non-degenerate block:
      meaningful-zero count increments and block flags.
"""
from __future__ import annotations

import hashlib
import json
import os
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

OUTDIR = HERE / "results" / "diag09"
RESULT = OUTDIR / "diag09_precision_gradients.json"
if RESULT.exists():
    sys.exit(f"ABORT: result file already exists (exclusive namespace): {RESULT}")

ART = REPO / "artifacts" / "part4" / "formal-decoder-02"
SNAPS = [(s, c) for s in range(5) for c in ("init", "memorization", "final")]
RAW_FLOOR = 1e-30       # relL2 / cosine denominators only
NORM_FLOOR = 1e-12      # decision-level: blocks/whole below are degenerate
TOL_REL = 1e-3          # material-defect threshold (PLAN.json v2)
COS_MIN = 0.999
LOG_TIE_TOL = 1e-5      # baseline tie-in vs exact corresponding log row
DETECTOR_TOL = 1e-9     # float64 roundoff slack for detector checks

BLOCKS = {
    "embedding": ["emb", "pos"],
    "attention": ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"],
    "mlp": ["mlp1.w", "mlp1.b", "mlp2.w", "mlp2.b"],
    "head": ["head.w", "head.b"],
}


def leaf_name(path) -> str:
    return jax.tree_util.keystr(path).replace("'", "") \
        .replace("[", ".").replace("]", "")


def load_params(seed: int, ckpt: str) -> dict:
    p = grok.load_params(ART / f"main-s{seed}" / f"checkpoint-{ckpt}.npz")
    for x in jax.tree_util.tree_leaves(p):
        assert np.asarray(x).dtype == np.float32, "checkpoint arrays must be float32"
    return p


def expected_log_row(seed: int, ckpt: str) -> dict:
    """Exact corresponding log row: step 0 / memorization_step / last row."""
    log = json.loads((ART / f"main-s{seed}" / "train-log.json").read_text())
    if ckpt == "init":
        row = next(r for r in log if r["step"] == 0)
    elif ckpt == "memorization":
        mem = json.loads(
            (ART / f"main-s{seed}" / "completion-receipt.json").read_text()
        )["memorization_step"]
        row = next(r for r in log if r["step"] == mem)
    else:
        row = log[-1]
    return row


def vec_stats(g32: np.ndarray, g64: np.ndarray) -> dict:
    n32 = float(np.linalg.norm(g32.astype(np.float64)))
    n64 = float(np.linalg.norm(g64))
    d = g32.astype(np.float64) - g64
    out = {"abs_diff": float(np.linalg.norm(d)), "norm32": n32, "norm64": n64,
           "relL2": float(np.linalg.norm(d) / max(n64, RAW_FLOOR)),
           "nonfinite32": int(np.count_nonzero(~np.isfinite(g32))),
           "nonfinite64": int(np.count_nonzero(~np.isfinite(g64))),
           "degenerate": bool(n64 < NORM_FLOOR)}
    out["cosine"] = (None if (n32 < RAW_FLOOR or n64 < RAW_FLOOR)
                     else float(np.dot(g32.astype(np.float64), g64) / (n32 * n64)))
    return out


def whole_norm(g64_parts: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(x @ x) for x in g64_parts)))


def main() -> None:
    data = grok.build_dataset()
    split = grok.fixed_split(0.30)
    tok = jnp.asarray(data["inputs"][split["train"]])
    tgt = jnp.asarray(data["targets"][split["train"]])

    # ---- pass 1: ALL float32 passes (x64 OFF) ---------------------------
    assert jax.config.jax_enable_x64 is False
    store = {}                              # same arrays reused in pass 2
    for seed, ckpt in SNAPS:
        key = f"main-s{seed}/{ckpt}"
        params = load_params(seed, ckpt)    # float32, in-memory, no reload
        row = expected_log_row(seed, ckpt)
        L32, g32 = jax.value_and_grad(
            lambda p: grok.loss_acc(p, tok, tgt)[0])(params)
        tie = abs(float(L32) - row["train_loss"]) / max(abs(row["train_loss"]),
                                                        RAW_FLOOR)
        if tie > LOG_TIE_TOL:
            fail = {"harness_defect": "baseline_tie_in", "snapshot": key,
                    "tie_rel": tie, "logged_row": row}
            OUTDIR.mkdir(parents=True, exist_ok=True)
            fd = OUTDIR / "attempt01-baseline-tie-failure.json"
            if not fd.exists():
                fd.write_text(json.dumps(fail, indent=1) + "\n")
            sys.exit(f"ABORT: baseline tie-in failed at {key}: rel {tie}")
        leaves, _ = jax.tree_util.tree_flatten_with_path(g32)
        store[key] = {"params32": params, "L32": float(L32), "tie": float(tie),
                      "logged_row_step": row["step"],
                      "logged_train_loss": row["train_loss"],
                      "g32": [np.asarray(x) for _, x in leaves]}
    assert jax.config.jax_enable_x64 is False, "float32 pass contaminated"

    # ---- single precision-context switch --------------------------------
    jax.config.update("jax_enable_x64", True)
    assert jnp.asarray(1.0).dtype == jnp.float64, "x64 toggle failed -> ABORT"
    assert jnp.asarray(np.float32(1.0)).dtype == jnp.float32, \
        "float32 retention failed -> ABORT"

    # ---- pass 2: ALL float64 passes (same arrays, explicit cast) --------
    snapshots = {}
    for seed, ckpt in SNAPS:
        key = f"main-s{seed}/{ckpt}"
        s = store[key]
        for x in jax.tree_util.tree_leaves(s["params32"]):
            assert np.asarray(x).dtype == np.float32, "arrays changed -> ABORT"
        params64 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64),
                                          s["params32"])
        L64, g64 = jax.value_and_grad(
            lambda p: grok.loss_acc(p, tok, tgt)[0])(params64)
        g64_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(g64)]
        names = [leaf_name(p) for p, _ in
                 jax.tree_util.tree_flatten_with_path(g64)[0]]
        wnorm = whole_norm(g64_leaves)
        mz_thresh = max(NORM_FLOOR, 1e-6 * wnorm)

        per_leaf = {n: vec_stats(a, b)
                    for n, a, b in zip(names, s["g32"], g64_leaves)}
        blocks, block_mz = {}, {}
        for b, members in BLOCKS.items():
            idx = [names.index(m) for m in members if m in names]
            v32 = np.concatenate([s["g32"][i].ravel() for i in idx])
            v64 = np.concatenate([g64_leaves[i].ravel() for i in idx])
            blocks[b] = vec_stats(v32, v64)
            block_mz[b] = int(np.count_nonzero(
                (v32 == 0) & (np.abs(v64) > mz_thresh)))
        whole_v32 = np.concatenate([x.ravel() for x in s["g32"]])
        whole_v64 = np.concatenate([x.ravel() for x in g64_leaves])
        whole = vec_stats(whole_v32, whole_v64)
        whole_mz = int(np.count_nonzero(
            (whole_v32 == 0) & (np.abs(whole_v64) > mz_thresh)))

        decision_units = {**blocks, "whole": whole}
        material = any(
            (not u["degenerate"]) and (
                u["relL2"] > TOL_REL
                or (u["cosine"] is not None and u["cosine"] < COS_MIN)
                or u["nonfinite32"] > 0)
            for u in decision_units
        ) or (block_mz and any(v > 0 for k, v in block_mz.items()
                                if not blocks[k]["degenerate"]))
        material = bool(material or (whole_mz > 0 and not whole["degenerate"]))

        snapshots[key] = {
            "loss32": s["L32"], "loss64": float(L64),
            "loss_abs_diff": float(s["L32"] - L64),
            "loss_rel": float(abs(s["L32"] - L64) / max(abs(L64), RAW_FLOOR)),
            "logged_row_step": s["logged_row_step"],
            "logged_train_loss": s["logged_train_loss"],
            "log_tie_rel": s["tie"],
            "per_leaf_raw": per_leaf,
            "gauge_note": "raw leaf errors (incl. potentially gauge-sensitive "
                          "bk) reported separately; never inferred as underflow",
            "per_block": blocks,
            "whole": whole,
            "meaningful_zero_counts": {**block_mz, "whole": whole_mz},
            "meaningful_zero_threshold": mz_thresh,
            "material_defect": material}

    defects = {k: v["material_defect"] for k, v in snapshots.items()}
    s0_memfinal = any(defects[f"main-s0/{c}"] for c in ("memorization", "final"))
    s0_init = defects["main-s0/init"]
    others = [defects[f"main-s{s}/{c}"] for s in range(1, 5)
              for c in ("init", "memorization", "final")]
    if s0_memfinal and not s0_init and not any(others):
        verdict = "supported_seed0_specific_engineering"
    elif any(others) or s0_init:
        verdict = "defect_present_not_seed0_specific"
    else:
        verdict = "no_material_defect_at_snapshots"

    # ---- positive control: deterministic metric-detector validation -----
    ref_key = "main-s4/final"
    ref = snapshots[ref_key]
    names = list(ref["per_leaf_raw"].keys())
    # rebuild the exact float64 gradient of main-s4 final for the checks
    p64 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64), store[ref_key]["params32"])
    _, g_ref = jax.value_and_grad(
        lambda p: grok.loss_acc(p, tok, tgt)[0])(p64)
    g_ref = [np.asarray(x) for x in jax.tree_util.tree_leaves(g_ref)]
    flat = np.concatenate([x.ravel() for x in g_ref])

    a_ok = True
    s_self = vec_stats(flat, flat)
    a_ok &= (s_self["relL2"] == 0.0
             and s_self["cosine"] is not None
             and s_self["cosine"] >= 1.0 - 1e-12)   # sqrt-vs-dot roundoff slack
    s_scale = vec_stats(flat * 1.1, flat)
    a_ok &= (abs(s_scale["relL2"] - 0.1) <= DETECTOR_TOL
             and s_scale["relL2"] > TOL_REL)
    j_max = int(np.argmax(np.abs(flat)))
    flat_zero = flat.copy()
    flat_zero[j_max] = 0.0
    s_zero = vec_stats(flat_zero, flat)
    mz = int(np.count_nonzero(
        (flat_zero == 0) & (np.abs(flat) >
                            max(NORM_FLOOR, 1e-6 * float(np.linalg.norm(flat))))))
    a_ok &= (mz == 1 and s_zero["abs_diff"] > 0)
    detector_ok = bool(a_ok)

    out = {
        "diagnostic": "diag09_precision_gradients", "version": 2,
        "hypothesis": "H1", "grok_py_sha256": ACTUAL_SHA,
        "x64_flag_pass1": False, "x64_flag_pass2": True,
        "same_in_memory_arrays_reused": True, "init_params_called": False,
        "thresholds": {"relL2": TOL_REL, "cosine_min": COS_MIN,
                       "raw_floor": RAW_FLOOR, "decision_norm_floor": NORM_FLOOR,
                       "log_tie_tol": LOG_TIE_TOL,
                       "detector_tol": DETECTOR_TOL},
        "conclusion_qualification": "metric-detector validation validates the "
        "metric implementation, not the float32 backend; matched float64 "
        "computation may differ by backend paths; no branch eliminates "
        "float32 training-dynamics effects or supports a causal claim",
        "metric_detector_validation": {"self": s_self, "scaled_1.1": s_scale,
                                       "zeroed_max_coord": s_zero,
                                       "meaningful_zero_count": mz,
                                       "ok": detector_ok},
        "verdict_engineering": verdict,
        "material_defects": defects,
        "snapshots": snapshots}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if RESULT.exists():
        sys.exit("ABORT: result file appeared concurrently; nothing overwritten")
    fd = os.open(RESULT, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    print(f"diag09: detector={'PASS' if detector_ok else 'FAIL'} "
          f"H1={verdict} -> {RESULT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
