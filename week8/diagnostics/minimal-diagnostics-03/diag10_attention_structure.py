"""diag10 (minimal-diagnostics-03, v2): H2 — attention saturation and/or order
sensitivity on symmetric unseen pairs (narrowed per Codex review).

Pre-registered in PLAN.md / PLAN.json v2 of this directory. Single-shot,
read-only with respect to the repository except for its own result file
results/diag10/diag10_attention_structure.json (O_EXCL; never overwrites).
No training; transient copies only for the single descriptive intervention.

The frozen helper grok.attention_weights returns only the first batch sample
([0][None]); a small diagnostic forward (same frozen ops, float32, no x64)
returns logits and per-sample attention maps for all B inputs. It is trusted
only after the baseline-match gate: elementwise |custom - frozen| <= 1e-5 +
1e-6*|frozen| on the test-split logits, and recomputed test accuracy equal to
the exact corresponding log row (step 0 / memorization_step / last) to rel 1e-6.

Engineering decision rule (NOT a course gate), fixed in PLAN.json v2:
  H2a saturation: seed0 head_mean_entropy at final strictly below every
      grokked seed AND seed0 head_mean_sat_frac > 0.5 while every grokked
      seed < 0.1.
  H2c order sensitivity: seed0 paired prediction symmetry disagreement on
      S = {(a,b): a!=b, test[a,b] AND test[b,a]} > 0.5 while every grokked
      seed < 0.05. (predeclared positive association, NOT causal)
  supported: H2a or H2c at final AND same-direction at memorization.
  falsified: all seed0 metrics within seeds-1-4 [min,max] envelope.
  INCONCLUSIVE: no signature but >=1 metric outside the envelope.
Positional (pos->0) intervention is DESCRIPTIVE ONLY. Entropy/saturation and
order disagreement are static-snapshot associations; they never establish a
necessary cause of past learning failure.
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
jax.config.update("jax_enable_x64", False)   # float32 only; never toggled here
import jax.numpy as jnp                 # noqa: E402
import math                             # noqa: E402

import grok                             # frozen, read-only  # noqa: E402

OUTDIR = HERE / "results" / "diag10"
RESULT = OUTDIR / "diag10_attention_structure.json"
if RESULT.exists():
    sys.exit(f"ABORT: result file already exists (exclusive namespace): {RESULT}")

ART = REPO / "artifacts" / "part4" / "formal-decoder-02"
SNAPS = [(s, c) for s in range(5) for c in ("init", "memorization", "final")]
LOGIT_ATOL, LOGIT_RTOL = 1e-5, 1e-6     # baseline match vs frozen forward
ACC_TOL = 1e-6                          # vs exact corresponding log row
SAT = 0.99
ENT_LO, SAT_HI, DIS_LO, DIS_HI = 0.0, 0.5, 0.5, 0.05   # H2a/H2c thresholds


def expected_log_row(seed: int, ckpt: str) -> dict:
    log = json.loads((ART / f"main-s{seed}" / "train-log.json").read_text())
    if ckpt == "init":
        return next(r for r in log if r["step"] == 0)
    if ckpt == "memorization":
        mem = json.loads(
            (ART / f"main-s{seed}" / "completion-receipt.json").read_text()
        )["memorization_step"]
        return next(r for r in log if r["step"] == mem)
    return log[-1]


def diag_forward(params: dict, tokens: jax.Array):
    """Same frozen ops as grok.forward; returns (logits[B,3,97], w[B,H,T,T])."""
    B, T = tokens.shape
    x = params["emb"][tokens] + params["pos"][:T]
    q = (x @ params["wq"] + params["bq"]).reshape(B, T, grok.HEADS, grok.WIDTH // grok.HEADS)
    k = (x @ params["wk"] + params["bk"]).reshape(B, T, grok.HEADS, grok.WIDTH // grok.HEADS)
    v = (x @ params["wv"] + params["bv"]).reshape(B, T, grok.HEADS, grok.WIDTH // grok.HEADS)
    d = grok.WIDTH // grok.HEADS
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d)
    tri = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    scores = jnp.where(tri, scores, -jnp.inf)
    w = jax.nn.softmax(scores, axis=-1)                       # [B,H,T,T]
    att = jnp.einsum("bhts,bshd->bthd", w, v).reshape(B, T, grok.WIDTH)
    x = x + att @ params["wo"] + params["bo"]
    h = jax.nn.gelu(x @ params["mlp1"]["w"] + params["mlp1"]["b"])
    x = x + h @ params["mlp2"]["w"] + params["mlp2"]["b"]
    return x @ params["head"]["w"] + params["head"]["b"], w


def entropy(w: np.ndarray) -> np.ndarray:
    """0*log0-safe row entropies; w any shape, last axis is the softmax axis."""
    pos = w > 0
    return -np.sum(np.where(pos, w * np.log(np.where(pos, w, 1.0)), 0.0), axis=-1)


def attn_metrics(w: np.ndarray) -> dict:
    """w: [B,H,T,T] float32 numpy. Answer row = index 2. All descriptive
    structure plus the pre-declared aggregate criteria."""
    a = w[:, :, 2, :].astype(np.float64)          # [B,H,3] answer row
    H = entropy(a)                                # [B,H]
    mx = a.max(axis=-1)                           # [B,H]
    per_head = {
        "entropy_mean": H.mean(axis=0).tolist(),
        "entropy_p10": np.percentile(H, 10, axis=0).tolist(),
        "entropy_p50": np.percentile(H, 50, axis=0).tolist(),
        "entropy_p90": np.percentile(H, 90, axis=0).tolist(),
        "sat_frac": (mx > SAT).mean(axis=0).tolist(),
        "maxw_mean": mx.mean(axis=0).tolist(),
    }
    row1 = w[:, :, 1, :2].astype(np.float64)      # rows 0 forced [1,0,0]
    sc = w[:, :, 2, :].astype(np.float64)
    return {
        "per_head": per_head,
        "head_mean_entropy": float(H.mean()),
        "head_mean_sat_frac": float((mx > SAT).mean()),
        "descriptive": {
            "input_dep_std_max": float(np.max(a.std(axis=0))),
            "row1_entropy_mean": float(entropy(row1).mean()),
            "answer_weight_min": float(sc.min()), "answer_weight_max": float(sc.max()),
            "answer_weight_range": float(sc.max() - sc.min()),
        },
    }


def main() -> None:
    data = grok.build_dataset()
    split = grok.fixed_split(0.30)
    te_tok = jnp.asarray(data["inputs"][split["test"]])
    te_tgt = jnp.asarray(data["targets"][split["test"]])
    all_tok = jnp.asarray(data["inputs"])                 # all 9409 pairs
    test_mask = np.zeros((grok.P, grok.P), dtype=bool)
    test_mask.ravel()[np.asarray(split["test"])] = True

    # fixed symmetric unseen set S: both orientations in test, off-diagonal
    S = np.array([(a, b) for a in range(grok.P) for b in range(grok.P)
                  if a != b and test_mask[a, b] and test_mask[b, a]],
                 dtype=np.int64)
    s_hash = hashlib.sha256(
        np.sort(S[:, 0] * grok.P + S[:, 1]).tobytes()).hexdigest()

    def snapshot_metrics(params: dict, tag: str) -> dict:
        # baseline-match gate: custom logits vs frozen forward (explicit tol)
        logits_d, w_all = diag_forward(params, te_tok)
        logits_f = grok.forward(params, te_tok)
        diff = np.abs(np.asarray(logits_d) - np.asarray(logits_f))
        if not np.all(diff <= LOGIT_ATOL + LOGIT_RTOL * np.abs(np.asarray(logits_f))):
            fail = {"harness_defect": "baseline_logits_match", "snapshot": tag,
                    "max_abs_diff": float(diff.max())}
            (OUTDIR).mkdir(parents=True, exist_ok=True)
            fp = OUTDIR / "attempt01-baseline-match-failure.json"
            if not fp.exists():
                fp.write_text(json.dumps(fail, indent=1) + "\n")
            sys.exit(f"ABORT: baseline logits mismatch at {tag}: {diff.max()}")
        acc = float((np.asarray(logits_d)[:, 2, :].argmax(-1)
                     == np.asarray(te_tgt)).mean())
        # full-grid logits + attention (all 9409 pairs, deterministic)
        logits_all, w_full = diag_forward(params, all_tok)
        logits_all = np.asarray(logits_all)
        pred = logits_all[:, 2, :].argmax(-1).reshape(grok.P, grok.P)
        dis = float(np.mean(pred[S[:, 0], S[:, 1]] != pred[S[:, 1], S[:, 0]]))
        # mean |logit(a,b) - logit(b,a)| over S on the argmax class of (a,b)
        ii = S[:, 0] * grok.P + S[:, 1]
        jj = S[:, 1] * grok.P + S[:, 0]
        la, lb = logits_all[:, 2, :][ii], logits_all[:, 2, :][jj]
        mean_logit_diff = float(np.mean(np.abs(la - lb).max(axis=-1)))
        # single-variable descriptive intervention: pos -> 0 (transient copy)
        p0 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()}
        p0["pos"] = jnp.zeros_like(params["pos"])
        logits0, w0 = diag_forward(p0, all_tok)
        liveness = bool(np.any(np.asarray(logits0) != logits_all))
        return {"test_acc_recomputed": acc,
                "attention": attn_metrics(np.asarray(w_full)),
                "order_sensitivity": {"set_size": int(S.shape[0]),
                                      "set_sha256_flat_sorted": s_hash,
                                      "symmetry_disagreement": dis,
                                      "mean_max_class_abs_logit_diff": mean_logit_diff},
                "pos0_descriptive": {"attention": attn_metrics(np.asarray(w0)),
                                     "harness_liveness_logits_changed": liveness}}

    results, harness_ok = {}, True
    for seed, ckpt in SNAPS:
        tag = f"main-s{seed}/{ckpt}"
        params = grok.load_params(ART / f"main-s{seed}" / f"checkpoint-{ckpt}.npz")
        m = snapshot_metrics(params, tag)
        row = expected_log_row(seed, ckpt)
        m["logged_row_step"] = row["step"]
        m["logged_test_acc"] = row["test_acc"]
        m["acc_tie_rel"] = abs(m["test_acc_recomputed"] - row["test_acc"]) \
            / max(row["test_acc"], 1e-30)
        if m["acc_tie_rel"] > ACC_TOL:
            fail = {"harness_defect": "acc_tie_in", "snapshot": tag,
                    "acc_tie_rel": m["acc_tie_rel"], "logged_row": row}
            OUTDIR.mkdir(parents=True, exist_ok=True)
            fp = OUTDIR / "attempt01-acc-tie-failure.json"
            if not fp.exists():
                fp.write_text(json.dumps(fail, indent=1) + "\n")
            sys.exit(f"ABORT: acc tie-in failed at {tag}")
        harness_ok &= m["pos0_descriptive"]["harness_liveness_logits_changed"]
        results[tag] = m
    # wd0-s0 final descriptor (excluded from all pass/fail logic)
    params = grok.load_params(ART / "wd0-s0" / "checkpoint-final.npz")
    m = snapshot_metrics(params, "wd0-s0/final")
    m["logged_row_step"] = None
    m["logged_test_acc"] = None          # descriptor only, no tie-in criterion
    harness_ok &= m["pos0_descriptive"]["harness_liveness_logits_changed"]
    results["wd0-s0/final"] = m

    # ---- decision (engineering, pre-declared) ---------------------------
    main = {f"main-s{s}/{c}": results[f"main-s{s}/{c}"]
            for s in range(5) for c in ("init", "memorization", "final")}
    grokked = [f"main-s{s}" for s in range(1, 5)]

    def env(metric, ckpt):
        vals = [main[f"{g}/{ckpt}"]["attention"][metric] for g in grokked]
        return min(vals), max(vals)

    s0_fin = main["main-s0/final"]["attention"]
    s0_mem = main["main-s0/memorization"]["attention"]
    e_lo_f, e_hi_f = env("head_mean_entropy", "final")
    sf_lo_f, sf_hi_f = env("head_mean_sat_frac", "final")
    h2a_fin = (s0_fin["head_mean_entropy"] < e_lo_f
               and s0_fin["head_mean_sat_frac"] > SAT_HI
               and sf_hi_f < 0.1)
    e_lo_m, _ = env("head_mean_entropy", "memorization")
    sf_hi_m = env("head_mean_sat_frac", "memorization")[1]
    dir_mem = (s0_mem["head_mean_entropy"] < e_lo_m
               and s0_mem["head_mean_sat_frac"] > sf_hi_m) if h2a_fin else \
              (main["main-s0/final"]["order_sensitivity"]["symmetry_disagreement"]
               > main["main-s1/final"]["order_sensitivity"]["symmetry_disagreement"])
    h2a = bool(h2a_fin and dir_mem)

    dis_fin = {g: main[f"{g}/final"]["order_sensitivity"]["symmetry_disagreement"]
               for g in grokked}
    dis_mem = {g: main[f"{g}/memorization"]["order_sensitivity"]["symmetry_disagreement"]
               for g in grokked}
    s0_dis_f = main["main-s0/final"]["order_sensitivity"]["symmetry_disagreement"]
    s0_dis_m = main["main-s0/memorization"]["order_sensitivity"]["symmetry_disagreement"]
    h2c_fin = s0_dis_f > DIS_LO and max(dis_fin.values()) < DIS_HI
    h2c = bool(h2c_fin and s0_dis_m > max(dis_mem.values()))

    inside = True
    for ckpt in ("init", "memorization", "final"):
        for met in ("head_mean_entropy", "head_mean_sat_frac"):
            lo, hi = env(met, ckpt)
            v = main[f"main-s0/{ckpt}"]["attention"][met]
            inside &= (lo <= v <= hi)
        if ckpt == "final":
            lo_d = min(main[f"{g}/final"]["order_sensitivity"]["symmetry_disagreement"]
                       for g in grokked)
            hi_d = max(main[f"{g}/final"]["order_sensitivity"]["symmetry_disagreement"]
                       for g in grokked)
            inside &= (lo_d <= s0_dis_f <= hi_d)
        if ckpt == "memorization":
            lo_dm = min(main[f"{g}/memorization"]["order_sensitivity"]
                        ["symmetry_disagreement"] for g in grokked)
            hi_dm = max(main[f"{g}/memorization"]["order_sensitivity"]
                        ["symmetry_disagreement"] for g in grokked)
            inside &= (lo_dm <= s0_dis_m <= hi_dm)

    if h2a or h2c:
        verdict = "supported_engineering_level"
    elif inside:
        verdict = "falsified_within_envelope"
    else:
        verdict = "inconclusive_outside_envelope"

    out = {
        "diagnostic": "diag10_attention_structure", "version": 2,
        "hypothesis": "H2 (narrowed: saturation + order sensitivity)",
        "grok_py_sha256": ACTUAL_SHA,
        "x64_flag": False, "init_params_called": False,
        "baseline_match": {"logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL,
                           "acc_tol_rel": ACC_TOL},
        "harness_liveness_all_snapshots": bool(harness_ok),
        "thresholds": {"sat": SAT, "h2a_entropy_requires_strict_min": True,
                       "h2a_sat_frac": [SAT_HI, "grokked < 0.1"],
                       "h2c_disagreement": [DIS_LO, "grokked < 0.05"]},
        "causality_caveat": "static-snapshot conditional associations; no "
        "necessary-cause claim; entropy alone never proves cause; position "
        "sensitivity alone is not proof of a shortcut",
        "verdict_engineering": verdict,
        "signatures": {"h2a_saturation": h2a, "h2c_order_sensitivity": h2c,
                       "seed0_within_envelope": bool(inside)},
        "snapshots": results}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if RESULT.exists():
        sys.exit("ABORT: result file appeared concurrently; nothing overwritten")
    fd = os.open(RESULT, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    print(f"diag10: harness={'PASS' if harness_ok else 'FAIL'} H2={verdict} "
          f"-> {RESULT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
