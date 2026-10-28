"""Follow-up: float32 log-softmax CE gradient cancellation probe (POST-HOC).
Plan: FOLLOWUP_PLAN.json here. NEW hypothesis declared AFTER diag09/diag10
outcomes; NOT pre-registered with them. Single-shot; writes only
results/followup_ce_cancellation.json (O_EXCL). No training/optimizer/
frozen-source edits; init_params never called. Scope: POINTWISE numeric
mechanism on saved values ONLY; stable form is NOT a fix for past grokking.
Forms (mathematically identical; float64 equivalence is the harness gate):
  original: nll = -log_softmax(z)[y]            (frozen grok.loss_acc formula)
  stable:   s = logsumexp(z masked at y) - z_y;  nll = softplus(s)
            = log(1 + sum_{j!=y} exp(z_j - z_y))   -> never forms p_true - 1.
Support (engineering, NOT course gate): float32 stable gradient closer to the
float64 original reference by >= 10x in relL2 AND stable relL2 < 1e-3 =>
cancellation is a LOCAL cause of the measured discrepancy at that snapshot.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
GROK_SHA = "d0501c3baf88a1194665123e47abc0a5b0035d0b203a56ca0dcc92853bc1d447"
if hashlib.sha256((REPO / "grok.py").read_bytes()).hexdigest() != GROK_SHA:
    sys.exit("ABORT: grok.py hash mismatch")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)  # float32 regime first
import grok                                 # frozen, read-only

OUT = HERE / "results" / "followup_ce_cancellation.json"
if OUT.exists():
    sys.exit(f"ABORT: exclusive result path exists: {OUT}")
ART = REPO / "artifacts" / "part4" / "formal-decoder-02"
SNAPS = [f"main-s{i}" for i in range(5)]        # ALL five finals, fixed set
GATE_ATOL, GATE_RTOL = 1e-9, 1e-7               # f64 form-equivalence gate
RATIO_MIN, STABLE_MAX, ACC_TOL = 10.0, 1e-3, 1e-6

def nll_orig(z, y):                             # frozen loss_acc formula
    lp = jax.nn.log_softmax(z)
    return -lp[jnp.arange(y.shape[0]), y]

def nll_stable(z, y):                           # masked logsumexp + softplus
    masked = z.at[jnp.arange(y.shape[0]), y].set(-jnp.inf)
    s = jax.nn.logsumexp(masked, axis=-1) - z[jnp.arange(y.shape[0]), y]
    return jax.nn.softplus(s)                   # = logaddexp(0, s)

def vg(form, params, tok, tgt):
    f = {"orig": nll_orig, "stable": nll_stable}[form]
    return jax.value_and_grad(
        lambda p: f(grok.forward(p, tok), tgt).mean())(params)

def rel_stats(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel()
    d = float(np.linalg.norm(a - b))
    return {"relL2": d / float(np.linalg.norm(b)), "abs": d,
            "norm_ref": float(np.linalg.norm(b))}
def gate_ok(x, y):
    return bool(np.all(np.abs(x - y) <= GATE_ATOL + GATE_RTOL * np.abs(y)))

def main():
    data = grok.build_dataset()
    sp = grok.fixed_split(0.30)
    tok = jnp.asarray(data["inputs"][sp["train"]])
    tgt = jnp.asarray(data["targets"][sp["train"]])
    loaded, imm0 = {}, {}
    for run in SNAPS:                           # float32, never re-init
        p = grok.load_params(ART / run / "checkpoint-final.npz")
        assert all(np.asarray(x).dtype == np.float32
                   for x in jax.tree_util.tree_leaves(p))
        loaded[run] = p
        imm0[run] = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(p)]
    # analytic fixtures (synthetic logits; float32 phase)
    eq = np.zeros((1, grok.P), np.float32)      # equal logits -> loss ln 97
    cm, im = eq.copy(), eq.copy()
    cm[0, 0], im[0, 0] = 1000.0, -1000.0        # correct / incorrect margins
    fix32 = {}
    for name, z in (("equal", eq), ("correct+1000", cm), ("incorrect-1000", im)):
        y, zj = jnp.asarray([0]), jnp.asarray(z)
        go = np.asarray(jax.grad(lambda w: nll_orig(w, y).mean())(zj))
        gs = np.asarray(jax.grad(lambda w: nll_stable(w, y).mean())(zj))
        fix32[name] = {"loss_orig": float(nll_orig(zj, y)[0]),
                       "loss_stable": float(nll_stable(zj, y)[0]),
                       "grad_orig_nonzero": bool(np.any(go != 0)),
                       "grad_stable_nonzero": bool(np.any(gs != 0)),
                       "grad_orig_target": float(go[0, 0]),
                       "grad_stable_target": float(gs[0, 0])}
    assert fix32["equal"]["grad_orig_nonzero"] and \
        fix32["incorrect-1000"]["grad_orig_nonzero"], "vacuous fixture"
    assert np.isfinite([[fix32[n][k] for k in ("loss_orig", "loss_stable")]
                        for n in fix32]).all()
    # float32 passes: original and stable on all five finals (x64 OFF)
    f32 = {}
    for run in SNAPS:
        _, go = vg("orig", loaded[run], tok, tgt)
        _, gs = vg("stable", loaded[run], tok, tgt)
        L32o = float(nll_orig(grok.forward(loaded[run], tok), tgt).mean())
        L32s = float(nll_stable(grok.forward(loaded[run], tok), tgt).mean())
        acc = float(grok.loss_acc(loaded[run], tok, tgt)[1])
        logged = json.loads(
            (ART / run / "train-log.json").read_text())[-1]["train_acc"]
        f32[run] = {"g_orig": [np.asarray(x) for x in
                               jax.tree_util.tree_leaves(go)],
                    "g_stable": [np.asarray(x) for x in
                                 jax.tree_util.tree_leaves(gs)],
                    "loss32_orig": L32o, "loss32_stable": L32s, "acc": acc,
                    "acc_tie_rel": abs(acc - logged) / logged}
    assert jax.config.jax_enable_x64 is False
    # single x64 toggle, then float64 reference passes
    jax.config.update("jax_enable_x64", True)
    assert jnp.asarray(1.0).dtype == jnp.float64
    assert jnp.asarray(np.float32(1.0)).dtype == jnp.float32
    fix64, harness_ok = {}, True
    for name, z in (("equal", eq), ("correct+1000", cm), ("incorrect-1000", im)):
        y, zj = jnp.asarray([0]), jnp.asarray(z.astype(np.float64))
        lo, ls = np.asarray(nll_orig(zj, y)), np.asarray(nll_stable(zj, y))
        go = np.asarray(jax.grad(lambda w: nll_orig(w, y).mean())(zj))
        gs = np.asarray(jax.grad(lambda w: nll_stable(w, y).mean())(zj))
        ok = gate_ok(lo, ls) and gate_ok(go, gs)
        harness_ok &= ok
        fix64[name] = {"loss_orig": float(lo[0]), "loss_stable": float(ls[0]),
                       "forms_agree_within_gate": ok,
                       "loss_equals_ln97": (bool(abs(lo[0] - np.log(97)) < 1e-9)
                                            if name == "equal" else None)}
    leaves = lambda g: [np.asarray(x) for x in jax.tree_util.tree_leaves(g)]
    flat = lambda ls: np.concatenate([x.ravel() for x in ls])
    snaps = {}
    for run in SNAPS:
        p64 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64),
                                     loaded[run])
        _, go64 = vg("orig", p64, tok, tgt)
        _, gs64 = vg("stable", p64, tok, tgt)
        l64o = float(nll_orig(grok.forward(p64, tok), tgt).mean())
        l64s = float(nll_stable(grok.forward(p64, tok), tgt).mean())
        go64f, gs64f = flat(leaves(go64)), flat(leaves(gs64))
        ok = gate_ok(np.asarray(l64o), np.asarray(l64s)) and gate_ok(go64f, gs64f)
        harness_ok &= ok
        r_o = rel_stats(flat(f32[run]["g_orig"]), go64f)
        r_s = rel_stats(flat(f32[run]["g_stable"]), go64f)
        ratio = r_o["relL2"] / max(r_s["relL2"], 1e-30)
        snaps[run] = {
            "loss64_orig": l64o, "loss64_stable": l64s,
            "f64_forms_agree_within_gate": ok,
            "loss32_orig_rel_vs_64": abs(f32[run]["loss32_orig"] - l64o) / abs(l64o),
            "loss32_stable_rel_vs_64": abs(f32[run]["loss32_stable"] - l64o) / abs(l64o),
            "grad_f32_orig_vs_f64_orig": r_o,
            "grad_f32_stable_vs_f64_orig": r_s,
            "improvement_ratio": float(ratio),
            "criterion_met": bool(ratio >= RATIO_MIN and r_s["relL2"] < STABLE_MAX),
            "acc_tie_rel": f32[run]["acc_tie_rel"],
            "grad_norm64": r_o["norm_ref"]}
    supported = snaps["main-s0"]["criterion_met"]
    acc_ok = all(s["acc_tie_rel"] <= ACC_TOL for s in snaps.values())
    imm_ok = all(np.array_equal(a, b) for run in SNAPS for a, b in
                 zip(imm0[run], jax.tree_util.tree_leaves(loaded[run])))
    verdict = ("cancellation_supported_locally" if supported and harness_ok
               else "harness_invalid" if not harness_ok
               else "unresolved_numeric_mechanism")
    out = {"followup": "ce_cancellation", "class": "POST_HOC_AFTER_OUTCOMES",
           "date": "2026-10-28", "grok_py_sha256": GROK_SHA,
           "x64_flag_f32_phase": False, "x64_flag_f64_phase": True,
           "init_params_called": False,
           "harness_gate": {"atol": GATE_ATOL, "rtol": GATE_RTOL,
                            "ok": bool(harness_ok)},
           "fixtures_float32": fix32, "fixtures_float64": fix64,
           "criterion": {"ratio_min": RATIO_MIN, "stable_relL2_max": STABLE_MAX},
           "verdict_engineering": verdict,
           "scope": "pointwise numeric mechanism only; NOT historical cause; "
                    "stable form is not a fix for past grokking",
           "final_checks": {"acc_unchanged": bool(acc_ok),
                            "arrays_immutable": bool(imm_ok)},
           "snapshots": snaps}
    HERE.joinpath("results").mkdir(exist_ok=True)
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    print(f"ce-cancellation: harness={'PASS' if harness_ok else 'FAIL'} "
          f"verdict={verdict} -> {OUT.relative_to(HERE)}")

if __name__ == "__main__":
    main()
