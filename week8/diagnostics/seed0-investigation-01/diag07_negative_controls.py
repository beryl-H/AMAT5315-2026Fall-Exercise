"""diag07: Negative controls — deliberately introduce known defects and
confirm the diagnostic checks FAIL (they are not vacuously passing).

NC1: AdamW cross-check with a wrong beta2 (0.999 instead of 0.98) must fail.
NC2: AdamW cross-check with coupled L2 weight decay (wd applied inside the
     gradient) instead of decoupled AdamW decay must fail.
NC3: Gradient check with a sign-flipped gradient must fail.
No frozen source is modified; defective copies live only in this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
jax.config.update("jax_enable_x64", True)   # float64 loss for a clean FD control
import jax.numpy as jnp

import grok  # frozen source, read-only
import diag01_adamw_crosscheck as d1

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def broken_adamw_wrong_beta2(p, g, m, v, lr, step, wd):
    return d1.reference_adamw(p, g, m, v, lr, step, wd)  # baseline


def run_crosscheck(reference_fn, lr, wd, structured):
    """Re-implementation of diag01's comparison with a pluggable reference."""
    rng = np.random.default_rng(20261028)
    shapes = [(64, 64), (64, 256), (256, 64), (98, 64), (64,)]
    params = {f"x{i}": jnp.asarray(rng.normal(size=s), dtype=jnp.float32)
              for i, s in enumerate(shapes)}
    state = grok.grok_adamw_init(params)
    ref_p = {k: np.asarray(v, np.float64) for k, v in params.items()}
    ref_m = {k: np.zeros_like(v) for k, v in ref_p.items()}
    ref_v = {k: np.zeros_like(v) for k, v in ref_p.items()}
    max_rel = 0.0
    for step in range(1, 201):
        gkey = jax.random.PRNGKey(step)
        if structured:
            grads = {k: (0.1 * np.sin(step / 7.0) + 0.05) * np.ones_like(np.asarray(v))
                     for k, v in params.items()}
        else:
            grads = {k: jax.random.normal(jax.random.fold_in(gkey, i), v.shape)
                     for i, (k, v) in enumerate(params.items())}
        params, state = grok.grok_adamw_step(params, grads, state, lr=lr,
                                             step=step, weight_decay=wd)
        for k in ref_p:
            ref_p[k], ref_m[k], ref_v[k] = reference_fn(
                ref_p[k], np.asarray(grads[k], np.float64), ref_m[k], ref_v[k],
                lr, step, wd)
        for k in ref_p:
            d = float(np.max(np.abs(np.asarray(params[k], np.float64) - ref_p[k])))
            max_rel = max(max_rel, d / max(float(np.max(np.abs(ref_p[k]))), 1e-3))
    return max_rel


def main():
    b1, b2, eps = 0.9, 0.98, 1e-8

    def wrong_beta2(p, g, m, v, lr, step, wd):
        m = b1 * m + (1 - b1) * g
        v = 0.999 * v + 0.001 * np.square(g)          # DEFECT: wrong beta2
        return (p - lr * wd * p
                - lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + eps),
                m, v)

    def coupled_wd(p, g, m, v, lr, step, wd):
        g = g + wd * p                                 # DEFECT: coupled L2, not decoupled
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * np.square(g)
        return (p - lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps),
                m, v)

    nc = {}
    nc["NC1_wrong_beta2"] = {
        "max_scale_relative_diff": run_crosscheck(wrong_beta2, 3e-3, 1.0, False),
        "expected": "FAIL (diff >> 1e-5)"}
    nc["NC2_coupled_weight_decay"] = {
        "max_scale_relative_diff": run_crosscheck(coupled_wd, 3e-3, 1.0, False),
        "expected": "FAIL (diff >> 1e-5)"}

    # NC3: sign-flipped gradient must fail the finite-difference check
    key = jax.random.PRNGKey(0)
    params = grok.init_params(key)   # float64 under x64
    data = grok.build_dataset()
    idx = grok.fixed_split(0.30)["train"][:64]
    tok = jnp.asarray(data["inputs"][idx])
    tgt = jnp.asarray(data["targets"][idx])

    def loss_fn(p):
        return grok.loss_acc(p, tok, tgt)[0]

    loss, grads = jax.value_and_grad(loss_fn)(params)
    flipped = jax.tree_util.tree_map(lambda x: -x, grads)   # DEFECT: sign flip
    # probe the coordinate with the LARGEST gradient magnitude
    g_leaves = jax.tree_util.tree_leaves(grads)
    p_leaves, treedef = jax.tree_util.tree_flatten(params)
    best = max(((li, int(np.argmax(np.abs(np.asarray(g))))), float(np.max(np.abs(np.asarray(g)))))
               for li, g in enumerate(g_leaves))
    li, fi = best[0]
    ag = -float(np.asarray(g_leaves[li]).flat[fi])          # sign-flipped value
    orig = np.asarray(p_leaves[li]).copy()
    h = 1e-5
    arr_p = orig.copy(); arr_p.flat[fi] += h
    arr_m = orig.copy(); arr_m.flat[fi] -= h
    nl = list(p_leaves)
    nl[li] = jnp.asarray(arr_p); lp = loss_fn(jax.tree_util.tree_unflatten(treedef, nl))
    nl[li] = jnp.asarray(arr_m); lm = loss_fn(jax.tree_util.tree_unflatten(treedef, nl))
    fd = float((lp - lm) / (2 * h))
    nc["NC3_sign_flipped_gradient"] = {
        "probed_leaf": li, "probed_index": fi, "fd": fd, "defective_grad": ag,
        "abs_disagreement": abs(fd - ag),
        "expected": "FAIL (|fd-ag| >> atol)"}

    results = {
        "diagnostic": "diag07_negative_controls",
        "controls": nc,
        "all_detected": bool(
            nc["NC1_wrong_beta2"]["max_scale_relative_diff"] > 1e-5
            and nc["NC2_coupled_weight_decay"]["max_scale_relative_diff"] > 1e-5
            and nc["NC3_sign_flipped_gradient"]["abs_disagreement"] > 1e-6)}
    (OUT / "diag07_negative_controls.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if results["all_detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
