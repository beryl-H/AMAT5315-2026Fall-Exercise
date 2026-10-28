"""diag02: Finite-difference gradient check of grok.loss_acc / grok.forward.

Central finite differences (float64) on ~40 random coordinates of a small
subset of the training data, compared with jax.value_and_grad of the exact
frozen loss. Pass: relative agreement within 2e-5 on every probed coordinate
(float64 FD truncation vs exact autodiff).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import grok  # frozen source, read-only

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

WIDTH, P = 16, 97


def to_f64(params):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)


def flatten(params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    return leaves, treedef


def main():
    rng = np.random.default_rng(97)
    params = to_f64(grok.init_params(jax.random.PRNGKey(0)))

    data = grok.build_dataset()
    split = grok.fixed_split(0.30)
    idx = split["train"][:64]                     # small subset
    tok = jnp.asarray(data["inputs"][idx])
    tgt = jnp.asarray(data["targets"][idx])

    def loss_fn(p):
        return grok.loss_acc(p, tok, tgt)[0]

    loss, grads = jax.value_and_grad(loss_fn)(params)
    g_leaves, _ = flatten(grads)

    h = 1e-5
    probed, worst = [], 0.0
    atol, rtol = 1e-6, 1e-3   # standard FD check: |fd-ag| <= atol + rtol*|ag|
    n_fail = 0
    for li in range(len(g_leaves)):
        g = np.asarray(g_leaves[li])
        flat_idx = rng.choice(g.size, size=min(8, g.size), replace=False)
        p_leaves, treedef = flatten(params)
        for fi in flat_idx:
            orig = np.asarray(p_leaves[li]).copy()
            def bumped(delta):
                arr = orig.copy()
                arr.flat[fi] = delta
                new_leaves = list(p_leaves)
                new_leaves[li] = jnp.asarray(arr)
                return loss_fn(jax.tree_util.tree_unflatten(treedef, new_leaves))
            fd = float((bumped(orig.flat[fi] + h) - bumped(orig.flat[fi] - h)) / (2 * h))
            ag = float(g.flat[fi])
            rel = abs(fd - ag) / max(1e-8, abs(fd), abs(ag))
            ok = abs(fd - ag) <= atol + rtol * abs(ag)
            probed.append({"leaf": li, "index": int(fi), "fd": fd,
                           "autodiff": ag, "rel_err": rel, "ok": ok})
            if not ok:
                n_fail += 1
            worst = max(worst, abs(fd - ag))
    result = {"diagnostic": "diag02_grad_finite_diff",
              "n_probed": len(probed), "h": h,
              "criterion": "|fd-ag| <= 1e-6 + 1e-3*|ag|",
              "n_failed": n_fail,
              "worst_abs_err": worst,
              "worst_offenders": sorted(probed, key=lambda p: -p["rel_err"])[:5],
              "pass": bool(n_fail == 0)}
    (OUT / "diag02_grad_finite_diff.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
