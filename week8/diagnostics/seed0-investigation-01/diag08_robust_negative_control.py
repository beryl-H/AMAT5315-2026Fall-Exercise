"""diag08: Robust float64 finite-difference sign-flip negative control.

Addresses the Codex review of diag07 NC3: in the first (float32) version the
FD returned exactly 0.0 (loss quantization), and the x64 redo used a single
h. This control demonstrates genuine sensitivity:
  - probe a coordinate whose analytic gradient magnitude exceeds 1e-3
    (actual, unmodified gradient);
  - central finite differences in float64 for h in {1e-3, 1e-4, 1e-5};
  - the FD must match the UNMODIFIED analytic gradient at every h;
  - the FD must REJECT the sign-flipped analytic gradient at every h;
  - all FD values must be nonzero.
Frozen sources are read-only; this is a diagnostics-only check.
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

GRAD_MIN = 1e-3
H_GRID = [1e-3, 1e-4, 1e-5]


def main():
    key = jax.random.PRNGKey(0)
    params = grok.init_params(key)                    # float64 under x64
    data = grok.build_dataset()
    idx = grok.fixed_split(0.30)["train"][:64]
    tok = jnp.asarray(data["inputs"][idx])
    tgt = jnp.asarray(data["targets"][idx])

    def loss_fn(p):
        return grok.loss_acc(p, tok, tgt)[0]

    _, grads = jax.value_and_grad(loss_fn)(params)
    g_leaves = jax.tree_util.tree_leaves(grads)
    p_leaves, treedef = jax.tree_util.tree_flatten(params)

    # largest-magnitude analytic gradient coordinate (actual, unmodified)
    candidates = [(li, int(np.argmax(np.abs(np.asarray(g)))),
                   float(np.max(np.abs(np.asarray(g)))))
                  for li, g in enumerate(g_leaves)]
    li, fi, gmag = max(candidates, key=lambda c: c[2])
    assert gmag > GRAD_MIN, f"largest gradient {gmag} <= {GRAD_MIN}"
    g_true = float(np.asarray(g_leaves[li]).flat[fi])

    orig = np.asarray(p_leaves[li]).copy()

    def fd_at(h, delta_sign):
        arr_p = orig.copy(); arr_p.flat[fi] += delta_sign * h
        arr_m = orig.copy(); arr_m.flat[fi] -= delta_sign * h
        nl = list(p_leaves)
        nl[li] = jnp.asarray(arr_p)
        lp = loss_fn(jax.tree_util.tree_unflatten(treedef, nl))
        nl[li] = jnp.asarray(arr_m)
        lm = loss_fn(jax.tree_util.tree_unflatten(treedef, nl))
        return float((lp - lm) / (2 * h))

    per_h = []
    all_ok = True
    for h in H_GRID:
        fd_true = fd_at(h, +1)   # FD of the actual loss = true derivative
        # sensitivity checks:
        ok_true = abs(fd_true - g_true) <= 1e-6 + 1e-3 * abs(g_true)
        rejected = abs(fd_true - (-g_true)) > 1e-6 + 1e-3 * abs(g_true)
        nonzero = fd_true != 0.0
        entry = {"h": h, "fd_unflipped": fd_true,
                 "analytic_grad": g_true, "sign_flipped_grad": -g_true,
                 "matches_unmodified_gradient": bool(ok_true),
                 "rejects_sign_flipped_gradient": bool(rejected and nonzero),
                 "fd_nonzero": bool(nonzero)}
        per_h.append(entry)
        all_ok &= ok_true and rejected and nonzero

    result = {"diagnostic": "diag08_robust_negative_control",
              "probed": {"leaf": li, "index": fi, "analytic_grad": g_true,
                         "grad_magnitude": abs(g_true), "grad_min_required": GRAD_MIN},
              "h_grid": H_GRID, "per_h": per_h,
              "dtype": "float64 (jax_enable_x64)",
              "pass": bool(all_ok)}
    (OUT / "diag08_robust_negative_control.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
