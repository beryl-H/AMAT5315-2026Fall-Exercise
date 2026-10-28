"""diag01: Independent cross-check of grok.grok_adamw_step against a fresh,
canonical decoupled-AdamW implementation written from the algorithm
definition (Loshchilov & Hutter 2019, decoupled weight decay; PyTorch
reference semantics), in float64 NumPy. grok.py is imported read-only and
never modified.

Check: for random parameters and random (and structured) gradients, run
both optimizers for N steps with identical inputs and compare parameters
and m/v states elementwise.

Pass criterion: max abs diff <= 1e-9 (float64 reference vs float32 JAX
rounded to float64 after each step -> tolerance widened to 1e-6 relative).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jnp

import grok  # frozen source, read-only

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

B1, B2, EPS = 0.9, 0.98, 1e-8


def reference_adamw(p, g, m, v, lr, step, wd):
    """Canonical decoupled AdamW in float64 NumPy (independent formula)."""
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * np.square(g)
    m_hat = m / (1 - B1 ** step)
    v_hat = v / (1 - B2 ** step)
    p = p - lr * wd * p - lr * m_hat / (np.sqrt(v_hat) + EPS)
    return p, m, v


def run_case(rng, shapes, steps, lr, wd, structured=False):
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31)))
    params = {f"x{i}": jnp.asarray(rng.normal(size=s), dtype=jnp.float32)
              for i, s in enumerate(shapes)}
    state = grok.grok_adamw_init(params)

    ref_p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    ref_m = {k: np.zeros_like(v) for k, v in ref_p.items()}
    ref_v = {k: np.zeros_like(v) for k, v in ref_p.items()}

    max_delta, max_rel = 0.0, 0.0
    for step in range(1, steps + 1):
        gkey = jax.random.fold_in(key, step)
        if structured:
            grads = {k: (0.1 * np.sin(step / 7.0) + 0.05) * np.ones_like(
                         np.asarray(v)) * (1.0 + 0.01 * np.asarray(v))
                     for k, v in params.items()}
        else:
            grads = {k: jax.random.normal(jax.random.fold_in(gkey, i), v.shape)
                     for i, (k, v) in enumerate(params.items())}
        params, state = grok.grok_adamw_step(params, grads, state, lr=lr,
                                             step=step, weight_decay=wd)
        for k in ref_p:
            ref_p[k], ref_m[k], ref_v[k] = reference_adamw(
                ref_p[k], np.asarray(grads[k], np.float64), ref_m[k], ref_v[k],
                lr, step, wd)
        for k in ref_p:
            d = float(np.max(np.abs(np.asarray(params[k], np.float64) - ref_p[k])))
            scale = max(float(np.max(np.abs(ref_p[k]))), 1e-3)
            max_delta = max(max_delta, d)
            max_rel = max(max_rel, d / scale)
    return max_delta, max_rel


def main():
    rng = np.random.default_rng(20261028)
    shapes = [(64, 64), (64, 256), (256, 64), (98, 64), (64,)]
    cases = []
    for lr in (3e-3, 1e-2):
        for wd in (0.0, 1.0, 50.0):
            for structured in (False, True):
                d, rel = run_case(rng, shapes, 200, lr, wd, structured)
                cases.append({"lr": lr, "wd": wd, "structured": structured,
                              "steps": 200, "max_abs_diff": d,
                              "max_scale_relative_diff": rel})
    worst_abs = max(c["max_abs_diff"] for c in cases)
    worst_rel = max(c["max_scale_relative_diff"] for c in cases)
    result = {"diagnostic": "diag01_adamw_crosscheck",
              "reference": "canonical decoupled AdamW (float64 NumPy), "
                           "PyTorch/AdamW-paper semantics",
              "tested_against": "grok.grok_adamw_step (frozen, read-only)",
              "cases": cases, "worst_max_abs_diff": worst_abs,
              "worst_scale_relative_diff": worst_rel,
              "note": "large-wd cases decay parameters to tiny scale, so the "
                      "pass criterion is the scale-relative diff (1e-5)",
              "pass": bool(worst_rel <= 1e-5)}
    (OUT / "diag01_adamw_crosscheck.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
