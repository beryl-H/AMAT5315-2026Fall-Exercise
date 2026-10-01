# Week 5 — Hand-written forward and reverse mode for the Lennard-Jones pair energy

Toolchain: Python 3.12, `jax` (CPU, 64-bit enabled), `numpy`, `matplotlib`, `pytest`.

## Problem

Differentiate the Lennard-Jones pair energy

$$U(r) = 4\,(r^{-12} - r^{-6})$$

at the separation $r = 1.3$ using 64-bit floats. Instead of letting JAX's
tracer see the raw expression, the computation is split into an explicit
computation-graph of four intermediate nodes, and the forward-mode (tangent)
and reverse-mode (adjoint) passes are written **by hand, one node at a time**:

| node | primal | tangent (forward, with $\dot r = 1$) | adjoint (reverse, with $\bar U = 1$) |
|------|--------|--------------------------------------|--------------------------------------|
| $r$  | $r$                          | $1$                            | $\bar a \cdot (-6 r^{-7})$ |
| $a$  | $a = r^{-6}$                 | $\dot a = -6 r^{-7}\,\dot r$   | $\bar b \cdot 2a - \bar c$ |
| $b$  | $b = a^{2}$                  | $\dot b = 2a\,\dot a$          | $\bar c$                   |
| $c$  | $c = b - a$                  | $\dot c = \dot b - \dot a$     | $4$                        |
| $U$  | $U = 4c$                     | $\dot U = 4\,\dot c$           | $1$                        |

The analytic cross-check is

$$U'(r) = -48\,r^{-13} + 24\,r^{-7}, \qquad U'(1.3) \approx 2.2399799297911431$$

## Deliverables

- `week5/lj_ad.py` — the four-node computation graph plus:
  - `forward_tangents(r, dr)`: hand-written forward pass, returns the primal
    and the tangent of every node;
  - `reverse_adjoints(r)`: hand-written reverse pass, returns the primal and
    the adjoint of every node, seeded with $\bar U = 1$;
  - `jax_grad(r)`: the derivative from JAX's own `jax.grad` applied to the
    same four-node function.
- `week5/test_lj_ad.py` — tests derived from this SPEC (not from the
  implementation), including the analytic reference for $U'(1.3)$.
- `week5/scripts/run_ad.py` — runs all three passes at $r = 1.3$ and writes
  `week5/artifacts/ad/derivatives.json` with the keys:
  - `r`, `energy`: the separation and $U(r)$;
  - `tangents`: the tangent of each node `r`, `a`, `b`, `c`, `U`;
  - `adjoints`: the adjoint of each node `r`, `a`, `b`, `c`, `U`;
  - `jax_grad`: the derivative from JAX's own gradient function, at `r`.

## Correctness criteria

1. Every node primal agrees with a direct float64 evaluation.
2. The tangent of `U` from the hand-written forward pass equals the analytic
   $U'(1.3)$ to relative tolerance $10^{-12}$; so does the adjoint of `r`
   from the hand-written reverse pass and `jax_grad`.
3. Forward and reverse agree with each other and with `jax_grad` to
   relative tolerance $10^{-12}$.
4. Negative control: perturbing any hand-written local derivative rule (for
   example replacing $-6r^{-7}$ with $-6r^{-6}$) must make the tests fail.
