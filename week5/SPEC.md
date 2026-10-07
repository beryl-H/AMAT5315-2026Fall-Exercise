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
- `week5/scripts/compare_modes.py` — compares hand-written forward mode,
  hand-written reverse mode and a centered finite difference
  $(U(r+h) - U(r-h))/(2h)$ with $h = 10^{-6}$ against the analytic
  $U'(r)$, on 601 equally spaced separations from 0.95 to 2.5
  (endpoints included). It saves `week5/artifacts/ad/modes.png` with two
  panels in reduced units — (1) $dU/dr$ vs $r$ for all methods,
  (2) absolute error vs $r$ on a log scale — and prints each method's
  maximum absolute error.

## Correctness criteria (comparison)

5. All three methods agree with the analytic $U'(r)$ across the sampled
   range: maximum absolute error below $10^{-12}$ for forward and reverse
   mode and below $10^{-8}$ for the centered finite difference with
   $h = 10^{-6}$ (roundoff-dominated, expected around $10^{-10}$).
6. The sampled grid has exactly 601 points and includes 0.95 and 2.5.
- `week5/scripts/plot_graphs.py` — draws the computational graph JAX
  actually records (via `jax.make_jaxpr`) for `U` at $r = 1.3$ and for
  `jax.grad(U)` at $r = 1.3$, one node per operation, each labelled with
  the operation's primitive name, and saves
  `week5/artifacts/ad/graph.png` (forward graph) and
  `week5/artifacts/ad/grad-graph.png` (gradient graph).

## Correctness criteria (graphs)

7. The forward graph contains exactly the primitives `pow, pow, sub, mul`
   (matching the four nodes $a, b, c, U$), and the gradient graph records
   the backward pass as 15 primitive operations. Both plots exist and are
   non-empty.

## Correctness criteria

1. Every node primal agrees with a direct float64 evaluation.
2. The tangent of `U` from the hand-written forward pass equals the analytic
   $U'(1.3)$ to relative tolerance $10^{-12}$; so does the adjoint of `r`
   from the hand-written reverse pass and `jax_grad`.
3. Forward and reverse agree with each other and with `jax_grad` to
   relative tolerance $10^{-12}$.
4. Negative control: perturbing any hand-written local derivative rule (for
   example replacing $-6r^{-7}$ with $-6r^{-6}$) must make the tests fail.
