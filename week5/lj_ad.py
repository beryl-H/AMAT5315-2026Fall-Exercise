"""Week 5: the Lennard-Jones pair energy as an explicit four-node computation
graph, with hand-written forward-mode (tangent) and reverse-mode (adjoint)
passes, plus JAX's own gradient as a cross-check.

Graph:

    a = r^-6      (a is r to the -6)
    b = a^2       (b is a squared, i.e. r^-12)
    c = b - a
    U = 4 c

64-bit floats: callers must enable jax x64 (done in lj_ad.__init__ via
module import side effect guarded below, and by the tests/scripts).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def energy(r):
    """The four-node LJ graph, as JAX sees it (for jax.grad)."""
    a = r ** -6.0
    b = a ** 2.0
    c = b - a
    return 4.0 * c


def forward_tangents(r: float, dr: float = 1.0) -> tuple[dict, dict]:
    """Hand-written forward mode, one node at a time.

    Returns (primals, tangents) with one entry per node r, a, b, c, U.
    The input tangent is `dr` (default 1 for dU/dr).
    """
    # --- primal pass ---
    a = r ** -6.0
    b = a ** 2.0
    c = b - a
    U = 4.0 * c

    # --- tangent pass (same order as the primals) ---
    d_a = -6.0 * r ** -7.0 * dr  # a = r^-6, so da = -6 r^-7 dr
    d_b = 2.0 * a * d_a          # b = a^2,  so db = 2 a da
    d_c = d_b - d_a              # c = b - a, so dc = db - da
    d_U = 4.0 * d_c              # U = 4 c,  so dU = 4 dc

    primals = {"r": r, "a": a, "b": b, "c": c, "U": U}
    tangents = {"r": dr, "a": d_a, "b": d_b, "c": d_c, "U": d_U}
    return primals, tangents


def reverse_adjoints(r: float) -> tuple[dict, dict]:
    """Hand-written reverse mode, one node at a time.

    Seeds U_bar = 1 and walks the graph backwards. Returns (primals,
    adjoints) with one entry per node r, a, b, c, U.
    """
    # --- primal pass (needed for the local derivatives) ---
    a = r ** -6.0
    b = a ** 2.0
    c = b - a
    U = 4.0 * c
    primals = {"r": r, "a": a, "b": b, "c": c, "U": U}

    # --- adjoint pass, in reverse topological order ---
    adjoints = {"r": 0.0, "a": 0.0, "b": 0.0, "c": 0.0, "U": 0.0}

    adjoints["U"] = 1.0            # seed: d/dU of the objective
    adjoints["c"] += adjoints["U"] * 4.0   # U = 4 c
    adjoints["b"] += adjoints["c"] * 1.0   # c = b - a
    adjoints["a"] += adjoints["c"] * (-1.0)
    adjoints["a"] += adjoints["b"] * 2.0 * a  # b = a^2
    adjoints["r"] += adjoints["a"] * (-6.0 * r ** -7.0)  # a = r^-6

    return primals, adjoints


def jax_grad(r: float) -> float:
    """Derivative from JAX's own gradient function, at r."""
    grad = jax.grad(energy)
    return float(grad(jnp.asarray(r, dtype=jnp.float64)))
