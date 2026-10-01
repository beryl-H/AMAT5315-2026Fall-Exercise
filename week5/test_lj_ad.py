"""Tests for week 5: hand-written forward/reverse mode on the LJ pair energy.

Derived from week5/SPEC.md, not from the implementation. The analytic
reference U'(r) = -48 r^-13 + 24 r^-7 is written out independently here.
"""

import json
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import lj_ad

R = 1.3
REL_TOL = 1e-12


def u_analytic(r: float) -> float:
    return 4.0 * (r ** -12.0 - r ** -6.0)


def du_analytic(r: float) -> float:
    return -48.0 * r ** -13.0 + 24.0 * r ** -7.0


NODES = ("r", "a", "b", "c", "U")


def test_primals_match_direct_evaluation():
    primals, _ = lj_ad.forward_tangents(R, 1.0)
    assert primals["r"] == R
    assert primals["a"] == R ** -6.0
    assert primals["b"] == (R ** -6.0) ** 2.0
    assert primals["c"] == (R ** -6.0) ** 2.0 - R ** -6.0
    assert primals["U"] == u_analytic(R)


def test_forward_tangents_node_by_node():
    r = R
    a = r ** -6.0
    da = -6.0 * r ** -7.0  # d/dr of r^-6, with dr = 1
    b = a ** 2.0
    db = 2.0 * a * da
    c = b - a
    dc = db - da
    dU = 4.0 * dc

    _, tangents = lj_ad.forward_tangents(R, 1.0)
    for name, expected in zip(NODES, (1.0, da, db, dc, dU)):
        assert math.isclose(tangents[name], expected, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(tangents["U"], du_analytic(R), rel_tol=REL_TOL)


def test_reverse_adjoints_node_by_node():
    r = R
    a = r ** -6.0
    # seed U_bar = 1, then walk the graph backwards
    c_bar = 4.0           # U = 4 c
    b_bar = c_bar         # c = b - a
    a_bar = 2.0 * a * b_bar - c_bar  # b = a^2 and c = b - a
    r_bar = a_bar * (-6.0 * r ** -7.0)  # a = r^-6

    primals, adjoints = lj_ad.reverse_adjoints(R)
    assert primals["U"] == u_analytic(R)
    for name, expected in zip(NODES, (r_bar, a_bar, b_bar, c_bar, 1.0)):
        assert math.isclose(adjoints[name], expected, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(adjoints["r"], du_analytic(R), rel_tol=REL_TOL)


def test_jax_grad_matches_analytic():
    assert math.isclose(lj_ad.jax_grad(R), du_analytic(R), rel_tol=REL_TOL)


def test_all_three_passes_agree():
    _, tangents = lj_ad.forward_tangents(R, 1.0)
    _, adjoints = lj_ad.reverse_adjoints(R)
    ref = du_analytic(R)
    for value in (tangents["U"], adjoints["r"], lj_ad.jax_grad(R)):
        assert math.isclose(value, ref, rel_tol=REL_TOL)


def test_float64_precision():
    # With 32-bit x default the tolerance test would be flaky; ensure the
    # results actually come out as float64-accurate against the analytic value.
    _, tangents = lj_ad.forward_tangents(R, 1.0)
    ref = du_analytic(R)
    assert abs(tangents["U"] - ref) <= REL_TOL * abs(ref)


def test_artifact_json(tmp_path):
    """The run script writes derivatives.json with the SPEC-required keys."""
    import subprocess
    import sys

    artifacts = Path(__file__).parent / "artifacts" / "ad"
    script = Path(__file__).parent / "scripts" / "run_ad.py"
    subprocess.run([sys.executable, str(script)], check=True)
    data = json.loads((artifacts / "derivatives.json").read_text())

    assert math.isclose(data["r"], R)
    assert math.isclose(data["energy"], u_analytic(R), rel_tol=REL_TOL)
    assert set(data["tangents"]) == set(NODES)
    assert set(data["adjoints"]) == set(NODES)
    assert math.isclose(data["jax_grad"], du_analytic(R), rel_tol=REL_TOL)
    assert math.isclose(data["tangents"]["U"], du_analytic(R), rel_tol=REL_TOL)
    assert math.isclose(data["adjoints"]["r"], du_analytic(R), rel_tol=REL_TOL)
