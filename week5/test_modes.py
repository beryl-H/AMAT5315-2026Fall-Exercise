"""Tests for the week 5 mode comparison (scripts/compare_modes.py).

Derived from week5/SPEC.md correctness criteria 5 and 6.
"""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "compare_modes", Path(__file__).parent / "scripts" / "compare_modes.py"
)
compare_modes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_modes)


def test_grid():
    r = compare_modes.sample_separations()
    assert r.shape == (601,)
    assert r[0] == 0.95 and r[-1] == 2.5
    assert np.all(np.diff(r) > 0)


def test_error_bounds():
    errors = compare_modes.run_comparison()
    assert errors["forward"] < 1e-12
    assert errors["reverse"] < 1e-12
    assert errors["finite_difference"] < 1e-8


def test_artifact_plot_exists():
    import subprocess
    import sys

    plot = Path(__file__).parent / "artifacts" / "ad" / "modes.png"
    script = Path(__file__).parent / "scripts" / "compare_modes.py"
    subprocess.run([sys.executable, str(script)], check=True)
    assert plot.exists() and plot.stat().st_size > 0
