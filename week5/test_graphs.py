"""Tests for the jaxpr computational-graph plots (scripts/plot_graphs.py).

Derived from week5/SPEC.md correctness criteria 7.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "plot_graphs", ROOT / "scripts" / "plot_graphs.py"
)
plot_graphs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_graphs)


def test_forward_graph_primitives():
    nodes, _ = plot_graphs.build_energy_graph()
    ops = [n for n in nodes if n["kind"] == "op"]
    assert [n["label"] for n in ops] == ["pow", "pow", "sub", "mul"]


def test_grad_graph_op_count():
    nodes, _ = plot_graphs.build_grad_graph()
    ops = [n for n in nodes if n["kind"] == "op"]
    assert len(ops) == 15
    labels = [n["label"] for n in ops]
    assert "neg" in labels and "add_any" in labels


def test_plots_exist():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_graphs.py")], check=True
    )
    for name in ("graph.png", "grad-graph.png"):
        p = ROOT / "artifacts" / "ad" / name
        assert p.exists() and p.stat().st_size > 0
