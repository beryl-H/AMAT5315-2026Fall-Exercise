"""Independently check saved Week 12 runs against the learning sheet."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

from tests.test_gap_evidence import energy

BASE = Path(__file__).resolve().parent / "artifacts"
POLICIES = ("most-frac", "strong", "neural", "root-neural")


def read(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def check_search(case, name, tolerance):
    search = case["results"][name]
    graph = case["graph"]
    n = len(graph["fields"])
    bits = search["ground_bits"]
    assert len(bits) == n and set(bits) <= {0, 1}
    oracle_x = case["oracle"]["x"][:n]
    oracle_bits = [round(bit) for bit in oracle_x]
    assert len(oracle_bits) == n and set(oracle_bits) <= {0, 1}
    assert all(abs(x - bit) < 1e-6 for x, bit in zip(oracle_x, oracle_bits, strict=True))
    for candidate in (bits, oracle_bits):
        assert abs(energy(graph, candidate) - case["oracle"]["value"]) < tolerance
    error = abs(search["optimum"] - case["oracle"]["value"])
    assert error < tolerance and error == search["abs_optimum_error"]
    assert search["nodes"] > 0
    assert search["lp_solves"] == search["nodes"] + 2 * search["probed_candidates"]
    if name != "strong":
        assert search["probed_candidates"] == 0
    assert 0 <= search["lp_ms"] <= search["runtime_ms"] + 1e-8
    return error


def bnb():
    most, strong = (read(f"bnb_{name}/run.json") for name in ("most", "strong"))
    gap = read("gap/run.json")
    assert most["seed"] == strong["seed"] == gap["seed"] == 2026
    assert most["parameters"] == strong["parameters"]
    assert len(most["instances"]) == len(strong["instances"]) == 48
    for run, policy in [(most, "most-frac"), (strong, "strong")]:
        trace = run["tetrahedron"]["trace"]
        assert [r["bound"] for r in trace] == [-6, -4, -3, -3, -2]
        assert [r["branch"] for r in trace] == [
            None, {"spin":0,"bit":1}, {"spin":1,"bit":1},
            {"spin":1,"bit":0}, {"spin":0,"bit":0},
        ]
        assert [r["disposition"] for r in trace] == ["split", "split", "incumbent", "bound", "bound"]
        assert trace[-1]["x"] == [0, 0.5, 0.5, 0.5]
        assert max(check_search(c, policy, 1e-9) for c in run["instances"]) < 1e-9
    for a, b in zip(most["instances"], strong["instances"], strict=True):
        assert a["id"] == b["id"] and a["graph"] == b["graph"]
    for a, b in zip(gap["instances"], most["instances"][:16], strict=True):
        assert a["id"] == b["id"] and a["graph"] == b["graph"]
        assert abs(a["mip"]["value"] - b["oracle"]["value"]) < 1e-9
    assert strong["summary"]["strong_mean_nodes"] < most["summary"]["most-frac_mean_nodes"]
    assert strong["summary"]["strong_mean_lp_solves"] > most["summary"]["most-frac_mean_lp_solves"]
    print("B&B: matched HiGHS for 96 solves and checked every LP count and tetrahedron branch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bnb",))
    args = parser.parse_args()
    bnb()
