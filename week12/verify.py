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


def check_model():
    data = read("bnb_strong/dataset.json")
    model = read("model.json")
    assert data["feature_names"] == model["feature_names"]
    val_rows = {}
    for name, ids in (("train", set(range(36))), ("validation", set(range(36, 48)))):
        assert set(data[f"{name}_instance_ids"]) == set(model[f"{name}_instance_ids"]) == ids
        rows = data[name]
        counts = Counter(row["instance_id"] for row in rows)
        assert set(counts) == ids and max(counts.values()) <= 80
        seen = set()
        for row in rows:
            key = (row["instance_id"], row["node"])
            assert key not in seen
            seen.add(key)
            candidates = row["candidates"]
            spins = [c["spin"] for c in candidates]
            assert len(spins) >= 2 and spins == sorted(set(spins))
            for c in candidates:
                fs = c["features"]
                assert len(fs) == 12 and all(math.isfinite(f) for f in fs)
                assert 1e-6 < fs[0] < 1 - 1e-6
                assert math.isclose(fs[1], 2 * min(fs[0], 1 - fs[0]))
                assert c["score"] == max(c["gain0"], 1e-6) * max(c["gain1"], 1e-6)
                assert c["pruned_children"] == sum(c[g] == 1e6 for g in ("gain0", "gain1"))
            if name == "validation":
                val_rows[key] = candidates
        print(f"{name}: {len(rows)} nodes / {len(ids)} graphs")

    train_feat = [c["features"] for row in data["train"] for c in row["candidates"]]
    for features, mu, sigma in zip(zip(*train_feat), model["mean"], model["std"], strict=True):
        assert abs(mean(features) - mu) < 1e-12
        sd = mean((f - mu) ** 2 for f in features) ** 0.5
        assert abs(max(sd, 1e-6) - sigma) < 1e-12
    history = model["history"]
    assert [r["epoch"] for r in history] == list(range(1, 121))
    eligible = (h for h in history if h["epoch"] == 1 or h["epoch"] % 5 == 0)
    assert model["best_epoch"] == min(eligible, key=lambda h: h["val_ce"])["epoch"]
    settings = model["settings"]
    assert settings["platform"] == "cpu" and settings["jax_version"] == "0.12.1"
    assert len(model["probes"]) == 6
    for p in model["probes"]:
        row = val_rows[p["instance_id"], p["node"]]
        assert p["columns"] == [c["spin"] for c in row]
        assert p["features"] == [c["features"] for c in row]
        assert abs(sum(p["probabilities"]) - 1) < 1e-12
    print(f"Model: validated teacher records, train-only normalization, epoch {model['best_epoch']} and six held-out probes.")
    return model


def check_eval(run, model, policies, expected):
    params = run["parameters"]
    assert (params["spins"], params["edges"], params["count"], run["seed"]) == expected
    cases = run["instances"]
    assert len(cases) == expected[2]
    for case in cases:
        assert set(case["results"]) == set(policies)
        for name in policies:
            check_search(case, name, 1e-6)
            s = case["results"][name]
            assert 0 <= s["inference_ms"] <= s["runtime_ms"] - s["lp_ms"] + 1e-8
            if name == "neural":
                assert s["feature_batches"] == (s["nodes"] - 1) // 2
            if name == "root-neural":
                assert s["feature_batches"] == 1
    parity = run["probe_parity"]
    assert parity["passed"] and len(parity["probes"]) == 6
    for rust, jax in zip(parity["probes"], model["probes"], strict=True):
        assert (rust["instance_id"], rust["node"], rust["columns"]) == (
            jax["instance_id"], jax["node"], jax["columns"])
        for key, rust_key in (("logits", "rust_logits"), ("probabilities", "rust_probabilities")):
            assert all(abs(a - b) < 1e-4 for a, b in zip(jax[key], rust[rust_key], strict=True))
    for name in ("neural", "root-neural"):
        if name in policies:
            assert run["summary"][f"{name}_mean_nodes"] < run["summary"]["most-frac_mean_nodes"]
    saving = run["largest_neural_saving"]
    def key(c):
        r = c["results"]
        return r["most-frac"]["nodes"] - r["neural"]["nodes"], -c["id"]
    best = max(cases, key=key)
    assert (saving["nodes_saved"], saving["id"]) == (key(best)[0], best["id"])
    for name in ("most-frac", "neural"):
        assert saving[name]["splits"] == best["results"][name]["splits"]
    print(f"Evaluation: {len(cases) * len(policies)} exact solves, all LP counts and six JAX probes verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bnb", "model"))
    args = parser.parse_args()
    if args.stage == "bnb":
        bnb()
    else:
        model = check_model()
        run = read("eval_test/run.json")
        check_eval(run, model, POLICIES[:3], (22, 52, 16, 3026))
