"""Independently verify the saved Branch-and-Bound, model and evaluation runs."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_batch(run, count, seed, policies, tolerance):
    assert run["seed"] == seed
    assert [row["id"] for row in run["instances"]] == list(range(count))
    workshop = run["workshop"]
    assert workshop["nodes"] == workshop["lp_solves"] == 3
    assert workshop["plan"] == [3, 2] and abs(workshop["optimum"] - 27) < tolerance
    assert len(workshop["trace"]) == 3
    assert len(workshop["splits"]) == 1
    assert workshop["splits"][0]["column"] == 1
    assert [row["disposition"] for row in workshop["trace"]] == ["split", "incumbent", "bound"]
    for row, expected in zip(workshop["trace"], (83 / 3, 27, 26.75), strict=True):
        assert abs(row["bound"] - expected) < tolerance
    total = Counter()
    for row in run["instances"]:
        problem = row["problem"]
        assert len(problem["costs"]) == run["parameters"]["vars"]
        assert len(problem["rows"]) == run["parameters"]["rows"]
        assert set(row["results"]) == set(policies)
        for policy, result in row["results"].items():
            plan = result["plan"]
            assert len(plan) == len(problem["costs"]) and all(x in (0, 1) for x in plan)
            assert all(sum(plan[j] * a for j, a in constraint) >= 1 for constraint in problem["rows"])
            actual_cost = sum(c * x for c, x in zip(problem["costs"], plan, strict=True))
            assert abs(actual_cost - result["optimum"]) < tolerance
            assert abs(actual_cost - row["oracle"]["value"]) < tolerance
            assert result["abs_optimum_error"] < tolerance
            extra = result["lp_solves"] - result["nodes"]
            assert extra >= 0 and extra % 2 == 0
            assert policy == "strong" or extra == 0
            assert result["lp_ms"] + result["inference_ms"] <= result["runtime_ms"] + 1e-4
            total[policy] += 1
    return dict(total)


def check_bnb(out):
    most = load(out / "bnb_most/run.json")
    strong = load(out / "bnb_strong/run.json")
    check_batch(most, 48, 2026, ["most-frac"], 1e-9)
    check_batch(strong, 48, 2026, ["strong"], 1e-9)
    assert [x["problem"] for x in most["instances"]] == [x["problem"] for x in strong["instances"]]
    data = load(out / "bnb_strong/dataset.json")
    assert data["train_instance_ids"] == list(range(36))
    assert data["validation_instance_ids"] == list(range(36, 48))
    assert len(data["feature_names"]) == len(set(data["feature_names"])) == 12
    labels_per_instance = Counter()
    lookaheads = Counter()
    for split, ids in (("train", data["train_instance_ids"]), ("validation", data["validation_instance_ids"])):
        for record in data[split]:
            idx, candidates = record["instance_id"], record["candidates"]
            assert idx in ids and len(candidates) >= 2
            labels_per_instance[idx] += 1
            lookaheads[idx] += 2 * len(candidates)
            assert len({c["column"] for c in candidates}) == len(candidates)
            for c in candidates:
                assert len(c["features"]) == 12 and all(math.isfinite(f) for f in c["features"])
                assert 1e-6 < c["features"][0] < 1 - 1e-6
                gain_product = max(1e-6, c["gain0"]) * max(1e-6, c["gain1"])
                assert math.isclose(gain_product, c["score"], rel_tol=1e-12)
    assert labels_per_instance and max(labels_per_instance.values()) <= 80
    for row in strong["instances"]:
        s = row["results"]["strong"]
        assert s["lp_solves"] - s["nodes"] >= lookaheads[row["id"]]
    print(f"B&B: 96 oracle checks, 48 paired instances, {len(data['train'])}/{len(data['validation'])} train/validation nodes")


def check_model(out):
    model = load(out / "model.json")
    data = load(out / "bnb_strong/dataset.json")
    assert model["train_instance_ids"] == data["train_instance_ids"]
    assert model["validation_instance_ids"] == data["validation_instance_ids"]
    assert model["feature_names"] == data["feature_names"]
    features = [c["features"] for record in data["train"] for c in record["candidates"]]
    for i, column in enumerate(zip(*features, strict=True)):
        m = math.fsum(column) / len(column)
        s = math.sqrt(math.fsum((x - m)**2 for x in column) / len(column))
        assert math.isclose(model["mean"][i], m, rel_tol=1e-8, abs_tol=1e-10)
        if s > 1e-10:
            assert math.isclose(model["std"][i], s, rel_tol=1e-8, abs_tol=1e-10)
    assert [r["epoch"] for r in model["history"]] == list(range(1, 121))
    best = min(model["history"], key=lambda r: r["val_ce"])
    assert best["epoch"] == model["best_epoch"]
    mean, std = model["mean"], model["std"]
    w1, b1, w2, b2 = model["w1"], model["b1"], model["w2"], model["b2"]

    def forward(features):
        normalized = [(x - m) / s for x, m, s in zip(features, mean, std, strict=True)]
        hidden = [math.tanh(b + math.fsum(x * row[k] for x, row in zip(normalized, w1, strict=True)))
                  for k, b in enumerate(b1)]
        return b2 + math.fsum(a * b for a, b in zip(hidden, w2, strict=True))

    scores_by_metric = {key: [] for key in ("val_ce", "top1", "top3", "chance")}
    for record in data["validation"]:
        teacher = [c["score"] for c in record["candidates"]]
        highest = max(teacher)
        tied = {i for i, score in enumerate(teacher) if abs(score - highest) <= model["settings"]["tie_tolerance"]}
        logits = [forward(c["features"]) for c in record["candidates"]]
        order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
        log_z = logits[order[0]] + math.log(math.fsum(math.exp(x - logits[order[0]]) for x in logits))
        scores_by_metric["val_ce"].append(log_z - math.fsum(logits[i] for i in tied) / len(tied))
        scores_by_metric["top1"].append(order[0] in tied)
        scores_by_metric["top3"].append(bool(tied.intersection(order[:3])))
        scores_by_metric["chance"].append(len(tied) / len(teacher))
    for metric, values in scores_by_metric.items():
        assert math.isclose(best[metric], math.fsum(values) / len(values), abs_tol=1e-10)
    assert best["top1"] > best["chance"]
    assert 0 <= best["top1"] <= best["top3"] <= 1
    assert len(model["probes"]) == 6
    for probe in model["probes"]:
        assert probe["instance_id"] in data["validation_instance_ids"]
        assert len(probe["logits"]) == len(probe["probabilities"]) == len(probe["features"])
        assert math.isclose(sum(probe["probabilities"]), 1, rel_tol=1e-10)
    print(f"JAX: 120 epochs, 6 validation probes, Top-1 {best['top1']:.3f}, chance {best['chance']:.3f}")


def check_eval(out):
    policies = ["most-frac", "strong", "neural", "root-neural"]
    for suite, count, seed, dims in (("test", 16, 3026, (32, 24)), ("transfer", 8, 4026, (48, 34))):
        run = load(out / f"eval_{suite}/run.json")
        assert (run["parameters"]["vars"], run["parameters"]["rows"]) == dims
        check_batch(run, count, seed, policies, 1e-4)
        parity = run["probe_parity"]
        assert parity["max_abs_logit_error"] < 1e-4
        assert parity["max_abs_probability_error"] < 1e-4
        for row in run["instances"]:
            for result in row["results"].values():
                assert math.isclose(result["inference_ms"], result["feature_ms"] + result["score_ms"])
        best = run["largest_neural_saving"]
        assert best["nodes_saved"] == max(row["results"]["most-frac"]["nodes"] - row["results"]["neural"]["nodes"]
                                          for row in run["instances"])
        print(f"Evaluation {suite}: {count * len(policies)} oracle checks, maximum probe probability error "
              f"{parity['max_abs_probability_error']:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bnb", "model", "eval", "all"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    checks = {"bnb": check_bnb, "model": check_model, "eval": check_eval}
    for check in checks.values() if args.stage == "all" else (checks[args.stage],):
        check(args.artifacts)
