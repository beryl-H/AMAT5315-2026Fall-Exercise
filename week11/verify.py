"""Independently verify saved evidence for the Week 11 exercises."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ARTIFACTS = ROOT / "artifacts"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(name):
    return read(RUNS / name / "run.json")


def save(name, value):
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(value, indent=2))


def little_endian(bits):
    assert all(bit in (0, 1) for bit in bits)
    return sum(bit << idx for idx, bit in enumerate(bits))


def verify_pred():
    folder = RUNS / "pred"
    path = (folder / "path.txt").read_text(encoding="utf-8")
    assert re.search(r"Factoring.*CircuitSAT.*SpinGlass", path, re.DOTALL)
    evidence = []
    for target, nbits in ((15, 5), (6, 4), (7, 4)):
        stem = folder / f"factor_{target}"
        bundle = read(stem.with_suffix(".bundle.json"))
        assert [p["name"] for p in bundle["path"]] == ["Factoring", "CircuitSAT", "SpinGlass"]
        assert bundle["target"]["type"] == "SpinGlass"
        variables = bundle["target"]["data"]["graph"]["num_vertices"]
        assert variables == (104 if target == 15 else 72)
        solve = read(stem.with_suffix(".solve.json"))
        evaluate = read(stem.with_suffix(".evaluate.json"))
        assert (solve["solver"], solve["reduced_to"]) == ("ilp", "SpinGlass")
        bits = solve["solution"]
        assert len(bits) == nbits
        p, q = little_endian(bits[:2]), little_endian(bits[2:])
        error = abs(target - p * q)
        assert solve["evaluation"] == evaluate["result"] == f"Min({error})"
        assert (p * q == target) == (target != 7)
        if target == 7:
            assert all(p * q != target for p in range(4) for q in range(4))
        evidence.append(dict(target=target, bits=bits, p=p, q=q, product=p*q, error=error, spins=variables))
    save("pred-check", evidence)


def verify_gates():
    data = run("gadgets")
    valid_checks = {
        "AND": lambda a, b, z: z == (a & b),
        "OR": lambda a, b, z: z == (a | b),
        "NOT": lambda a, z: z != a,
        "COPY": lambda a, z: z == a,
        "SET0": lambda z: z == 0,
        "SET1": lambda z: z == 1,
        "XOR": lambda a, b, z: z == (a ^ b),
    }
    floors = dict(AND=-3, OR=-3, NOT=-1, COPY=-1, SET0=-1, SET1=-1, XOR=-4)
    sizes = dict(AND=8, OR=8, NOT=4, COPY=4, SET0=2, SET1=2, XOR=8)
    gaps = {}
    for name, ground in floors.items():
        rows = data["gates"][name]
        assert len(rows) == sizes[name] == len({row["bits"] for row in rows})
        invalid = []
        for row in rows:
            valid = valid_checks[name](*map(int, row["bits"]))
            assert row["valid"] == valid
            if valid:
                assert row["energy"] == ground
            else:
                invalid.append(row["energy"] - ground)
        assert min(invalid) >= 2
        gaps[name] = min(invalid)
    circuit = data["circuit"]
    assert len(circuit) == len({row["bits"] for row in circuit}) == 16
    assert min(row["energy"] for row in circuit) == -5
    ground = sorted(row["bits"] for row in circuit if row["energy"] == -5)
    assert ground == ["0001", "0101", "1001"]
    save("gates-check", dict(floors=floors, gaps=gaps, circuit_ground_rows=ground))


def ground_config():
    data = run("sample_f15")
    ground = next(sample["bits"] for sample in data["samples"] if sample["energy"] == data["e_min"])
    print(",".join(map(str, ground)))


def verify_magnet():
    ferro = run("count_p0")
    assert (ferro["n_variables"], ferro["e_min"], ferro["g0"]) == (100, -180, 2)
    assert len(ferro["magnet"]["bonds"]) == 180
    assert all(energy == -1 for _, _, energy in ferro["magnet"]["bonds"])
    assert ferro["log_z"] >= 360 + math.log(2)
    factor = run("count_f15")
    assert (factor["n_variables"], factor["e_min"], factor["g0"]) == (104, -299, 1)
    assert abs(factor["log_z"] - 598.3179) < 5e-5
    draws = run("sample_f15")
    assert abs(factor["log_z"] - draws["log_z"]) < 1e-9
    assert (factor["e_min"], factor["g0"]) == (draws["e_min"], draws["g0"])
    beta = draws["parameters"]["beta"]
    assert beta == 2
    magnet = read(RUNS / "pred" / "factor_15.bundle.json")["target"]["data"]
    totals = Counter()
    for sample in draws["samples"]:
        bits = sample["bits"]
        assert len(bits) == 104 and set(bits) <= {0, 1}
        s = [2 * b - 1 for b in bits]
        energy = sum(k * s[i] * s[j] for k, (i, j) in zip(magnet["couplings"], magnet["graph"]["edges"]))
        energy += sum(h * v for h, v in zip(magnet["fields"], s))
        assert energy == sample["energy"] and energy >= -299
        totals[energy] += 1
    assert sum(totals.values()) == 2000
    p0 = factor["g0"] * math.exp(-beta * factor["e_min"] - factor["log_z"])
    expected = 2000 * p0
    sigma = math.sqrt(expected * (1 - p0))
    observed = totals[-299]
    assert observed == draws["ground_samples"]
    assert abs(observed - expected) < 3 * sigma
    extracted = read(RUNS / "sample_f15" / "extract.json")
    bits = extracted["solution"]
    p, q = little_endian(bits[:2]), little_endian(bits[2:])
    assert (p, q) == (3, 5)
    assert (extracted["evaluation"], extracted["solver"]) == ("Min(0)", "external")
    assert extracted["intermediate"]["evaluation"] == "Min(-299)"
    scaling = []
    for k in range(2, 9):
        line = {"k": k}
        for order in ("greedy", "treesa"):
            plan = run(f"plan_k{k}_{order}")
            assert plan["n_variables"] == 14 * k * k + 8 * k
            assert plan["planner"] == order and plan["mode"] == "plan"
            assert all(key not in plan for key in ("log_z", "e_min", "samples"))
            assert plan["sc"] >= 0 and plan["tc"] >= 0
            line[order] = {key: plan[key] for key in ("sc", "tc", "runtime_ms")}
        scaling.append(line)
    save("magnet-check", dict(
        ferro={"e_min": ferro["e_min"], "g0": ferro["g0"], "log_z": ferro["log_z"]},
        factor={"e_min": factor["e_min"], "g0": factor["g0"], "log_z": factor["log_z"], "sc": factor["sc"]},
        factors=[p, q], expected_ground=expected, observed_ground=observed, p0=p0, binomial_sigma=sigma,
        histogram=dict(sorted(totals.items())), scaling=scaling,
    ))



def verify_queens():
    reference = dict(zip(range(4, 11), [2, 10, 4, 40, 92, 352, 724]))
    for n, count in reference.items():
        data = run(f"count_nq{n}")
        assert data["board_count"] == count
    records = {}
    for n in (6, 8):
        data = run(f"sample_nq{n}")
        assert data["board_count"] == reference[n]
        assert len(data["samples"]) == 200
        frequencies = Counter(map(tuple, data["samples"]))
        for board in frequencies:
            assert set(board) == set(range(n)) and len(board) == n
            assert len({r + c for r, c in enumerate(board)}) == n
            assert len({r - c for r, c in enumerate(board)}) == n
        entry = dict(distinct=len(frequencies), example=list(data["samples"][0]))
        if n == 6:
            assert len(frequencies) == 4
            chi_squared = sum((count - 50)**2 / 50 for count in frequencies.values())
            assert chi_squared < 11.345  # 99th percentile, three degrees of freedom.
            entry["frequency_by_lexicographic_board"] = [frequencies[b] for b in sorted(frequencies)]
            entry["chi_squared"] = chi_squared
        else:
            m = reference[n]
            missed = (1 - 1/m)**200
            variance = m * missed * (1-missed) + m*(m-1)*((1-2/m)**200 - missed**2)
            expected = m * (1 - missed)
            assert abs(len(frequencies) - expected) < 4 * math.sqrt(variance)
            entry["expected_distinct"] = expected
            entry["distinct_sigma"] = math.sqrt(variance)
        records[n] = entry
    save("nqueens-check", dict(counts=reference, samples=records))



def main():
    checks = {"pred": verify_pred, "gadgets": verify_gates, "magnet": verify_magnet, "queens": verify_queens}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*checks, "all", "ground-config"))
    phase = parser.parse_args().phase
    if phase == "ground-config":
        ground_config()
        return
    for name, check in checks.items():
        if phase in ("all", name):
            check()


if __name__ == "__main__":
    main()
