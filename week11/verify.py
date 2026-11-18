"""Independently verify saved evidence for the Week 11 exercises."""
import argparse
import json
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


def main():
    checks = {"pred": verify_pred, "gadgets": verify_gates}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*checks, "all"))
    phase = parser.parse_args().phase
    for name, check in checks.items():
        if phase in ("all", name):
            check()


if __name__ == "__main__":
    main()
