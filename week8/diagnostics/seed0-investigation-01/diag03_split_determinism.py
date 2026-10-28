"""diag03: Split and run-configuration determinism across the 12 formal runs.

Checks (read-only, no training):
 1. fixed_split(0.30)/(0.50) recomputed: sizes 2823/6586/4704, 50% superset
    of 30%, train/test disjoint, split identical for every seed (label- and
    seed-independent by construction, verified by recomputation).
 2. All 12 settings.json share identical source_hashes for grok.py/SPEC.md.
 3. Every settings.json matches the frozen protocol constants.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import grok  # frozen source, read-only

ROOT = Path(__file__).resolve().parents[2]
ATT = ROOT / "artifacts" / "part4" / "formal-decoder-02"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    s30 = grok.fixed_split(0.30)
    s50 = grok.fixed_split(0.50)
    tr, te = set(s30["train"].tolist()), set(s30["test"].tolist())
    checks = {
        "n_train_30": len(s30["train"]), "n_test_30": len(s30["test"]),
        "n_train_50": len(s50["train"]), "n_test_50": len(s50["test"]),
        "expected": {"n_train_30": 2823, "n_test_30": 9409 - 2823,
                      "n_train_50": 4704, "n_test_50": 9409 - 4704},
        "train_test_disjoint": not (tr & te),
        "frac50_superset_frac30": tr <= set(s50["train"].tolist()),
        "split_seed_independent": True,
    }
    # split seed-independence: the split function takes no seed argument;
    # verify by calling with different ambient seeds
    np.random.seed(0); a = grok.fixed_split(0.30)["train"].tolist()
    np.random.seed(12345); b = grok.fixed_split(0.30)["train"].tolist()
    checks["split_seed_independent"] = a == b

    runs = sorted(p.name for p in ATT.iterdir() if p.is_dir())
    hashes, settings_ok = {}, True
    for r in runs:
        st = json.loads((ATT / r / "settings.json").read_text())
        hashes[r] = st["source_hashes"]
        ok = (st["lr"] == grok.LR and st["beta1"] == grok.BETA1
              and st["beta2"] == grok.BETA2 and st["log_every"] == 100
              and st["acc_threshold"] == grok.ACC_THRESHOLD)
        if not ok:
            settings_ok = False
        if r.startswith("main-"):
            ok &= st["fraction"] == 0.30 and st["wd"] == 1.0 and st["seed"] == int(r[-1])
        if r.startswith("wd0-"):
            ok &= st["wd"] == 0.0
        if r.startswith("frac50-"):
            ok &= st["fraction"] == 0.50
        settings_ok &= ok
    distinct_hashes = {json.dumps(h, sort_keys=True) for h in hashes.values()}
    result = {"diagnostic": "diag03_split_determinism", "checks": checks,
              "n_runs": len(runs), "settings_match_protocol": settings_ok,
              "n_distinct_source_hash_sets": len(distinct_hashes),
              "source_hashes_grok": json.loads(
                  (ATT / "main-s0" / "settings.json").read_text())["source_hashes"],
              "pass": bool(checks["train_test_disjoint"]
                           and checks["frac50_superset_frac30"]
                           and checks["split_seed_independent"]
                           and checks["n_train_30"] == 2823
                           and settings_ok
                           and len(distinct_hashes) == 1)}
    (OUT / "diag03_split_determinism.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
