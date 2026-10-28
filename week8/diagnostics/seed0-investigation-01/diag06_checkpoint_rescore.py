"""diag06: Independent checkpoint rescoring for all 5 main runs.

Reloads checkpoint-final.npz for main-s0..s4, scores train/test with the
frozen loss_acc, and compares against the final log entry. Confirms the
logged numbers are reproducible from the saved weights (no drift between
checkpoints and logs). Also verifies the completion receipts against the
logs (memorization/grokking/steps_run) and computes source hashes of the
frozen files for the diagnostic receipt.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax.numpy as jnp

import grok  # frozen source, read-only

ROOT = Path(__file__).resolve().parents[2]
ATT = ROOT / "artifacts" / "part4" / "formal-decoder-02"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    data = grok.build_dataset()
    split = grok.fixed_split(0.30)
    tr_tok = jnp.asarray(data["inputs"][split["train"]])
    tr_tgt = jnp.asarray(data["targets"][split["train"]])
    te_tok = jnp.asarray(data["inputs"][split["test"]])
    te_tgt = jnp.asarray(data["targets"][split["test"]])

    per_seed = {}
    for seed in range(5):
        d = ATT / f"main-s{seed}"
        params = grok._detree(grok.load_params(d / "checkpoint-final.npz"))
        tr_l, tr_a = grok.loss_acc(params, tr_tok, tr_tgt)
        te_l, te_a = grok.loss_acc(params, te_tok, te_tgt)
        log = [json.loads(l) for l in (d / "train-log.jsonl").read_text().splitlines()]
        last = log[-1]
        receipt = json.loads((d / "completion-receipt.json").read_text())
        per_seed[seed] = {
            "rescored_train_acc": float(tr_a), "log_train_acc": last["train_acc"],
            "rescored_test_acc": float(te_a), "log_test_acc": last["test_acc"],
            "acc_diff": abs(float(te_a) - last["test_acc"]),
            "rescored_test_loss": float(te_l), "log_test_loss": last["test_loss"],
            "steps_run_matches_log": receipt["steps_run"] == last["step"],
            "mem_step_matches": receipt["memorization_step"] == next(
                (e["step"] for e in log if e["train_acc"] >= 0.99), None),
            "grok_step_matches": receipt["grok_step"] == next(
                (e["step"] for e in log if e["test_acc"] >= 0.99), None)}

    src = {}
    for name in ("grok.py", "run_part4.py", "grok_symmetric_variant.py"):
        p = ROOT / name
        src[name] = hashlib.sha256(p.read_bytes()).hexdigest()

    ok = all(v["acc_diff"] < 1e-5 and v["steps_run_matches_log"]
             and v["mem_step_matches"] and v["grok_step_matches"]
             for v in per_seed.values())
    result = {"diagnostic": "diag06_checkpoint_rescore", "per_seed": per_seed,
              "frozen_source_sha256": src,
              "pass": bool(ok)}
    (OUT / "diag06_checkpoint_rescore.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
