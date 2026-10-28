"""Part 4 formal driver (NOT launched in this checkpoint).

Producing command (when released by Codex):
    JAX_PLATFORMS=cpu uv run python run_part4.py --attempt formal-01

Runs the frozen Part-4 program: seeds 0..4 main (30%, wd 1), controls
seeds 0..2 wd 0 and 50% fraction, one random-label run (seed 0, 3000
steps). Exclusive-new attempt directories; every run via grok.run_one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import grok as gk

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True,
                    help="exclusive-new attempt name, e.g. formal-01")
    args = ap.parse_args(argv)
    attempt = ROOT / "artifacts" / "part4" / args.attempt
    assert not attempt.exists(), f"refusing existing attempt: {attempt}"
    attempt.mkdir(parents=True)
    started = time.time()
    plan = []
    runs = {}
    # main runs: seeds 0..4, 30% split, wd 1
    for seed in range(5):
        name = f"main-s{seed}"
        plan.append(name)
        runs[name] = gk.run_one(seed=seed, fraction=0.30, wd=gk.WD,
                                max_steps=gk.MAX_STEPS,
                                outdir=attempt / name)
    # controls: wd 0 and 50% fraction, seeds 0..2
    for seed in range(3):
        name = f"wd0-s{seed}"
        plan.append(name)
        runs[name] = gk.run_one(seed=seed, fraction=0.30, wd=0.0,
                                max_steps=gk.MAX_STEPS,
                                outdir=attempt / name)
    for seed in range(3):
        name = f"frac50-s{seed}"
        plan.append(name)
        runs[name] = gk.run_one(seed=seed, fraction=0.50, wd=gk.WD,
                                max_steps=gk.MAX_STEPS,
                                outdir=attempt / name)
    # random-label control: seed 0, 3000 steps, main settings
    name = "random-labels-s0"
    plan.append(name)
    runs[name] = gk.run_one(seed=0, fraction=0.30, wd=gk.WD, max_steps=3000,
                            outdir=attempt / name, permuted_labels=True)

    summary = {"attempt": args.attempt, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "elapsed_s": round(time.time() - started, 1), "runs": plan,
               "agent": "pi (native WSL) volcengine-plan/glm-5-3-flash"}
    (attempt / "attempt-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"attempt {args.attempt} complete: {attempt}")


if __name__ == "__main__":
    main()
