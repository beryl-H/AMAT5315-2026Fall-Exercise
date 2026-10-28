"""Formal dispatch CLI (ROOT only; this agent never launches formal runs).

    uv run python src/train_models.py --model mlp|transformer --attempt formal-01
    uv run python src/train_models.py --model mlp --attempt smoke-01 --smoke

Thin wrapper: fixes the project root on sys.path, then delegates to
src.train_driver.main(). The formal run uses full 4000/6000 updates with
midpoint checkpoints; the full held-out score is computed as postprocessing
AFTER the completion receipt (checkpoints are never lost to a later audit)."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main(argv=None):
    import argparse

    from src.train_driver import main as driver_main

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "transformer"], required=True)
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    return driver_main(["--model", args.model, "--attempt", args.attempt] +
                       (["--smoke"] if args.smoke else []))


if __name__ == "__main__":
    sys.exit(main())
