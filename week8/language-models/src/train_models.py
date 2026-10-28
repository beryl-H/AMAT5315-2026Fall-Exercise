"""Formal dispatch CLI (root only). Thin wrapper so the root controller can
run: uv run python src/train_models.py --model mlp|transformer --attempt formal-01
without --smoke. Identical to the driver smoke path but with full 4000/6000
updates, midpoint checkpoints and the full held-out score."""
import sys

from src.train_driver import main as driver_main


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "transformer"], required=True)
    ap.add_argument("--attempt", required=True)
    args = ap.parse_args(argv)
    # delegate; full run (no --smoke) with full held-out score
    sys.argv = ["train_models", "--model", args.model, "--attempt", args.attempt]
    return driver_main(["--model", args.model, "--attempt", args.attempt])


if __name__ == "__main__":
    from src.train_driver import main as dm
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "transformer"], required=True)
    ap.add_argument("--attempt", required=True)
    a = ap.parse_args(argv)
    sys.exit(dm(["--model", a.model, "--attempt", a.attempt]))
