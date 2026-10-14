"""Command-line entrypoint: `diffusion train` and `diffusion sample`.

No unstated defaults: every option is required on the command line. The only
default is --trajectory being off, as prescribed by diffusion.design.toml.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffusion",
        description="train or sample a diffusion model; no unstated defaults",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a noise predictor; no unstated defaults")
    train.add_argument("--data", required=True, help="dataset: mnist now, ising in Part 4")
    train.add_argument("--input", required=True, help="training data file")
    train.add_argument("--updates", required=True, type=int, help="number of parameter updates")
    train.add_argument("--batch", required=True, type=int, help="training examples per update")
    train.add_argument("--seed", required=True, type=int,
                       help="random seed for initialization and all training draws")
    train.add_argument("--out", required=True, help="folder for this training run")

    sample = sub.add_parser("sample", help="sample a trained noise predictor; no unstated defaults")
    sample.add_argument("--data", required=True, help="dataset: mnist or ising")
    sample.add_argument("--checkpoint", required=True,
                        help="trained model to load; its schedule and image shape apply")
    sample.add_argument("--sampler", required=True, help="ddpm now; ddim in Part 3")
    sample.add_argument("--steps", required=True, type=int,
                        help="number of network evaluations per sample, from 2 to 1000")
    sample.add_argument("--n", required=True, type=int, help="number of generated samples")
    sample.add_argument("--seed", required=True, type=int,
                        help="random seed for initial noise and all reverse-step noise")
    sample.add_argument("--out", required=True, help="folder for this sampling run")
    sample.add_argument("--trajectory", action="store_true",
                        help="also record 8 samples using 1000 DDPM steps; default off")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "train":
        if args.data != "mnist":
            print(f"error: data {args.data!r} is not implemented (ising arrives in Part 4)",
                  file=sys.stderr)
            return 2
        if args.updates < 1 or args.batch < 1:
            print("error: updates and batch must be positive integers", file=sys.stderr)
            return 2
        from .train import run_training
        run_training(args.input, args.out, args.updates, args.batch, args.seed)
        return 0

    # sample
    if args.data != "mnist":
        print(f"error: data {args.data!r} is not implemented (ising arrives in Part 4)",
              file=sys.stderr)
        return 2
    if args.sampler != "ddpm":
        print(f"error: sampler {args.sampler!r} is not implemented (ddim arrives in Part 3)",
              file=sys.stderr)
        return 2
    if not 2 <= args.steps <= 1000:
        print("error: steps must be between 2 and 1000", file=sys.stderr)
        return 2
    if args.n < 1:
        print("error: n must be a positive integer", file=sys.stderr)
        return 2
    from .sample import run_sampling
    run_sampling(args.checkpoint, args.out, args.sampler, args.steps, args.n,
                 args.seed, trajectory=args.trajectory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())