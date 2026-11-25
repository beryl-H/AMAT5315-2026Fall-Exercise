"""Regenerate the four required figures from saved spin-glass runs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = ("most-frac", "strong", "neural", "root-neural")
COLORS = {"most-frac": "#2D69A3", "strong": "#CF6637",
          "neural": "#12826A", "root-neural": "#8056A2"}


def read(out, relative):
    return json.loads((out / relative).read_text(encoding="utf-8"))


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180)
    plt.close(fig)


def gap(out):
    runs = read(out, "gap/run.json")["instances"]
    ids = [row["id"] for row in runs]
    lp = [row["lp"]["value"] for row in runs]
    mip = [row["mip"]["value"] for row in runs]
    fig, (bounds, excess) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    bounds.vlines(ids, lp, mip, color="#CBD2DB", lw=1)
    bounds.plot(ids, lp, "o-", ms=4, label="LP lower bound", color=COLORS["most-frac"])
    bounds.scatter(ids, mip, s=26, marker="s", label="Exact ground energy", color=COLORS["strong"])
    bounds.set_ylabel("Energy")
    for shift, key, label, color in [
        (-0.18, "gap", "Integrality gap", COLORS["most-frac"]),
        (0.18, "rounding_excess", "Rounding excess", COLORS["strong"]),
    ]:
        excess.bar([i + shift for i in ids], [row[key] for row in runs], width=0.36,
                   label=label, color=color)
    excess.set_ylabel("Energy difference")
    for ax in (bounds, excess):
        ax.set(xlabel="Instance ID", xticks=ids[::2])
        ax.legend(frameon=False)
    fig.suptitle("LP relaxation and rounding · 16 spin glasses, 22 spins / 52 bonds")
    save(fig, out, "gap")



def bnb(out):
    runs = [read(out, f"bnb_{name}/run.json")["instances"] for name in ("most", "strong")]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
    for ax, key, ylabel in zip(axes, ("nodes", "lp_solves"),
                               ("Explored nodes", "LP solves"), strict=True):
        for name, run in zip(POLICIES[:2], runs, strict=True):
            data = sorted(row["results"][name][key] for row in run)
            ax.plot(range(1, len(data) + 1), data, label=name, color=COLORS[name])
        ax.set(xlabel="Instance rank (per policy)", ylabel=ylabel)
        ax.legend(frameon=False)
    fig.suptitle("Branching rules · 48 spin glasses, 22 spins / 52 bonds")
    save(fig, out, "bnb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("gap", "bnb"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    {"gap": gap, "bnb": bnb}[args.figure](args.artifacts)
