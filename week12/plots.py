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



def neural_bnb(out):
    model = read(out, "model.json")
    run = read(out, "eval_test/run.json")["instances"]
    history = model["history"]
    epochs = [row["epoch"] for row in history]
    fig, (loss, nodes) = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for key, label, color in [("train_ce", "Training", "most-frac"),
                               ("val_ce", "Validation", "strong")]:
        loss.plot(epochs, [row[key] for row in history], label=label, color=COLORS[color])
    loss.axvline(model["best_epoch"], color="#656B73", ls="--", lw=1, label="Saved epoch")
    loss.set(xlabel="Epoch", ylabel="Cross-entropy")
    for name in POLICIES[:3]:
        counts = sorted(row["results"][name]["nodes"] for row in run)
        nodes.plot(range(1, len(counts) + 1), counts, label=name, color=COLORS[name])
    nodes.set(xlabel="Instance rank (per policy)", ylabel="Explored nodes")
    for ax in (loss, nodes):
        ax.legend(frameon=False)
    fig.suptitle("Imitation learning and exact search · 16 unseen spin glasses, 22 spins / 52 bonds")
    save(fig, out, "neural_bnb")



def evaluation(out):
    test, transfer = [read(out, f"eval_{name}/run.json") for name in ("test", "transfer")]
    fig, (nodes, timing) = plt.subplots(1, 2, figsize=(11.7, 4.4), layout="constrained")
    labels = ("22 spins / 52 bonds", "30 spins / 72 bonds")
    for i, name in enumerate(POLICIES):
        means = [run["summary"][f"{name}_mean_nodes"] for run in (test, transfer)]
        x = [group + (i - 1.5) * 0.19 for group in range(2)]
        bars = nodes.bar(x, means, width=0.18, label=name, color=COLORS[name])
        nodes.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    nodes.set(title="Mean explored nodes", xticks=[0, 1], xticklabels=labels)
    nodes.set_ylim(0, max(nodes.get_ylim()[1] * 1.17, 1))
    nodes.legend(frameon=False, fontsize=8, ncols=2)

    summary = transfer["summary"]
    lp = [summary[f"{p}_mean_lp_ms"] for p in POLICIES]
    infer = [summary[f"{p}_mean_inference_ms"] for p in POLICIES]
    total = [summary[f"{p}_mean_runtime_ms"] for p in POLICIES]
    bottom = [a + b for a, b in zip(lp, infer, strict=True)]
    other = [max(0, a - b) for a, b in zip(total, bottom, strict=True)]
    timing.bar(POLICIES, lp, color=COLORS["most-frac"], label="HiGHS LP")
    timing.bar(POLICIES, infer, bottom=lp, color=COLORS["strong"], label="Features + MLP")
    timing.bar(POLICIES, other, bottom=bottom, color="#C5CCD4", label="Other search")
    for i, (t, m) in enumerate(zip(total, infer, strict=True)):
        detail = f"\nMLP: {m:.3f}" if m else ""
        timing.annotate(f"{t:.1f}{detail}", (i, t), ha="center", fontsize=8,
                        xytext=(0, 4), textcoords="offset points")
    timing.set(title="Mean CPU search time · 30 spins / 72 bonds",
               ylabel="Elapsed time (ms)", ylim=(0, max(total) * 1.19))
    timing.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Neural branching on held-out and larger spin glasses")
    save(fig, out, "eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("gap", "bnb", "neural_bnb", "eval", "all"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    figures = {"gap": gap, "bnb": bnb, "neural_bnb": neural_bnb, "eval": evaluation}
    for figure in figures.values() if args.figure == "all" else (figures[args.figure],):
        figure(args.artifacts)
