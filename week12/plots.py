"""Regenerate the Week 12 figures from saved JSON evidence."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ("most-frac", "strong", "neural", "root-neural")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finish(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=190)
    plt.close(fig)


def gap(out):
    instances = read(out / "gap/run.json")["instances"]
    ids = [record["id"] for record in instances]
    fig, (bounds, overhead) = plt.subplots(1, 2, figsize=(11.3, 4.2), layout="constrained")
    lp = [record["lp"]["value"] for record in instances]
    mip = [record["mip"]["value"] for record in instances]
    repair = [record["repaired_cost"] for record in instances]
    bounds.vlines(ids, lp, repair, color="#c9cfd5")
    bounds.scatter(ids, lp, label="LP lower bound", color=COLORS[0])
    bounds.scatter(ids, mip, label="HiGHS MILP optimum", color=COLORS[1], marker="s")
    bounds.scatter(ids, repair, label="Repaired rounding", color=COLORS[2], marker="^")
    bounds.set_ylabel("Covering cost")
    bounds.legend()
    relative_gap = [100 * record["relative_gap"] for record in instances]
    extra_cost = [100 * record["rounding_overhead"] for record in instances]
    overhead.bar([i - 0.19 for i in ids], relative_gap, width=0.38, color=COLORS[0], label="Integrality gap")
    overhead.bar([i + 0.19 for i in ids], extra_cost, width=0.38, color=COLORS[2], label="Repair overhead")
    overhead.set_ylabel("Percent of MILP optimum")
    overhead.legend()
    for ax in (bounds, overhead):
        ax.set(xlabel="Instance ID", xticks=ids[::2])
    fig.suptitle("Set Covering LP relaxation and rounding, 32 × 24")
    finish(fig, out, "gap.png")


def bnb(out):
    runs = [read(out / f"bnb_{name}/run.json")["instances"] for name in ("most", "strong")]
    metrics = (("nodes", "Nodes"), ("lp_solves", "LP solves"), ("runtime_ms", "Elapsed time (ms)"))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.9), layout="constrained")
    for (key, label), ax in zip(metrics, axes, strict=True):
        values = [mean(row["results"][name][key] for row in run) for name, run in zip(POLICIES, runs)]
        bars = ax.bar(POLICIES[:2], values, color=COLORS[:2])
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_ylabel(f"Mean {label.lower()} per instance")
        ax.set_ylim(0, max(values) * 1.22)
    fig.suptitle("Strong Branching lookahead, 48 paired instances of size 32 × 24")
    finish(fig, out, "bnb.png")


def training(out):
    model = read(out / "model.json")
    history = model["history"]
    epochs = [row["epoch"] for row in history]
    fig, (loss, acc) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for key, label, color in zip(("train_ce", "val_ce"), ("Training", "Validation"), COLORS):
        loss.plot(epochs, [row[key] for row in history], label=label, color=color)
    loss.set_ylabel("Cross-entropy")
    for key, label, color in zip(("top1", "top3", "chance"), ("Top-1", "Top-3", "Random Top-1"), COLORS):
        acc.plot(epochs, [row[key] for row in history], label=label, color=color)
    acc.set_ylabel("Validation success rate")
    acc.set_ylim(0, 1)
    for ax in (loss, acc):
        ax.set_xlabel("Epoch")
        ax.axvline(model["best_epoch"], color="#585f68", linestyle="--", linewidth=1, label="Saved epoch")
        ax.legend()
    fig.suptitle("JAX imitation of Strong Branching")
    finish(fig, out, "training.png")


def evaluation(out):
    suites = ("32 × 24, held out", "48 × 34, transfer")
    runs = [read(out / f"eval_{name}/run.json")["instances"] for name in ("test", "transfer")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for name, run, (nodes, time) in zip(suites, runs, axes, strict=True):
        def average(policy, key):
            return mean(row["results"][policy][key] for row in run)

        node_means = [average(policy, "nodes") for policy in POLICIES]
        bars = nodes.bar(POLICIES, node_means, color=COLORS)
        nodes.bar_label(bars, fmt="%.1f", padding=3)
        nodes.set_ylim(0, max(node_means) * 1.22)
        nodes.set(ylabel="Mean explored nodes", title=name)
        lp = [average(policy, "lp_ms") for policy in POLICIES]
        mlp = [average(policy, "inference_ms") for policy in POLICIES]
        totals = [average(policy, "runtime_ms") for policy in POLICIES]
        subtotal = [t + m for t, m in zip(lp, mlp, strict=True)]
        other = [max(0.0, t - s) for t, s in zip(totals, subtotal, strict=True)]
        time.bar(POLICIES, lp, color=COLORS[0], label="HiGHS LP")
        time.bar(POLICIES, mlp, bottom=lp, color=COLORS[1], label="Features + scores")
        time.bar(POLICIES, other, bottom=subtotal, color="#aab3bd", label="Other search work")
        for i, (total, infer) in enumerate(zip(totals, mlp, strict=True)):
            detail = f"\nincl. {infer:.2f}" if infer > 0 else ""
            time.annotate(f"{total:.1f}{detail}", (i, total), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=8)
        time.set_ylim(0, max(totals) * 1.27)
        time.set(ylabel="Mean elapsed time (ms)", title=name)
    axes[0, 1].legend()
    fig.suptitle("Exact branch-and-bound with four branching policies")
    finish(fig, out, "evaluation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("gap", "bnb", "training", "evaluation", "all"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    functions = {fn.__name__: fn for fn in (gap, bnb, training, evaluation)}
    for fn in functions.values() if args.figure == "all" else (functions[args.figure],):
        fn(args.artifacts)
