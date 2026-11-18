"""Render the required Week 11 figures from saved, validated JSON runs."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE = "#176a9b"
ORANGE = "#b64d21"
DIM = "#707983"
RUNS = Path("runs")
ARTIFACTS = Path("artifacts")


def load(name):
    return json.loads((RUNS / name / "run.json").read_text(encoding="utf-8"))


def save(fig, name):
    ARTIFACTS.mkdir(exist_ok=True)
    fig.savefig(ARTIFACTS / f"{name}.png", dpi=180)
    plt.close(fig)


def plot_gates():
    data = load("gadgets")
    fig, (ax, circuit) = plt.subplots(1, 2, figsize=(12.8, 4.6), layout="constrained")
    names = ("AND", "OR", "NOT", "COPY", "SET0", "SET1", "XOR")
    for group, name in enumerate(names):
        rows = data["gates"][name]
        for idx, row in enumerate(rows):
            x = group + (idx - (len(rows) - 1) / 2) * 0.095
            valid = row["valid"]
            ax.scatter(x, row["energy"], color=BLUE if valid else ORANGE, marker="o" if valid else "x", s=26)
    ax.set(xticks=range(len(names)), xticklabels=names, ylabel="Ising energy", title="(a) All truth-table rows")
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", linestyle="none", label="Allowed"),
        Line2D([], [], color=ORANGE, marker="x", linestyle="none", label="Forbidden"),
    ], loc="upper right")
    rows = data["circuit"]
    circuit.bar(range(len(rows)), [row["energy"] for row in rows],
                color=[BLUE if r["valid"] else DIM for r in rows], width=0.72)
    circuit.axhline(-5, color=BLUE, linewidth=0.9, linestyle="--")
    circuit.set(
        xticks=range(len(rows)), xticklabels=[row["bits"] for row in rows], xlabel="Bits a, b, t, out",
        ylabel="Circuit energy", title="(b) AND → NOT with output pinned to 1",
    )
    circuit.tick_params(axis="x", rotation=55, labelsize=8)
    for axes in (ax, circuit):
        axes.grid(axis="y", alpha=0.22)
        axes.set_axisbelow(True)
    save(fig, "gates")


def plot_factoring():
    fig, (ax, hist) = plt.subplots(1, 2, figsize=(11.5, 4.35), layout="constrained")
    sizes = list(range(2, 9))
    for order, color, marker in (("greedy", ORANGE, "s"), ("treesa", BLUE, "o")):
        widths = [load(f"plan_k{k}_{order}")["sc"] for k in sizes]
        ax.plot(sizes, widths, marker=marker, label=order, color=color)
    ax.plot(sizes[1:], [2*k+1 for k in sizes[1:]], "--", color=DIM, label="Reference 2k + 1")
    ax.axhline(30, ls=":", lw=1, color=DIM, label="8 GiB at 8 bytes/entry")
    ax.set(xticks=sizes, xlabel="Bits per factor k", ylabel="Space complexity sc (log₂ entries)",
           title="(a) OMECO contraction plan width")
    ax.legend()
    data = load("sample_f15")
    counts = Counter(sample["energy"] for sample in data["samples"])
    energies = sorted(counts)
    e0 = data["e_min"]
    beta = data["parameters"]["beta"]
    expected = len(data["samples"]) * data["g0"] * math.exp(-beta * e0 - data["log_z"])
    bars = hist.bar(energies, [counts[e] for e in energies], width=1.38,
                    color=[BLUE if e == e0 else DIM for e in energies], label="Sampled")
    hist.bar_label(bars, padding=3, fontsize=8)
    hist.scatter([e0], [expected], marker="D", facecolors="white", edgecolors=ORANGE,
                 s=65, linewidths=1.5, zorder=4, label=f"Expected ground count {expected:.1f}")
    hist.set(xticks=energies, xlabel="Spin-glass energy", ylabel="Sample count",
             title=f"(b) Factoring 15 at β = {beta:g}")
    hist.set_ylim(top=max(max(counts.values()), expected) * 1.18)
    hist.legend()
    for axes in (ax, hist):
        axes.grid(axis="y", alpha=0.22)
        axes.set_axisbelow(True)
    save(fig, "factoring")



def plot_nqueens():
    fig, (ax, hist) = plt.subplots(1, 2, figsize=(10.6, 4.1), layout="constrained")
    n_values = list(range(4, 11))
    observed = [load(f"count_nq{n}")["board_count"] for n in n_values]
    known = [2, 10, 4, 40, 92, 352, 724]
    ax.plot(n_values, known, "--", color=DIM, linewidth=1, label="OEIS A000170")
    ax.scatter(n_values, observed, facecolors="white", edgecolors=BLUE, zorder=3,
               s=46, linewidths=1.5, label="Carry contraction")
    for n, count in zip(n_values, observed):
        ax.annotate(str(count), (n, count), xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set(xticks=n_values, xlabel="Board size N", ylabel="Valid boards (log scale)", yscale="log",
           ylim=(1.36, max(known)*2), title="(a) Exact board counts")
    ax.legend(loc="upper left")
    data = load("sample_nq6")
    counts = Counter(map(tuple, data["samples"]))
    labels = [" ".join(map(str, b)) for b in sorted(counts)]
    bars = hist.bar(labels, [counts[b] for b in sorted(counts)], width=0.68, color=BLUE)
    expected = len(data["samples"]) / data["board_count"]
    hist.axhline(expected, color=ORANGE, linestyle="--", label=f"Uniform expectation {expected:g}")
    hist.bar_label(bars, padding=3)
    hist.set(xlabel="Board (zero-based column by row)", ylabel="Sample count",
             title="(b) 200 independent draws for N = 6",
             ylim=(0, 1.28 * max(max(counts.values()), expected)))
    hist.legend(loc="upper right")
    for axes in (ax, hist):
        axes.grid(axis="y", alpha=0.22)
        axes.set_axisbelow(True)
    save(fig, "nqueens")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    figures = {"gates": plot_gates, "factoring": plot_factoring, "nqueens": plot_nqueens}
    parser.add_argument("figure", choices=figures)
    figures[parser.parse_args().figure]()


if __name__ == "__main__":
    main()
