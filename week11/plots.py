"""Render the required Week 11 figures from saved, validated JSON runs."""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=["gates"])
    parser.parse_args()
    plot_gates()


if __name__ == "__main__":
    main()
