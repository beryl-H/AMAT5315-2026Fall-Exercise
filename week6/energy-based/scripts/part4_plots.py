"""Part 4 plots from evidence/part4_metrics.json and raw point files."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
m = json.loads((EV / "part4_metrics.json").read_text())
raw_dir = sorted((ROOT / "artifacts/part4").glob("evaluation-*"))[-1]
PS = m["settings"]["PS"]
BETAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# ---- figure 1: softmax weights over the 100 stored patterns, query 0
p100 = next(p for p in m["points"] if p["P"] == 100)
w = p100["weights_q0"]  # target of query 0 is index 0
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for beta, ls in (("0.003", ":"), ("0.03", "--"), ("0.3", "-")):
    ws = np.array(w[beta])
    ax.plot(range(100), ws, ls, lw=1.2, label=rf"$\beta={beta}$")
    ax.plot([0], [ws[0]], "o", ms=6, color=ax.lines[-1].get_color())
    ax.annotate(f"target weight {ws[0]:.2f}", xy=(0, ws[0]),
                xytext=(12, ws[0]), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7))
ax.set_xlabel("stored pattern index")
ax.set_ylabel(r"softmax weight $\propto e^{\beta x^\mu\cdot x}$")
ax.set_title("softmax weights for the FIRST query (target = pattern 0), P=100")
ax.legend(fontsize=8)
fig.savefig(EV / "part4_softmax_weights.png", dpi=150)
plt.close(fig)
print("wrote part4_softmax_weights.png")

# ---- figure 2: P=100 gallery — targets, queries, modern outputs, classical
d = np.load(raw_dir / "point_P100.npz")
targets = d["targets"]; queries = d["queries"]
n_show = 4
rows = [("target", d["queries"] * 0)]  # placeholder replaced below
X = None
# reconstruct stored patterns? not needed: targets ARE stored patterns;
# re-load them from MNIST for exact display
from ebm.data import load_mnist_pair
from ebm.modern_hopfield import spinize
train_images, _ = load_mnist_pair(
    ROOT / "inputs/mnist/train-images-idx3-ubyte.gz",
    ROOT / "inputs/mnist/train-labels-idx1-ubyte.gz")
X = np.asarray(spinize(train_images[:100]))  # (784, 100)

grid = []
grid.append([X[:, int(targets[q])] for q in range(n_show)])              # targets
grid.append([queries[q] for q in range(n_show)])                          # queries
for beta in BETAS:
    grid.append([d[f"modern_out_beta{beta}"][q] for q in range(n_show)])  # modern
grid.append([d["classical_out"][q] for q in range(n_show)])               # classical

fig, axes = plt.subplots(len(grid), n_show, figsize=(1.5 * n_show, 1.5 * len(grid)),
                         constrained_layout=True)
labels = (["target"] + ["corrupted query"] +
          [f"modern β={b}" for b in BETAS] + ["classical recall"])
for r, row in enumerate(grid):
    for c in range(n_show):
        ax = axes[r, c]
        img = np.asarray(row[c]).reshape(28, 28)
        ax.imshow(img, interpolation="nearest", cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(labels[r], fontsize=7)
fig.suptitle("P=100: first four queries\nContinuous modern outputs across beta vs classical recall", fontsize=10)
fig.savefig(EV / "part4_gallery_P100.png", dpi=150)
plt.close(fig)
print("wrote part4_gallery_P100.png")

# ---- figure 3: success vs P for all betas + classical
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
for beta in BETAS:
    ys = [next(p for p in m["points"] if p["P"] == P)
          ["modern_success_frac"][str(beta)] for P in PS]
    ax.plot(PS, ys, marker="o", ms=4, label=rf"modern $\beta={beta}$")
ys = [next(p for p in m["points"] if p["P"] == P)["classical_success_frac"]
      for P in PS]
ax.plot(PS, ys, marker="s", ms=5, color="k", label="classical recall")
ax.axhline(0.95, color="0.6", ls="--", lw=0.8)
ax.set_xscale("log")
ax.set_xticks(PS, [str(p) for p in PS])
ax.set_xlabel("stored patterns P")
ax.set_ylabel("retrieval success fraction")
ax.set_title("modern retrieval survives loads where classical recall fails\n"
             "(100 queries per P; success = m≥0.95 with target AND target nearest)")
ax.legend(fontsize=8)
fig.savefig(EV / "part4_success_vs_P.png", dpi=150)
plt.close(fig)
print("wrote part4_success_vs_P.png")
