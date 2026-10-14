"""VERIFY2: paired CD-surrogate / held-out one-step reconstruction curves
for CD-1 and CD-10 from the tracked training metrics JSON."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
m = json.loads((EV / "part2_training_metrics.json").read_text())

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col",
                         constrained_layout=True)
for col, run in enumerate(("cd1", "cd10")):
    r = m["runs"][run]
    ep = [e["epoch"] for e in r["epochs"]]
    rec = [e["recon_hamming"] for e in r["epochs"]]
    sur = [e["cd_surrogate"] for e in r["epochs"]]
    axes[0, col].plot(ep, rec, marker="o", ms=3, color="tab:blue")
    axes[0, col].axhline(r["initial_recon"], color="0.6", ls="--", lw=0.8,
                         label="initialization")
    axes[0, col].set_title(f"{run} (k={r['k']}), {r['total_time_s']:.0f}s")
    axes[0, col].set_ylabel("held-out one-step recon\nHamming fraction")
    axes[0, col].legend(fontsize=8)
    axes[1, col].plot(ep[1:], sur[1:], marker="s", ms=3, color="tab:orange")
    axes[1, col].set_ylabel("CD surrogate (batch mean)")
    axes[1, col].set_xlabel("epoch")
fig.suptitle("VERIFY2: training diagnostics — surrogate need not decrease; "
             "reconstruction should improve from initialization")
fig.savefig(EV / "part2_learning_curves.png", dpi=150)
print("wrote part2_learning_curves.png")
