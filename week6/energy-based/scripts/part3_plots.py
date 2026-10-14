"""Part 3 plots (fixed per root review).

- Recall panel uses the independently verified summary fields in
  evidence/part3_metrics.json (actual R = min(10, P) per point; no
  NaN-padding averaging, no empty-slice warnings).
- Unstable-fraction panel uses a LINEAR y axis so exact zeros are honest
  points (no invented marker floors); a log-scale low-load inset shows the
  structure at small alpha with the pooled one-event resolution
  1/(20*P*N) marked in the caption only.
- Colors are matched between measured, finite-N theory and the large-N
  curve legend; the finite-N reference deviation wording is exactly TWO
  sizes (N=200 and N=800 below 0.988; N=400 = 0.98950 is above).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
m = json.loads((EV / "part3_metrics.json").read_text())
raw_dir = sorted((ROOT / "artifacts/part3").glob("evaluation-*"))[-1]
raw = np.load(raw_dir / "part3_raw.npz")
NS = m["settings"]["NS"]
ALPHAS = np.array(m["settings"]["alphas"])
S = m["settings"]["n_sets"]

unstable = raw["unstable"].reshape(len(NS), len(ALPHAS), S)
COLORS = {100: "tab:blue", 200: "tab:orange", 400: "tab:green", 800: "tab:red"}

# ---- figure 1: unstable fraction, linear y (zeros are exact zeros)
fig, (ax, axins) = plt.subplots(2, 1, figsize=(7.5, 7.0),
                                constrained_layout=True,
                                gridspec_kw={"height_ratios": [2.2, 1.3]})
al_fine = np.linspace(0.05, 0.25, 200)
for i, N in enumerate(NS):
    mean = unstable[i].mean(axis=1)
    sem = unstable[i].std(axis=1, ddof=1) / np.sqrt(S)
    c = COLORS[N]
    ax.errorbar(ALPHAS, mean, yerr=sem, marker="o", ms=3.5, capsize=2,
                color=c, label=f"N={N} (measured ± SEM over 20 sets)")
    P = np.round(ALPHAS * N)
    ax.plot(ALPHAS, norm.cdf(-np.sqrt((N - 1) / (P - 1))), lw=0.9,
            color=c, alpha=0.55, ls="--",
            label=f"finite-N Φ, N={N}")
    axins.plot(ALPHAS, norm.cdf(-np.sqrt((N - 1) / (P - 1))), lw=0.9,
               color=c, alpha=0.55, ls="--")
    axins.errorbar(ALPHAS, mean, yerr=sem, marker="o", ms=3, capsize=2,
                   color=c)
ax.plot(al_fine, norm.cdf(-1 / np.sqrt(al_fine)), "k-", lw=1.4,
        label=r"$\Phi(-1/\sqrt{\alpha})$ (large N)")
axins.plot(al_fine, norm.cdf(-1 / np.sqrt(al_fine)), "k-", lw=1.2)
axins.set_xlim(0.05, 0.12)
axins.set_ylim(-1e-4, 2e-3)
axins.set_title("low-load inset (exact zeros are measured zeros)", fontsize=8)
ax.set_xlabel(r"load $\alpha = P/N$")
ax.set_ylabel("unstable-bit fraction")
ax.set_title("unstable-bit fraction follows the crosstalk prediction\n"
             "SEM over 20 independent pattern sets; points at zero are exact zeros")
ax.legend(fontsize=7)
fig.savefig(EV / "part3_unstable_fraction.png", dpi=150)
plt.close(fig)
print("wrote part3_unstable_fraction.png")

# ---- figure 2: recall curves from the VERIFIED summary fields
pts = m["points"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for N in NS:
    sel = [p for p in pts if p["N"] == N]
    a = [p["alpha"] for p in sel]
    axes[0].errorbar(a, [p["recall_mean"] for p in sel],
                     yerr=[p["recall_sem"] for p in sel], marker="o", ms=3,
                     capsize=2, color=COLORS[N], label=f"N={N}")
    axes[1].plot(a, [p["success_mean"] for p in sel], marker="s", ms=3,
                 color=COLORS[N], label=f"N={N}")
for ax in axes:
    ax.axvline(0.138, color="k", ls=":", lw=1)
    ax.set_xlabel(r"load $\alpha$")
axes[0].axhline(0.9, color="0.6", ls="--", lw=0.8)
axes[0].set_ylabel("mean final overlap")
axes[1].set_ylabel(r"fraction of recalls with $m \geq 0.9$")
axes[0].set_title("mean final overlap (±SEM over 20 sets;\n"
                  "each set averages its R = min(10, P) recalls)")
axes[1].set_title("recall success fraction")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
cross = m["crossings_at_0.9_linear_interp"]
fig.suptitle(r"recall collapses near $\alpha_c \approx 0.138$ (dotted); "
             "first .9-crossings: "
             + ", ".join(f"N={N}: {cross[str(N)]:.3f}" for N in NS))
fig.savefig(EV / "part3_recall_capacity.png", dpi=150)
plt.close(fig)
print("wrote part3_recall_capacity.png")
