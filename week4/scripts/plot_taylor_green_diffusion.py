#!/usr/bin/env python3
"""Diffusive-stability evidence for the Taylor-Green run (Week 4 Part 3).

Reads the traces of artifacts/taylor-green-dt-0.032 and -0.033 (nothing is
re-run) and draws E(t) on a log y-axis against the exact Taylor-Green decay
E(t) = 0.25 exp(-4 nu t) (nu = 0.1). The lecture prediction is
kappa_max^2 = 2*floor(64/3)^2 = 882, dt_crit = 2.7853/(0.1*882) = 0.0316:
dt = 0.032 (0.13% above) must ride just outside the stability region with
per-step growth |R| = 1.053 -- too slow to lift a 1e-16 roundoff seed above
visibility in 250 steps -- while dt = 0.033 (4.5% above, |R| = 1.206)
blows up. Marks enstrophy as the leading indicator (Z leaves Z = 2E first).

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_taylor_green_diffusion.py

Saves week4/evidence/taylor-green-diffusion.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NU = 0.1

RUNS = [
    ("taylor-green-dt-0.032", "RK4 dt = 0.032 (reaches t = 8)", "tab:blue", "-"),
    ("taylor-green-dt-0.033", "RK4 dt = 0.033 (unstable)", "tab:red", "-"),
]

fig, ax = plt.subplots(figsize=(9.0, 5.6))
summary = {}
for folder, label, color, ls in RUNS:
    rows = [l.split("\t") for l in (ROOT / "artifacts" / folder / "trace.tsv")
            .read_text().splitlines()[1:] if l.strip()]
    t = np.array([float(r[0]) for r in rows])
    E = np.array([float(r[1]) for r in rows])
    Z = np.array([float(r[2]) for r in rows])
    finite = np.isfinite(E) & np.isfinite(Z)
    stop = None if finite.all() else t[~finite][-1]
    t, E, Z = t[finite], E[finite], Z[finite]
    ax.semilogy(t, E, ls, color=color, lw=1.8, label=label)
    ax.semilogy(t, Z, ls, color=color, lw=1.0, alpha=0.45,
                label=f"enstrophy Z, {label.split(' (')[0].lower()}")
    if stop is not None:
        ax.axvline(stop, color=color, lw=0.9, ls=":", alpha=0.8)
        ax.annotate(f"stops at t = {stop:g}", xy=(stop, 1e2),
                    xytext=(stop - 2.6, 1e2), fontsize=8.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
    summary[folder] = {"t": t.tolist(), "E": E.tolist(), "Z": Z.tolist(),
                       "stopping_time": stop}
    print(f"{folder}: {len(t)} finite samples"
          + (f", stops at t = {stop:g}" if stop is not None else ", reaches t = 8"))

tex = np.linspace(0.0, 8.0, 400)
ax.semilogy(tex, 0.25 * np.exp(-4 * NU * tex), "k--", lw=1.2,
            label=r"exact $E = 0.25\,e^{-4\nu t}$")
ax.set_xlabel("t")
ax.set_ylabel("E(t), Z(t) (log scale)")
ax.set_ylim(3e-3, 3e2)
ax.set_title("Taylor-Green, n = 64, RK4, nu = 0.1: diffusive stability limit\n"
             r"lecture prediction $\kappa_{max}^2 = 882$, "
             r"$dt_{crit} = 2.7853/(0.1 \cdot 882) = 0.0316$"
             "  (measured boundary in (0.032, 0.033))")
ax.legend(loc="lower left", fontsize=8.5)
ax.grid(alpha=0.25, which="both")

out = ROOT / "evidence" / "taylor-green-diffusion.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
(ROOT / "evidence" / "taylor-green-diffusion.json").write_text(
    __import__("json").dumps(summary, indent=1))
print(f"wrote {out}")
