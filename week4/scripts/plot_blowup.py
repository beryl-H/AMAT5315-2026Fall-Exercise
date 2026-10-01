#!/usr/bin/env python3
"""Week 4 lecture artifact: diffusive and advective stability boundaries.

Two panels on logarithmic energy axes (saved as one figure,
week4/evidence/blowup.png). Everything is read from stored run traces
(week4/artifacts/...); nothing is re-run.

Left panel -- Taylor-Green diffusive limit (n = 64, nu = 0.1):
RK4 dt = 0.032 reaches t = 8 on the exact decay E = 0.25 exp(-4 nu t);
RK4 dt = 0.033 rides outside the stability region for the corner mode
(kappa^2 = 882, dt_crit = 2.7853/(0.1*882) = 0.0316) and blows up from a
roundoff seed at t = 7.821, with enstrophy diverging first.

Right panel -- random-band flow advective limit (n = 128, nu = 0.004):
RK4 dt = 0.033 (largest stable tested) decays smoothly to t = 10; RK4
dt = 0.034 (smallest unstable tested) and Euler dt = 0.01 (negative
comparison: any imaginary eigenvalue lies outside |1+z| <= 1) blow up at
t = 1.87 and t = 1.16. Dotted vertical lines mark stopping times.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_blowup.py

Saves week4/evidence/blowup.png and week4/evidence/blowup.json.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def trace(folder):
    rows = [l.split("\t") for l in (ROOT / "artifacts" / folder / "trace.tsv")
            .read_text().splitlines()[1:] if l.strip()]
    t = np.array([float(r[0]) for r in rows])
    E = np.array([float(r[1]) for r in rows])
    Z = np.array([float(r[2]) for r in rows])
    finite = np.isfinite(E) & np.isfinite(Z)
    stop = None if finite.all() else t[~finite][-1]
    return t[finite], E[finite], Z[finite], stop


summary = {"taylor_green": {}, "random": {}}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.4))

# --- left panel: Taylor-Green diffusive limit --------------------------------
NU = 0.1
for folder, label, color, ls in [
    ("taylor-green-dt-0.032", "RK4 dt = 0.032 (stable)", "tab:blue", "-"),
    ("taylor-green-dt-0.033", "RK4 dt = 0.033 (unstable)", "tab:red", "-"),
]:
    t, E, Z, stop = trace(folder)
    axL.semilogy(t, E, ls, color=color, lw=1.8, label=label)
    axL.semilogy(t, Z, ls, color=color, lw=1.0, alpha=0.45,
                 label=f"enstrophy Z, dt = {folder.split('-')[-1]}")
    if stop is not None:
        axL.axvline(stop, color=color, lw=0.9, ls=":", alpha=0.8)
        axL.annotate(f"stops at t = {stop:g}", xy=(stop, 3e1),
                     xytext=(stop - 3.4, 3e1), fontsize=8.5, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
    summary["taylor_green"][folder] = {"t": t.tolist(), "E": E.tolist(),
                                       "Z": Z.tolist(), "stopping_time": stop}
    print(f"{folder}: {len(t)} finite samples"
          + (f", stops at t = {stop:g}" if stop is not None else ", reaches t = 8"))
tex = np.linspace(0.0, 8.0, 400)
axL.semilogy(tex, 0.25 * np.exp(-4 * NU * tex), "k--", lw=1.2,
             label=r"exact $E = 0.25\,e^{-4\nu t}$")
axL.set_xlabel("t")
axL.set_ylabel("E(t), Z(t) (log scale)")
axL.set_ylim(3e-3, 3e2)
axL.set_title("Taylor-Green, n = 64, RK4, nu = 0.1:\n"
              r"diffusive limit $\kappa_{max}^2 = 882$, $dt_{crit} = 0.0316$",
              fontsize=10.5)
axL.legend(loc="lower left", fontsize=8)
axL.grid(alpha=0.25, which="both")

# --- right panel: random-flow advective limit --------------------------------
for folder, label, color, ls, y_lab in [
    ("scan/dt-0.033", "RK4 dt = 0.033 (largest stable)", "tab:blue", "-", None),
    ("scan/dt-0.034", "RK4 dt = 0.034 (smallest unstable)", "tab:red", "-", 3e7),
    ("scan/euler-dt-0.010", "Euler dt = 0.01 (negative comparison)", "tab:green", "--", 3e5),
]:
    t, E, Z, stop = trace(folder)
    axR.semilogy(t, E, ls, color=color, lw=1.8, label=label)
    if stop is not None:
        axR.axvline(stop, color=color, lw=0.9, ls=":", alpha=0.8)
        axR.annotate(f"stops at t = {stop:g}", xy=(stop, y_lab),
                     xytext=(stop + 0.4, y_lab), fontsize=8.5, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
    summary["random"][folder] = {"t": t.tolist(), "E": E.tolist(),
                                 "stopping_time": stop}
    print(f"{folder}: {len(t)} finite samples"
          + (f", stops at t = {stop:g}" if stop is not None else ", reaches t = 10"))
axR.set_xlabel("t")
axR.set_ylabel("E(t) (log scale)")
axR.set_ylim(1e-2, 1e12)
axR.set_title("random-band flow, n = 128, RK4, nu = 0.004:\n"
              r"advective limit, measured boundary in $(0.033, 0.034)$",
              fontsize=10.5)
axR.legend(loc="center right", fontsize=8)
axR.grid(alpha=0.25, which="both")

fig.suptitle("Week 4: energy blow-up at the stability limit"
             " (left: diffusion-limited, right: advection-limited)",
             fontsize=12)

out = ROOT / "evidence" / "blowup.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
(ROOT / "evidence" / "blowup.json").write_text(json.dumps(summary, indent=1))
print(f"wrote {out} and evidence/blowup.json")
