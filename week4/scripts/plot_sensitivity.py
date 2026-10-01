#!/usr/bin/env python3
"""Sensitivity to initial conditions of the two Week 4 flows.

Reads the stored snapshot pairs under artifacts/sensitivity/ (base vs
ripple-perturbed initial vorticity, RK4 dt = 0.01, t = 20, snapshots every
0.5; nothing is re-run) and draws the relative vorticity distance

    d(t) = ||omega_ripple(t) - omega_base(t)||_2 / ||omega_base(t)||_2

on a log y-axis for both cases. The ripple had initial size
|delta omega|_max = 7e-5 * M with M = max|u|, |v| (taylor-green: M = 1,
random: M = 2.035269), i.e. d(0) ~ 4e-5 for both cases.

Expected lecture picture: the laminar Taylor-Green vortex is linearly
stable -- the perturbation decays with the flow until it sinks below the
6-decimal storage resolution of fields.jsonl (dotted floor, a data artifact
not dynamics) -- while the random-band flow is chaotic: the distance grows
exponentially (fitted straight line, doubling time annotated) until it
saturates at the size of the flow itself.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_sensitivity.py

Saves week4/evidence/sensitivity.png and week4/evidence/sensitivity.json.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SENS = ROOT / "artifacts" / "sensitivity"

CASES = [
    ("taylor-green", "Taylor-Green (n = 64, nu = 0.1)", "tab:blue"),
    ("random", "random-band flow (n = 128, nu = 0.004)", "tab:red"),
]


def load(case, var):
    frames = [json.loads(l) for l in
              (SENS / case / var / "fields.jsonl").read_text().splitlines()
              if l.strip()]
    return {round(f["t"], 6): np.asarray(f["omega"]) for f in frames}


summary = {}
fig, ax = plt.subplots(figsize=(9.5, 5.8))
for case, label, color in CASES:
    base, rip = load(case, "base"), load(case, "ripple")
    times = sorted(set(base) & set(rip))
    d = []
    for t_ in times:
        wb, wr = base[t_], rip[t_]
        d.append(np.linalg.norm(wr - wb) / np.linalg.norm(wb))
    d = np.asarray(d)
    t = np.asarray(times)
    # 6-decimal storage floor: rounding two fields to 1e-6 gives an
    # absolute difference of ~1.5e-6; relative to ||omega_base(t)||, which
    # itself decays, the floor RISES with time. Below it the stored pair can
    # be identical (d = 0): mask so the log plot breaks instead of stemming.
    floor = 1.5e-6 / np.array([np.linalg.norm(base[t_]) for t_ in times])
    visible = d > floor
    d_plot = np.where(visible, d, np.nan)
    ax.semilogy(t, d_plot, "-o", color=color, ms=3.5, lw=1.4, label=label)
    ax.semilogy(t[~visible], floor[~visible], ":", color=color, lw=0.9,
                alpha=0.8)
    summary[case] = {"t": t.tolist(), "d": d.tolist(),
                     "storage_floor": floor.tolist(),
                     "below_storage_floor_from_t": (float(t[~visible][0])
                                                   if (~visible).any() else None)}
    print(f"{case}: d(0) = {d[0]:.3e}, d(20) = {d[-1]:.3e}, "
          f"max d = {d.max():.3e} at t = {t[d.argmax()]:g}, "
          f"below storage floor from t = "
          f"{summary[case]['below_storage_floor_from_t']}")

# exponential fit for the random flow in its clean growth window
d = np.asarray(summary["random"]["d"])
t = np.asarray(summary["random"]["t"])
lo, hi = 4.0, 12.0                      # growth window (below saturation)
floor_r = np.asarray(summary["random"]["storage_floor"])
m = (t >= lo) & (t <= hi) & (d > floor_r)  # above the storage noise floor
lam, loga = np.polyfit(t[m], np.log(d[m]), 1)
fit = np.exp(loga + lam * t)
ax.semilogy(t, fit, "k--", lw=1.2,
            label=rf"fit $d \propto e^{{\lambda t}}$, $\lambda$ = {lam:.3f},"
                  f" doubling time = {np.log(2)/lam:.2f}")
summary["random"]["lambda_fit"] = lam
summary["random"]["doubling_time"] = float(np.log(2) / lam)
summary["random"]["fit_window"] = [lo, hi]
print(f"random fit over t in [{lo}, {hi}]: lambda = {lam:.4f}, "
      f"doubling time = {np.log(2)/lam:.2f}")

# horizontal reference: full decorrelation
ax.axhline(1.0, color="gray", lw=0.7, ls="-.", alpha=0.6)
ax.text(12.2, 1.15, "size of the unperturbed flow", fontsize=8,
        color="gray")
ax.text(4.0, 4e-7, "dotted: 6-decimal storage floor\n"
        "(rounded snapshots identical below it)", fontsize=7.5,
        color="tab:blue")

ax.set_xlabel("t")
ax.set_ylabel(r"$\|\omega_{ripple}(t) - \omega_{base}(t)\|_2\,/\,\|\omega_{base}(t)\|_2$"
              "  (log scale)")
ax.set_ylim(1e-7, 3)
ax.set_title("Sensitivity to a vorticity ripple "
             r"$\delta\omega = -7\cdot10^{-5} M\cos 3x\cos 4y$"
             " (RK4, dt = 0.01): laminar decay vs chaotic growth")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(alpha=0.25, which="both")

out = ROOT / "evidence" / "sensitivity.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
(ROOT / "evidence" / "sensitivity.json").write_text(json.dumps(summary, indent=1))
print(f"wrote {out} and evidence/sensitivity.json")
