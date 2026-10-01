#!/usr/bin/env python3
"""Richardson-based timestep selection for the random-flow convergence study.

Uses the retained final omega fields (artifacts/convergence/, t = 2) and the
measured errors already stored in evidence/convergence.json. Nothing is
re-run.

1. Fourth-order Richardson estimate of the relative temporal error of the
   dt = 0.01 solution from the coarse/fine pair (ratio exactly 2):

       e_h = ||omega_0.02 - omega_0.01|| / ((2^4 - 1) ||omega_0.01||)

2. Prediction e(dt) = e_h (dt / 0.01)^4 for the candidate steps.
3. Selection: the largest candidate with predicted error < 5e-6.
4. Figure: measured relative omega errors vs dt (log-log), the log-log fit
   of the measured series, the Richardson prediction curve, the 5e-6
   tolerance, and the selected timestep marked.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_convergence.py

Saves week4/evidence/convergence.png (updates evidence/convergence.json in
place with the Richardson data).
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONV = ROOT / "artifacts" / "convergence"

data = json.loads((ROOT / "evidence" / "convergence.json").read_text())


def omega_at2(dt):
    frames = [json.loads(l) for l in
              (CONV / f"dt-{dt}" / "fields.jsonl").read_text().splitlines()
              if l.strip()]
    last = frames[-1]
    assert abs(last["t"] - 2.0) < 1e-9
    return np.asarray(last["omega"])


# 1. Richardson estimate from the retained final fields (dt ratio exactly 2)
w_coarse, w_fine = omega_at2("0.02"), omega_at2("0.01")
rich = float(np.linalg.norm(w_coarse - w_fine)
             / ((2.0**4 - 1.0) * np.linalg.norm(w_fine)))
print(f"Richardson estimate e_h (relative error of dt = 0.01): {rich:.6e}")

# 2. fourth-order predictions; 3. selection below the 5e-6 tolerance
TOL = 5e-6
candidates = (0.02, 0.0125, 0.01)
pred = {dt: rich * (dt / 0.01)**4 for dt in candidates}
selected = max(dt for dt in candidates if pred[dt] < TOL)
print("predicted vs measured errors:")
for dt in candidates:
    print(f"  dt = {dt:<7} predicted {pred[dt]:.4e}   measured "
          f"{[r['relative_omega_error_vs_ref'] for r in data['runs'] if r['dt'] == dt][0]:.4e}")
print(f"selected dt = {selected}")

# update the stored evidence with the Richardson block
data["richardson"] = {
    "pair": [0.02, 0.01],
    "e_h_dt0.01": rich,
    "predicted": {str(dt): pred[dt] for dt in candidates},
    "tolerance": TOL,
    "selected_dt": selected,
}
(ROOT / "evidence" / "convergence.json").write_text(json.dumps(data, indent=1))

# 4. figure -------------------------------------------------------------------
runs = {r["dt"]: r["relative_omega_error_vs_ref"] for r in data["runs"]
        if not r["is_reference"]}
dts = np.asarray(sorted(runs))
errs = np.asarray([runs[d] for d in dts])
slope, intercept = np.polyfit(np.log(dts), np.log(errs), 1)

fig, ax = plt.subplots(figsize=(8.5, 5.8))
ax.loglog(dts, errs, "o", ms=8, color="tab:blue", zorder=5,
          label="measured relative omega error (vs dt = 0.0025)")
ts = np.linspace(0.008, 0.026, 100)
ax.loglog(ts, np.exp(intercept) * ts**slope, "k--", lw=1.2,
          label=f"log-log fit of measured errors: slope = {slope:.2f}")
ax.loglog(ts, rich * (ts / 0.01)**4, color="tab:red", lw=1.2,
          label=rf"Richardson prediction $e_h\,(dt/0.01)^4$, $e_h$ = {rich:.2e}")
ax.axhline(TOL, color="tab:green", lw=1.0, ls=":",
           label=r"tolerance 5e-6")
ax.axvline(selected, color="tab:purple", lw=1.2, ls="-.")
ax.annotate(f"selected dt = {selected}\n"
            f"predicted {pred[selected]:.2e} < 5e-6\n"
            f"measured  {runs[selected]:.2e}",
            xy=(selected, runs[selected]), xytext=(0.0105, 3e-5),
            fontsize=9, color="tab:purple",
            arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.0))
for dt in candidates:
    ax.annotate("", xy=(dt, pred[dt]), xytext=(dt, runs[dt]),
                arrowprops=dict(arrowstyle="<->", color="tab:red", lw=0.7,
                                alpha=0.6))
ax.set_xlabel("time step dt")
ax.set_ylabel(r"$\|\omega_{dt} - \omega_{0.0025}\|_2 \,/\, \|\omega_{0.0025}\|_2$"
              "  at t = 2 (log scale)")
ax.set_ylim(1e-6, 2e-4)
ax.set_title("Random-band flow, n = 128, RK4, nu = 0.004, t = 2:\n"
             "Richardson timestep selection (largest dt with predicted"
             " error below 5e-6)")
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(alpha=0.3, which="both")

out = ROOT / "evidence" / "convergence.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out} (convergence.json updated with the Richardson block)")
