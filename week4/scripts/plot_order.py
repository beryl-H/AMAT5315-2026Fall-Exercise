#!/usr/bin/env python3
"""RK4 temporal convergence on the Taylor-Green vortex (n = 8, nu = 0.5).

Reads the final (t = 2) velocity snapshots of the runs stored under
artifacts/order/rk4-dt{0.4,0.25,0.2} (nothing is re-run) and computes the
relative L2 error of the velocity field against the exact course solution

    u = cos(x) sin(y) exp(-2 nu t),   v = -sin(x) cos(y) exp(-2 nu t),

computed analytically here (and cross-checked against the `field` binary).
On the 8x8 grid the Taylor-Green field and all products of the nonlinear
term lie inside the two-thirds band (floor(8/3) = 2), so the spectral
spatial operator is exact and the error is purely temporal: RK4 error
~ dt^4. Plots error vs dt on log-log axes with a linear fit in log-log
space, slope labelled in the legend.

Run from week4/scripts/:

    MPLCONFIGDIR=/tmp/matplotlib /tmp/lineplotenv/bin/python plot_order.py

Saves week4/evidence/order.png and week4/evidence/order.json.
"""

from pathlib import Path
import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORDER = ROOT / "artifacts" / "order"
N, NU, T = 8, 0.5, 2.0

# analytic exact field at t = 2 on the periodic [0, 2pi)^2 grid
x = 2.0 * np.pi * np.arange(N) / N
X, Y = np.meshgrid(x, x)
decay = np.exp(-2.0 * NU * T)
u_exact = np.cos(X) * np.sin(Y) * decay
v_exact = -np.sin(X) * np.cos(Y) * decay
norm_exact = np.sqrt(((u_exact**2).sum() + (v_exact**2).sum()))

# cross-check: the `field` binary at n = 8, nu = 0.5, t = 2
raw = subprocess.run(
    [str(ROOT / "target" / "release" / "field"),
     "taylor-green", "--n", str(N), "--nu", str(NU), "--t", str(T)],
    capture_output=True, text=True, check=True).stdout
f = json.loads(raw)
check = (np.abs(np.asarray(f["u"]).reshape(N, N) - u_exact).max()
         + np.abs(np.asarray(f["v"]).reshape(N, N) - v_exact).max())
print(f"exact-formula vs `field taylor-green --t 2` max abs diff: {check:.3e}")
assert check < 1e-14

dts, errs = [], []
for dt in ("0.4", "0.25", "0.2"):
    run = json.loads((ORDER / f"rk4-dt{dt}" / "run.json").read_text())
    frames = [json.loads(l) for l in
              (ORDER / f"rk4-dt{dt}" / "fields.jsonl").read_text().splitlines()
              if l.strip()]
    last = frames[-1]
    assert abs(last["t"] - T) < 1e-9, f"dt = {dt}: last frame at t = {last['t']}"
    u = np.asarray(last["u"]).reshape(N, N)
    v = np.asarray(last["v"]).reshape(N, N)
    err = np.sqrt((((u - u_exact)**2).sum() + ((v - v_exact)**2).sum())) / norm_exact
    dts.append(run["dt"])
    errs.append(err)
    print(f"dt = {run['dt']:<5} steps = {last['step']:>3}: relative L2 error = {err:.6e}")

dts, errs = np.asarray(dts), np.asarray(errs)
slope, intercept = np.polyfit(np.log(dts), np.log(errs), 1)
print(f"log-log fit: slope = {slope:.3f} (RK4 expectation: 4)")

summary = {
    "n": N, "nu": NU, "t": T, "method": "rk4",
    "dt": dts.tolist(), "relative_l2_error": errs.tolist(),
    "fit_slope": float(slope), "fit_intercept": float(intercept),
    "exact_vs_field_binary_maxabs": float(check),
}
(ROOT / "evidence" / "order.json").write_text(json.dumps(summary, indent=1))

fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.loglog(dts, errs, "o", ms=8, color="tab:blue",
          label="measured relative L2 error at t = 2")
ts = np.linspace(0.9 * dts.min(), 1.1 * dts.max(), 50)
ax.loglog(ts, np.exp(intercept) * ts**slope, "k--", lw=1.2,
          label=rf"log-log fit: slope = {slope:.2f}")
ref = errs[-1] * (ts / dts.min())**4
ax.loglog(ts, errs[0] * (ts / dts.max())**4, color="tab:gray", lw=0.9,
          ls=":", label=r"reference slope 4 ($\sim dt^4$, RK4)")
ax.set_xlabel("time step dt")
ax.set_ylabel(r"$\| (u,v)_{run} - (u,v)_{exact} \|_2 \,/\, \| (u,v)_{exact} \|_2$")
ax.set_title("RK4 temporal convergence, Taylor-Green n = 8, nu = 0.5, t = 2\n"
             "(spectral spatial error absent: field and products inside the"
             " two-thirds band)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

out = ROOT / "evidence" / "order.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out} and evidence/order.json")
