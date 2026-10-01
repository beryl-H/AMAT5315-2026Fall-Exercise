#!/usr/bin/env python3
"""Prepare base and ripple-perturbed input fields for the sensitivity runs.

For each case (taylor-green n=64 nu=0.1, random n=128 nu=0.004) the `field`
binary generates the base velocity field. The lecture ripple

    delta omega(x, y) = -7e-5 * M * cos(3x) cos(4y),   M = max |u|, |v|

is added to the INITIAL VORTICITY. Because the solver computes vorticity
from velocity (omega = d/dx v - d/dy u) the ripple is injected through the
solver's own stream-function pairing (u = dpsi/dy, v = -dpsi/dx,
omega = -laplacian psi): with A = -7e-5*M,

    psi  = (A/25) cos(3x) cos(4y)          (kappa^2 = 3^2 + 4^2 = 25)
    du   = dpsi/dy  = -(4A/25) cos(3x) sin(4y)
    dv   = -dpsi/dx =  (3A/25) sin(3x) cos(4y)

so vorticity_from_velocity(u+du, v+dv) = omega_base + delta omega exactly
(both ripple modes (3,4) are inside the two-thirds band of both grids).
Grid convention of src/field.rs: u[j*n + i], x_i = 2 pi i / n (i axis),
y_j = 2 pi j / n (j axis). The solver is not modified; it only receives
different stdin.

Run from week4/scripts/:

    ../target/release/field is invoked directly; writes
    artifacts/sensitivity/<case>/field-base.json and field-ripple.json
"""

import json
import subprocess
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "sensitivity"

CASES = {
    "taylor-green": ["taylor-green", "--n", "64"],
    "random": ["random", "--n", "128", "--seed", "2026", "--k-min", "2",
               "--k-max", "6"],
}
RIPPLE_AMP = -7e-5  # delta omega = RIPPLE_AMP * M * cos(3x) cos(4y)


def ripple_velocity(n, A):
    du = [0.0] * (n * n)
    dv = [0.0] * (n * n)
    for j in range(n):
        y = 2.0 * math.pi * j / n
        for i in range(n):
            x = 2.0 * math.pi * i / n
            q = j * n + i
            du[q] = -(4.0 * A / 25.0) * math.cos(3.0 * x) * math.sin(4.0 * y)
            dv[q] = (3.0 * A / 25.0) * math.sin(3.0 * x) * math.cos(4.0 * y)
    return du, dv


for case, args in CASES.items():
    out = OUT / case
    out.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run([str(ROOT / "target" / "release" / "field"), *args],
                         capture_output=True, text=True, check=True).stdout
    field = json.loads(raw)
    n = field["n"]
    M = max(max(abs(v) for v in field["u"]), max(abs(v) for v in field["v"]))
    A = RIPPLE_AMP * M
    du, dv = ripple_velocity(n, A)
    rip = dict(field)
    rip["u"] = [u + d for u, d in zip(field["u"], du)]
    rip["v"] = [v + d for v, d in zip(field["v"], dv)]
    (out / "field-base.json").write_text(json.dumps(field))
    (out / "field-ripple.json").write_text(json.dumps(rip))
    print(f"{case}: n = {n}, M = {M:.6f}, A = {A:.6e}  "
          f"(|delta omega|_max = {abs(A):.3e})")
