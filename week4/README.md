# Week 4 — regeneration guide

Every committed file in this folder is produced by the commands below, run
from `week4/` on a clean clone. `week4/artifacts/` is gitignored: it holds
only generated solver output (and one course-provided input file), and every
evidence script regenerates what it needs — this guide lists those commands
too so the evidence is reproducible end to end.

## 1. Prerequisites

```sh
# Rust 1.75+ (crate: advdiff, edition 2021)
cargo build --release
cargo test                      # unit + integration tests (src, tests/)

# Python 3 with numpy and matplotlib for the scripts
python3 -m pip install numpy matplotlib
# (committed locally with /tmp/lineplotenv/bin/python; any interpreter
#  with numpy + matplotlib works, e.g. python3 below)
```

## 2. The two solver pipelines

Both pipelines pipe the initial field from the `field` binary into the
`fluid` solver; the seed and parameters fully determine the data.

```sh
# Taylor-Green vortex: n = 64, nu = 0.1, t = 1
target/release/field taylor-green --n 64 | \
  target/release/fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
    --out artifacts/taylor-green

# Random-band flow: n = 128, nu = 0.004, seed 2026, wavenumbers 2..6, t = 10
target/release/field random --n 128 --seed 2026 --k-min 2 --k-max 6 | \
  target/release/fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
    --out artifacts/random
```

`artifacts/taylor-green/exact-t1.json` is the course-provided exact field at
t = 1; copy it in before running `scripts/compare_taylor_green.py` (the
script verifies it against the closed form, so a wrong file fails loudly).

## 3. Further solver runs used as evidence inputs (gitignored outputs)

```sh
# Stability-limit scan (artifacts/scan/): identical piped random field
for d in 0.020 0.025 0.030 0.031 0.032 0.033 0.034 0.036 0.038 0.040; do
  target/release/field random --n 128 --seed 2026 --k-min 2 --k-max 6 | \
    target/release/fluid --method rk4 --nu 0.004 --dt $d --t-end 10 --every 0.1 \
      --out artifacts/scan/dt-$d > artifacts/scan/dt-$d/trace.tsv
done
target/release/field random --n 128 --seed 2026 --k-min 2 --k-max 6 | \
  target/release/fluid --method euler --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
    --out artifacts/scan/euler-dt-0.010 > artifacts/scan/euler-dt-0.010/trace.tsv

# Taylor-Green diffusive pair (artifacts/taylor-green-dt-*/):
for d in 0.032 0.033; do
  target/release/field taylor-green --n 64 | \
    target/release/fluid --method rk4 --nu 0.1 --dt $d --t-end 8 --every 0.5 \
      --out artifacts/taylor-green-dt-$d > artifacts/taylor-green-dt-$d/trace.tsv
done

# Sensitivity runs (artifacts/sensitivity/): ripple-perturbed initial fields
python3 scripts/prepare_sensitivity.py          # writes field-base/-ripple.json
for case in taylor-green random; do
  nu=0.1; [ $case = random ] && nu=0.004
  for var in base ripple; do
    in=artifacts/sensitivity/$case/field-$var.json
    target/release/fluid --method rk4 --nu $nu --dt 0.01 --t-end 20 --every 0.5 \
      --out artifacts/sensitivity/$case/$var < $in \
      > artifacts/sensitivity/$case/$var/trace.tsv
  done
done

# RK4 temporal order (artifacts/order/): Taylor-Green n = 8, nu = 0.5, t = 2
for d in 0.4 0.25 0.2; do
  target/release/field taylor-green --n 8 | \
    target/release/fluid --method rk4 --nu 0.5 --dt $d --t-end 2 --every 2 \
      --out artifacts/order/rk4-dt$d > artifacts/order/rk4-dt$d/trace.tsv
done

# Convergence vs reference (artifacts/convergence/): random flow to t = 2
for d in 0.02 0.0125 0.01 0.0025; do
  target/release/field random --n 128 --seed 2026 --k-min 2 --k-max 6 | \
    target/release/fluid --method rk4 --nu 0.004 --dt $d --t-end 2 --every 2 \
      --out artifacts/convergence/dt-$d > artifacts/convergence/dt-$d/trace.tsv
done
```

Unstable scan runs (dt >= 0.034) print a non-finite energy line and exit 1 by
design — that stopping line is part of the evidence.

## 4. Scripts, in order, with the files they write

Rust binaries (write JSON evidence directly; run from `week4/`):

| command | writes |
|---|---|
| `cargo run --release --bin measure_growth -- evidence/line_growth.json` | `evidence/line_growth.json` |
| `cargo run --release --bin run_accuracy -- evidence/line_accuracy.json` | `evidence/line_accuracy.json` |
| `cargo run --release --bin run_convergence -- evidence/line_convergence.json` | `evidence/line_convergence.json` |
| `cargo run --release --bin run_gaussian -- evidence/line_evolution.json` | `evidence/line_evolution.json` |
| `cargo run --release --bin derivatives_check` | (check only, no file) |

Python scripts (`python3 scripts/<name>.py` from `week4/`):

| script | writes |
|---|---|
| `verify_two_mode.py` | (check only; reference for the `tests/fluid.rs` contract) |
| `plot_line_accuracy.py` | `evidence/line-accuracy.png` |
| `plot_line_stability.py` | `evidence/line_stability.png` |
| `compare_taylor_green.py` | `evidence/taylor-green.png` |
| `plot_random_vorticity.py` | `evidence/random.png` |
| `plot_blowup.py` | `evidence/blowup.png`, `evidence/blowup.json` |
| `plot_taylor_green_diffusion.py` | `evidence/taylor-green-diffusion.png`, `evidence/taylor-green-diffusion.json` |
| `plot_sensitivity.py` | `evidence/sensitivity.png`, `evidence/sensitivity.json` |
| `plot_order.py` | `evidence/order.png`, `evidence/order.json` |
| `plot_convergence.py` | `evidence/convergence.json`, `evidence/convergence.png` |

## 5. Evidence files and the command that produces each

| file in `week4/evidence/` | produced by |
|---|---|
| `line_growth.json` | `measure_growth` (§4) |
| `line_accuracy.json` | `run_accuracy` (§4) |
| `line_convergence.json` | `run_convergence` (§4) |
| `line_evolution.json` | `run_gaussian` (§4) |
| `line-accuracy.png` | `python3 scripts/plot_line_accuracy.py` |
| `line_stability.png` | `python3 scripts/plot_line_stability.py` (needs `line_growth.json`, `line_evolution.json`) |
| `taylor-green.png` | `python3 scripts/compare_taylor_green.py` (needs `artifacts/taylor-green/` + `exact-t1.json`) |
| `random.png` | `python3 scripts/plot_random_vorticity.py` (needs `artifacts/random/`) |
| `blowup.png`, `blowup.json` | `python3 scripts/plot_blowup.py` (needs `artifacts/scan/`, `artifacts/taylor-green-dt-*/`) |
| `taylor-green-diffusion.png`, `.json` | `python3 scripts/plot_taylor_green_diffusion.py` (needs `artifacts/taylor-green-dt-*/`) |
| `sensitivity.png`, `sensitivity.json` | `python3 scripts/plot_sensitivity.py` (needs §3 sensitivity runs) |
| `order.png`, `order.json` | `python3 scripts/plot_order.py` (needs §3 order runs) |
| `convergence.json`, `convergence.png` | `python3 scripts/plot_convergence.py` (needs §3 convergence runs) |
