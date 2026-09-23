# Week 4 Field & Fluid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `field` and `fluid` on top of the existing `advdiff` integrators: 2D spectral machinery (DFT2, per-axis Nyquist derivatives, two-thirds dealias), the 2D incompressible vorticity equation with Taylor–Green verification, and two clap binaries whose `--method` selects Euler/Midpoint/RK4 through the shared `Integrator` trait.

**Architecture:** `src/field.rs` — 2D state (`Vec<Complex<f64>>`, row-major, spectral index `[ky*n + kx]`), row-column DFT2 (1/n per axis), per-axis wavenumbers with the independent no-travel rule on first derivatives, Laplacian with full κ², two-thirds dealias, initial-condition generators (Hermitian single mode, Taylor–Green), linear 2D advection–diffusion rate + closed-form mode exponential. `src/fluid.rs` — pairing A stream function (ψ̂ = −ω̂/κ², mean untouched), velocity (u_x = i·ky·ψ̂, u_y = −i·kx·ψ̂ with per-axis Nyquist rule), pseudospectral nonlinear term (physical products, dealiased), full rate, Parseval energy/enstrophy. Binaries `src/bin/field.rs`, `src/bin/fluid.rs` (clap, hand-rolled JSON, no new deps besides clap).

**Tech Stack:** advdiff crate, num-complex, clap 4.4 (week3 pin), hand-rolled JSON writers.

**Spec:** `week4/field.design.toml`, `week4/fluid.design.toml`, `week4/SPEC.md`.

## Global Constraints

- History: red commit (test watched failing for the right reason) → green commit per task.
- All rate functions honor the Scaling convention (real `y * a` scaling only; complex products live in rate functions).
- Two-thirds rule: zero modes with |kx| > ⌊n/3⌋ or |ky| > ⌊n/3⌋ — on the vorticity and after every product.
- No-travel rule per axis, first derivatives only; Laplacian keeps full κ² (moot after dealias).
- `cargo test` green and warning-free from `week4/` after every green commit.
- Evidence runs use the usage lines from the design tomls; artifacts committed under `week4/runs/`.

---

### Task 1: field library (DFT2, derivatives, dealias, generators, linear rate)

**Files:** Create `week4/tests/field.rs`, `week4/src/field.rs`; modify `week4/src/lib.rs`.
**Tests (red):** wavenumber/Nyquist rule; DFT2↔IDFT2 round trip; plane wave → Hermitian mode pair; in-band single-mode derivatives (i·kx, i·ky, −κ²); per-axis Nyquist no-travel (x-Nyquist: d/dx=0, d/dy=i·ky; symmetric; Laplacian full κ²); dealias band (⌊32/3⌋=10, boundary kept, ±11 zeroed); Hermitian-mode generator (dealiased, physical 2cos(kx·x+ky·y)); linear rate vs exact exponential through `advance`+RK4 (1e−10); Taylor–Green generator matches 2 sin x sin y samples.
**Green:** functions per Architecture above; `advdiff_rate(n, cx, cy, nu, u_hat)` = coefficient (−i·cx·kx·[¬x-Nyq] − i·cy·ky·[¬y-Nyq] − ν·κ²)·û; `advdiff_exponential(...)` = exp of the same per-mode factor; `hermitian_mode`, `taylor_green_spectral` = dealias(dft2(2 sin x sin y)).
**Commits:** `test: add failing 2D field machinery tests (red)` / `feat: implement 2D spectral field machinery (green)`.

### Task 2: field binary

**Files:** Modify `week4/Cargo.toml` (clap = "=4.4.18", derive); create `week4/src/bin/field.rs`.
**Tests (red):** bin unit test — `exact_solution` equals `hermitian_mode` scaled by `advdiff_exponential` at t=1 (and t=0 identity).
**Behavior:** required args per design (`--n --kx --ky --cx --cy --method --nu --t-final --dt --out`); validate kx, ky within the two-thirds band (exit 2 with message otherwise); dispatch method → `advance`; write `run.json`, `final.json` (physical field, max spectral error); stdout header + `method dt steps max_error`.
**Commits:** `test: add failing field binary exact-solution check (red)` / `feat: add field verification binary (green)`; then `evidence: field single-mode verification run`.

### Task 3: fluid library (stream function, velocity, nonlinear term, rate)

**Files:** Create `week4/tests/fluid.rs`, `week4/src/fluid.rs`; modify `week4/src/lib.rs`.
**Tests (red):** stream-function sign (ψ̂ = −ω̂/κ², mean zero); Taylor–Green velocity = (sin x cos y, −cos x sin y) via IDFT2; single Hermitian mode nonlinear term ≡ 0 (machine); two-mode interaction spectrum = −i at (4,3) and (−2,1), +i at conjugates, zero elsewhere (values from fluid.design.toml); Taylor–Green rate = pure viscous decay (rate = −2ν·ω̂ per mode) and RK4 evolution matches exp(−2νt) (1e−10).
**Green:** `stream_function`, `velocity`, `nonlinear_term` (IDFT2 ×4 → physical products → DFT2 → dealias → negate), `rate(n, nu, omega)`.
**Commits:** `test: add failing vorticity-equation contract tests (red)` / `feat: implement pseudospectral vorticity dynamics (green)`.

### Task 4: diagnostics + measured inviscid conservation bound

**Files:** Modify `week4/src/fluid.rs`, `week4/tests/fluid.rs`.
**Tests (red):** Taylor–Green energy exactly 1/4, enstrophy exactly 1/2 (1e−12).
**Then measure:** run ν=0, n=32, dt=0.01, t=0.5 evolution; record relative energy drift in the transcript; write regression test `inviscid_energy_drift_stays_at_measured_bound` with the measured bound (1 significant digit of margin) and its provenance comment.
**Commits:** `test: add failing Parseval diagnostics tests (red)` / `feat: add energy and enstrophy diagnostics (green)` / `test: pin measured inviscid energy-drift bound`.

### Task 5: fluid binary

**Files:** Create `week4/src/bin/fluid.rs`.
**Behavior:** args per design (`--n --method --nu --dt --t-final --every --out`); Taylor–Green start; `advance` full history; series frames at every-th step; exact reference 2 sin x sin y·e^{−2νt}; writes stdout line, `run.json`, `series.jsonl`, `final.json`.
**Commits:** `feat: add fluid vorticity-solver binary (green)`; then `evidence: fluid Taylor-Green verification runs` (usage-line runs from both design tomls + the inviscid measurement run).

## Verification

`cargo test` green/warning-free from `week4/`; both binaries run their design usage lines; committed artifacts under `week4/runs/`.
