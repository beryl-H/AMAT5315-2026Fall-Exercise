# Week 4 Field & Fluid Course-Contract Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the `field` and `fluid` binaries to the course-provided contracts in `week4/field.design.toml` and `week4/fluid.design.toml` (verbatim; not to be modified), on the existing `advdiff` integrators and `Integrator` trait.

**Architecture:** `field` becomes a subcommand CLI (`taylor-green`, `random`) emitting one JSON object (case, n, seed, k_band, u, v) to stdout — the initial-condition generator. `fluid` reads that JSON from stdin, converts velocity to spectral vorticity (curl, per-axis Nyquist rule), dealiases, integrates omega_t + u.grad(omega) = nu laplacian(omega) with euler/rk2/rk4 through the `Integrator` trait, prints t/E/Z snapshot lines (stopping at the first non-finite energy with exit 1), and writes run.json + fields.jsonl. Lib: keep DFT2/derivatives/dealias/stream-function/velocity/nonlinear/rate/energy/enstrophy; retarget the Taylor–Green fixture to the course vorticity omega = -2 cos(x) cos(y); add `taylor_green_velocity`, `random_vorticity` (splitmix64 phases, order independent of n, amplitude from E(0)=1/2), `vorticity_from_velocity`; drop `advdiff_rate`, `advdiff_exponential`, `hermitian_mode` (draft-only API).

**Tech Stack:** advdiff crate; new dependency serde_json (stdin parsing, structured output).

**Spec:** `week4/field.design.toml`, `week4/fluid.design.toml` (course-provided; do not modify).

## Global Constraints

- History: red commit (test watched failing for the right reason) → green commit.
- The design tomls are the contract; any conflict is resolved in code, never by editing the tomls.
- Two-thirds rule on the vorticity itself and every product; per-axis Nyquist no-travel on first derivatives.
- `cargo test` green and warning-free from `week4/` after every green commit.

---

### Task 1: library red → green

**Tests (red):** retarget `taylor_green_spectral` fixture (omega = -2 cos x cos y, modes (+-1,+-1) all -1/2); `taylor_green_velocity` formula (u = cos x sin y e^{-2 nu t}, v = -sin x cos y e^{-2 nu t}); `random_vorticity` (E = 1/2 exactly; deterministic per seed; phases independent of n across n = 64/96; Hermitian; equal amplitudes inside the annulus, zero outside); `vorticity_from_velocity` (course TG velocity -> the course TG vorticity spectrum). Remove tests of the dropped draft API.
**Green:** lib changes per Architecture. Commit pair: `test: add failing course-contract library tests (red)` / `feat: align field and fluid libraries with the course contracts (green)`.

### Task 2: binaries red → green

**field (red tests in bin):** `taylor_green_output` JSON parses with case/n/seed/k_band/u/v, sample values exact, nu required when t > 0 (Err); `random_output` JSON parses and delivers E = 1/2 through `vorticity_from_velocity` + `energy`.
**field (green):** clap subcommands; stdout = `serde_json` object + newline.
**fluid (red tests in bin):** `parse_field_input` extracts case/n/seed/k_band/u/v; `snapshot_schedule(n_steps, s)` = [0, s, 2s, ... <= n_steps].
**fluid (green):** stdin parse; omega = dealias(curl(u, v)); fixed steps `round(t_end/dt)` (never shortened); snapshots at step 0 and every `round(every/dt)` steps; stdout header `t\tE\tZ` + lines; first non-finite energy prints its line and exits 1; writes run.json (case/n/seed/k_band piped + method, nu, dt, t_end, snapshot_every) and fields.jsonl (t, step, u, v, omega at 6 decimals). Commit pairs: `test: add failing course-contract binary tests (red)` / `feat: rebuild field and fluid binaries to the course contracts (green)`.

### Task 3: evidence

Remove the obsolete draft-run artifacts; run the contract usage lines (`field taylor-green --n 64 --nu 0.1 --t 1 | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green` and a `random` pipe); commit under `week4/artifacts/`. Commit: `evidence: course-contract field and fluid runs`.

## Verification

`cargo test` green and warning-free from `week4/`; piped field|fluid runs complete with finite energy throughout.
