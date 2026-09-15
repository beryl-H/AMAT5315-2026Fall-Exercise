# Week 2: Heating Ramp (`md run --ramp-to`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `md run --ramp-to <temperature>` heating thermostat: production velocities are rescaled every production step toward a target that rises linearly from `--temperature` at step 0 to `--ramp-to` at the final production step, with `ramp_to` recorded in run.json — while preserving the unheated equilibrium contract exactly.

**Architecture:** A pure `ramp_target` schedule function and the per-step production rescale live in `simulate.rs` (reusing the existing `rescale_to`); `SimConfig` and `RunConfig` gain an optional `ramp_to: Option<f64>` (run.json omits the key when `None`); `RunArgs` gains the `--ramp-to` flag. No new physics, no integrator change, no checker change.

**Tech Stack:** Rust 2021, existing crate only.

**Spec:** `week2/SPEC.md` (corrected commit `98aa93d`)

## Global Constraints

- Run all cargo commands from `week2/` with `--manifest-path md/Cargo.toml`.
- Preserve the unheated equilibrium behavior exactly: equilibration Schedule B unchanged; production thermostat OFF when `--ramp-to` absent; the default contract run, `md check` bounds, and the energy-drift test unchanged.
- The heating-production cadence is **[Need confirmation]** in the SPEC; the approved **[Suggestion]** is a **per-step** rescale (`s = 1..=S`). Do not present every-50 as a course fact for production heating.
- Heating injects energy: do NOT weaken/redefine the unheated drift criterion; no energy-conservation acceptance condition for heated runs.
- `ramp_to` recording: **[Course Requirement]** heating runs record it; **[Suggestion]** represent as `Option<f64>` with `skip_serializing_if = "Option::is_none"` so unheated run.json stays schema-compatible.
- Preserve `ForceMethod::Cells` default, `--force naive`, all Part 2–4/5 tests and tolerances.
- Ramp reuses the existing uniform `rescale_to` (mass 1, COM preserved); no new integrator.
- Ordering inside a heated production step: integrate step `s` → rescale to `T_target(s)` → save frame if `s % sample_every == 0`.
- The official heating run is separate from the unheated contract artifacts; `make reproduce`/`artifacts/` never uses `--ramp-to`.

## Ramp target (course-derived) and schedule (Suggestion)

```
T_target(s) = T0 + (T1 - T0) * s / S        s in [0, S]
T_target(0) = T0                             T_target(S) = T1
```

with `T0 = --temperature`, `T1 = --ramp-to`, `S = --steps`. Production runs
steps `1..=S`; after each step `s` (no rescale at step 0) rescale to
`T_target(s)`. This applies `T1` exactly at `s = S` for every `S` (no
divisibility condition).

---

### Task 1: First RED — heating acceptance tests (before the heating API exists)

**Files:**
- Modify: `week2/md/src/simulate.rs` (add `ramp_target` unit test to the existing `#[cfg(test)] mod tests`).
- Create: `week2/md/tests/heating.rs`.
- Modify: `week2/md/tests/cli.rs` (add three CLI tests).

**Interfaces:**
- Consumes: `md::simulate::{SimConfig, run_simulation}`, `md::Frame`, `md::ForceMethod`, `md::thermodynamic_temperature`, `md::io::{RunConfig, write_artifacts}` (all exist).
- Produces: the failing acceptance tests that later tasks turn green. They reference `SimConfig.ramp_to`, the per-step schedule, and `--ramp-to` — none of which exist yet, which is the RED.

- [ ] **Step 1: Write the failing tests**

In `simulate.rs` tests module append:

```rust
    #[test]
    fn ramp_target_matches_linear_schedule() {
        // T_target(0) = T0, T_target(S) = T1, interior linear [Course Req].
        assert!((ramp_target(0.2, 1.2, 0, 200) - 0.2).abs() < 1e-15);
        assert!((ramp_target(0.2, 1.2, 200, 200) - 1.2).abs() < 1e-15);
        assert!((ramp_target(0.2, 1.2, 100, 200) - 0.7).abs() < 1e-15);
        // works for any S (no divisibility condition)
        assert!((ramp_target(0.2, 1.2, 125, 125) - 1.2).abs() < 1e-15);
    }
```

Create `week2/md/tests/heating.rs`:

```rust
//! Heating ramp acceptance tests: per-step [Suggestion] production
//! thermostat, run.json recording, and no-ramp preservation.

use md::io::{RunConfig, write_artifacts};
use md::simulate::{SimConfig, run_simulation};
use md::thermodynamic_temperature;

const TOL: f64 = 1e-9;

fn heated_config(steps: usize, sample_every: usize) -> SimConfig {
    SimConfig {
        n: 16,
        rho: 0.8,
        temperature: 0.2,
        dt: 0.01,
        eq_steps: 0,
        steps,
        sample_every,
        seed: 2026,
        force_method: md::ForceMethod::Cells,
        ramp_to: Some(1.2),
    }
}

fn ramp_expected(s: usize, s_total: usize) -> f64 {
    0.2 + (1.2 - 0.2) * (s as f64) / (s_total as f64)
}

fn frame_temp(frame: &md::Frame) -> f64 {
    thermodynamic_temperature(&frame.vel)
}

#[test]
fn heated_production_pins_every_saved_frame_and_reaches_endpoint() {
    // S = 100 (divisible by 50) and S = 125 (not divisible by 50): the
    // per-step thermostat must pin every saved frame to T_target(s) and
    // apply exactly T1 at the final step s = S.
    for (steps, sample_every) in [(100usize, 25usize), (125usize, 25usize)] {
        let config = heated_config(steps, sample_every);
        let frames = run_simulation(&config);
        assert_eq!(frames.len(), steps / sample_every);
        for f in &frames {
            let expected = ramp_expected(f.step, steps);
            let t = frame_temp(f);
            assert!((t - expected).abs() < TOL, "step {}: T {t} != {expected}", f.step);
        }
        let last = frames.last().unwrap();
        assert_eq!(last.step, steps);
        assert!((frame_temp(last) - 1.2).abs() < TOL, "endpoint T1 not reached");
    }
}

#[test]
fn no_ramp_leaves_production_thermostat_off() {
    // Without ramp_to the production thermostat stays completely OFF: a
    // frame at a production multiple of 50 is NOT pinned to any target.
    let config = SimConfig {
        n: 16,
        rho: 0.8,
        temperature: 0.5,
        dt: 0.01,
        eq_steps: 0,
        steps: 200,
        sample_every: 25,
        seed: 2026,
        force_method: md::ForceMethod::Cells,
        ramp_to: None,
    };
    let frames = run_simulation(&config);
    let pinned = frames
        .iter()
        .filter(|f| f.step % 50 == 0)
        .all(|f| (frame_temp(f) - 0.5).abs() < 1e-6);
    assert!(!pinned, "production thermostat must stay OFF without --ramp-to");
}

#[test]
fn heating_run_json_records_ramp_to_and_unheated_omits_it() {
    let dir = std::env::temp_dir().join(format!("md-heat-json-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let heated = heated_config(100, 25);
    let frames = run_simulation(&heated);
    write_artifacts(&dir, &RunConfig::from(&heated), &frames).unwrap();
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(v["ramp_to"], 1.2);
    let _ = std::fs::remove_dir_all(&dir);

    let dir2 = std::env::temp_dir().join(format!("md-heat-json2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir2);
    let mut plain = heated_config(100, 25);
    plain.ramp_to = None;
    let frames2 = run_simulation(&plain);
    write_artifacts(&dir2, &RunConfig::from(&plain), &frames2).unwrap();
    let v2: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir2.join("run.json")).unwrap(),
    )
    .unwrap();
    assert!(v2.get("ramp_to").is_none(), "unheated run.json must omit ramp_to");
    let _ = std::fs::remove_dir_all(&dir2);
}
```

Append to `week2/md/tests/cli.rs`:

```rust
#[test]
fn run_accepts_ramp_to_and_records_it() {
    let out = temp_dir("ramp-ok");
    let output = run_md(&["run", "--n", "16", "--eq-steps", "0", "--steps", "100",
                          "--sample-every", "50", "--temperature", "0.2", "--ramp-to", "1.2",
                          "--out", out.to_str().unwrap()]);
    assert!(output.status.success(), "ramp run failed: {}", String::from_utf8_lossy(&output.stderr));
    let run: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(run["ramp_to"], 1.2);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn run_rejects_nonpositive_ramp_to() {
    for bad in ["0", "-0.5"] {
        let out = temp_dir("ramp-bad");
        let output = run_md(&["run", "--n", "16", "--ramp-to", bad, "--out", out.to_str().unwrap()]);
        assert!(!output.status.success(), "--ramp-to {bad} must be rejected");
        let _ = std::fs::remove_dir_all(&out);
    }
}

#[test]
fn run_without_ramp_to_omits_key() {
    let out = temp_dir("ramp-none");
    let output = run_md(&["run", "--n", "16", "--eq-steps", "0", "--steps", "100",
                          "--sample-every", "50", "--out", out.to_str().unwrap()]);
    assert!(output.status.success());
    let run: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).unwrap(),
    )
    .unwrap();
    assert!(run.get("ramp_to").is_none());
    let _ = std::fs::remove_dir_all(&out);
}
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --test heating`
Expected: compile FAIL — `error[E0609]: no field 'ramp_to' on type 'SimConfig'` (plus `--ramp-to` is not yet accepted by the binary in the CLI tests).

Run: `cargo test --manifest-path md/Cargo.toml --lib ramp_target`
Expected: compile FAIL — `error[E0425]: cannot find function ramp_target`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add week2/md/src/simulate.rs week2/md/tests/heating.rs week2/md/tests/cli.rs
git commit -m "test: add failing heating ramp acceptance tests"
```

---

### Task 2: `ramp_target` pure function

**Files:**
- Modify: `week2/md/src/simulate.rs`.

**Interfaces:**
- Produces (crate-visible): `pub(crate) fn ramp_target(t0: f64, t1: f64, s: usize, s_total: usize) -> f64`

- [ ] **Step 1: RED already exists** (Task 1 unit test). Confirm:

Run: `cargo test --manifest-path md/Cargo.toml --lib ramp_target`
Expected: compile FAIL — `cannot find function ramp_target`.

- [ ] **Step 2: Implement** (place above `run_simulation`):

```rust
/// Linear ramp target at production step `s` of a total of `s_total`:
/// T_target(s) = t0 + (t1 - t0) * s / s_total [Course Requirement].
/// Callers guarantee s_total >= 1 (production steps are validated > 0).
pub(crate) fn ramp_target(t0: f64, t1: f64, s: usize, s_total: usize) -> f64 {
    t0 + (t1 - t0) * (s as f64) / (s_total as f64)
}
```

- [ ] **Step 3: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib ramp_target`
Expected: PASS.

- [ ] **Step 4: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib`
Expected: PASS (the `--test heating` target is still the staged compile-RED — `SimConfig.ramp_to` missing — until Task 3).

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/simulate.rs
git commit -m "feat: add linear ramp target function"
```

---

### Task 3: `SimConfig` optional `ramp_to`

**Files:**
- Modify: `week2/md/src/simulate.rs` (struct, Default, `small_config`, `defaults_are_the_course_contract_values`).
- Modify: `week2/md/src/cli.rs` (run_command literal), `week2/md/src/video.rs` (test literal), `week2/md/tests/checker.rs` (four `SimConfig { .. }` literals).

**Interfaces:**
- Produces: `SimConfig { ..., pub ramp_to: Option<f64> }` with `Default` `None`.

- [ ] **Step 1: Write the failing test change** (update the existing `defaults_are_the_course_contract_values` in `simulate.rs`):

```rust
        assert!(c.ramp_to.is_none(), "unheated default has no ramp target");
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib defaults_are_the_course_contract_values`
Expected: compile FAIL — `error[E0609]: no field 'ramp_to' on type 'SimConfig'`.

- [ ] **Step 3: Implement**

In `simulate.rs` add the field and staged default:

```rust
pub struct SimConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub force_method: ForceMethod,
    pub ramp_to: Option<f64>,
}
```

In `Default`: `ramp_to: None,`.

Update every `SimConfig { .. }` literal with `ramp_to: None,`:
- `week2/md/src/simulate.rs` `small_config()` (test helper).
- `week2/md/src/cli.rs` `run_command` (line ~122) — `ramp_to: None,` (replaced by `args.ramp_to` in Task 6).
- `week2/md/src/video.rs` test literal.
- `week2/md/tests/checker.rs` literals (4 sites).

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib defaults_are_the_course_contract_values`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib --test lj --test dimer --test pair --test fluid --test checker --test cli`
Expected: all PASS. (`--test heating` now compiles and its schedule tests run but FAIL at runtime — the per-step rescale does not exist yet. This is the expected staged RED until Task 4.)

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/simulate.rs week2/md/src/cli.rs week2/md/src/video.rs week2/md/tests/checker.rs
git commit -m "feat: add optional ramp_to to SimConfig"
```

---

### Task 4: Production heating schedule in `run_simulation`

**Files:**
- Modify: `week2/md/src/simulate.rs` (production loop).

**Interfaces:**
- Consumes: `ramp_target` (Task 2), `SimConfig.ramp_to` (Task 3), existing `rescale_to` (already imported).
- Produces: per-step heating in production. Ordering: integrate step `s` → rescale to `T_target(s)` → save frame if `s % sample_every == 0`.

- [ ] **Step 1: Write the failing test** (append to `simulate.rs` tests):

```rust
    #[test]
    fn heated_production_rescales_every_step_toward_linear_target() {
        // Per-step [Suggestion]: the thermodynamic temperature after each
        // production step equals the linear ramp target for that step.
        let mut c = small_config();
        c.temperature = 0.2;
        c.ramp_to = Some(1.2);
        let frames = run_simulation(&c); // steps = 200, sample_every = 25
        for f in &frames {
            let expected = ramp_target(0.2, 1.2, f.step, c.steps);
            let t = crate::thermostat::thermodynamic_temperature(&f.vel);
            assert!((t - expected).abs() < 1e-9, "step {}: T {t} != {expected}", f.step);
        }
        let last = frames.last().unwrap();
        assert_eq!(last.step, c.steps);
        assert!((crate::thermostat::thermodynamic_temperature(&last.vel) - 1.2).abs() < 1e-9);
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib heated_production_rescales_every_step`
Expected: FAIL — frames are not pinned (production thermostat still OFF), e.g. `T <expected> != <target>`.

- [ ] **Step 3: Implement** — replace the production loop body:

```rust
    // Production: thermostat OFF unless --ramp-to is set, time starts at zero.
    let mut frames = Vec::with_capacity(config.steps / config.sample_every);
    for step in 1..=config.steps {
        verlet.step(&mut state, config.dt, &accelerations);
        wrap_state(&mut state, &bx);
        // [Suggestion] per-step heating: rescale after every production
        // integration step toward the linear ramp target (no step-0 event).
        if let Some(ramp_to) = config.ramp_to {
            let target = ramp_target(config.temperature, ramp_to, step, config.steps);
            rescale_to(&mut state.velocities, target);
        }
        if step % config.sample_every == 0 {
            frames.push(Frame {
                step,
                t: step as f64 * config.dt,
                pos: state.positions.clone(),
                vel: state.velocities.clone(),
                e_pot: config.force_method.potential_energy(&state, &bx),
                e_kin: crate::kinetic_energy(&state),
            });
        }
    }
```

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib heated_production_rescales_every_step`
Expected: PASS. Also run `cargo test --manifest-path md/Cargo.toml --test heating` — schedule and no-ramp tests now PASS; the run.json test still FAILS (io does not write `ramp_to` yet).

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib`
Expected: PASS (unheated behavior unchanged — with `ramp_to: None` the loop is identical to before).

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/simulate.rs
git commit -m "feat: apply per-step production heating thermostat"
```

---

### Task 5: run.json optional `ramp_to` (io.rs)

**Files:**
- Modify: `week2/md/src/io.rs` (RunConfig, `From<SimConfig>`, unit tests).

**Interfaces:**
- Produces: `RunConfig { ..., pub ramp_to: Option<f64> }` with `#[serde(skip_serializing_if = "Option::is_none")]`; `From<SimConfig>` maps `c.ramp_to`.

- [ ] **Step 1: Write the failing unit tests** (append to `io.rs` tests):

```rust
    #[test]
    fn ramp_to_round_trips_and_omits_when_none() {
        let dir = temp("ramp");
        // heated: Some(1.2) serialized and read back
        let mut c = crate::simulate::SimConfig::default();
        c.ramp_to = Some(1.2);
        let run = RunConfig::from(c);
        write_artifacts(&dir, &run, &sample_frames()).unwrap();
        let text = std::fs::read_to_string(dir.join("run.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["ramp_to"], 1.2);
        let (run2, _) = read_artifacts(&dir).unwrap();
        assert_eq!(run2.ramp_to, Some(1.2));
        // unheated: key absent, reads back as None
        let dir2 = temp("ramp2");
        let run3 = RunConfig::from(crate::simulate::SimConfig::default());
        assert_eq!(run3.ramp_to, None);
        write_artifacts(&dir2, &run3, &sample_frames()).unwrap();
        let text2 = std::fs::read_to_string(dir2.join("run.json")).unwrap();
        assert!(!text2.contains("ramp_to"));
        let (run4, _) = read_artifacts(&dir2).unwrap();
        assert_eq!(run4.ramp_to, None);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib ramp_to_round_trips`
Expected: compile FAIL — `error[E0609]: no field 'ramp_to' on type 'RunConfig'` / `SimConfig`.

- [ ] **Step 3: Implement**

In `io.rs`:

```rust
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_dim: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    /// Heating target; present in run.json only for heating runs
    /// [Course Requirement: heating runs record ramp_to; Suggestion:
    /// omitted when None so the unheated schema is unchanged].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ramp_to: Option<f64>,
}
```

and in `From<SimConfig>`: `ramp_to: c.ramp_to,`.

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib ramp_to_round_trips`
Expected: PASS. Also `cargo test --manifest-path md/Cargo.toml --lib io` — the existing `run_json_keys_are_exactly_the_course_contract` and round-trip tests stay GREEN (unheated run.json unchanged).

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib --test cli`
Expected: PASS. `cargo test --manifest-path md/Cargo.toml --test heating` — now only the CLI tests in `tests/cli.rs` (Task 1) and the io-run tests are left; `heating_run_json_records_ramp_to_and_unheated_omits_it` now PASSES.

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/io.rs
git commit -m "feat: record optional ramp_to in run.json"
```

---

### Task 6: CLI `--ramp-to` parsing/validation → full heating GREEN

**Files:**
- Modify: `week2/md/src/cli.rs` (RunArgs + run_command).

**Interfaces:**
- Consumes: `SimConfig.ramp_to` (Task 3).
- Produces: `RunArgs { ..., pub ramp_to: Option<f64> }`; validation rejects `ramp_to <= 0` per SPEC [Suggestion].

- [ ] **Step 1: RED already exists** (Task 1 CLI tests). Confirm:

Run: `cargo test --manifest-path md/Cargo.toml --test cli run_accepts_ramp_to`
Expected: FAIL — `error: unexpected argument '--ramp-to' found` (subprocess non-zero).

- [ ] **Step 2: Implement**

In `RunArgs`:

```rust
    /// Heating target temperature; with it, production rescales every step
    /// toward a linear ramp ending here. [Suggestion] optional.
    #[arg(long)]
    pub ramp_to: Option<f64>,
```

In `run_command`, after the existing `temperature <= 0` check:

```rust
    if let Some(ramp_to) = args.ramp_to {
        if ramp_to <= 0.0 {
            return Err(format!("--ramp-to must be positive, got {ramp_to}"));
        }
    }
```

and in the `SimConfig` literal: `ramp_to: args.ramp_to,` (replacing the Task 3 `None`).

- [ ] **Step 3: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test cli`
Expected: all cli tests PASS (now 11, incl. the three new ramp tests).

Run: `cargo test --manifest-path md/Cargo.toml --test heating`
Expected: all 3 tests PASS — the acceptance suite is fully GREEN.

- [ ] **Step 4: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib --test lj --test dimer --test pair --test fluid --test checker --test contract`
Expected: all PASS (unheated contract unchanged; `--ramp-to` absent → production thermostat OFF exactly as before).

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/cli.rs
git commit -m "feat: add --ramp-to CLI flag and validation"
```

---

### Task 7: Negative control

**Files:**
- Modify: `week2/md/src/simulate.rs` (TEMPORARY, restored before commit).

- [ ] **Step 1: Introduce the defect** — replace the ramp target with a constant:

```rust
        if let Some(_ramp_to) = config.ramp_to {
            // NEGATIVE CONTROL ONLY: constant T0 instead of the linear ramp.
            rescale_to(&mut state.velocities, config.temperature);
        }
```

- [ ] **Step 2: Run the heating schedule tests**

Run: `cargo test --manifest-path md/Cargo.toml --test heating`
Expected: FAIL — `heated_production_pins_every_saved_frame_and_reaches_endpoint` fails (frames pinned to 0.2, final temperature != 1.2), and the simulate.rs unit test `heated_production_rescales_every_step_toward_linear_target` fails. This proves the schedule tests exercise the linear target, not a constant.

- [ ] **Step 3: Restore the correct implementation** (revert the `_ramp_to`/constant change back to `ramp_target(config.temperature, ramp_to, step, config.steps)`).

- [ ] **Step 4: Re-run GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test heating && cargo test --manifest-path md/Cargo.toml --lib heated_production`
Expected: PASS.

- [ ] **Step 5: Confirm no defect is committed**

Run: `git diff --stat` — only the intended file changes; the defect line is absent.

---

### Task 8: Release regression verification

**Files:** none (verification only).

- [ ] **Step 1: Full release suite**

Run: `cargo test --manifest-path md/Cargo.toml --release`
Expected: all PASS, including the naive/cells equality tests (`tests/cells.rs`), a heating schedule test (`tests/heating.rs` + simulate.rs unit test), Part 2–4 regression, contract, checker, cli.

- [ ] **Step 2: Unheated contract check unchanged**

```bash
make reproduce
cargo run --manifest-path md/Cargo.toml --release -- check artifacts
```
Expected: `make reproduce` writes 200 frames to `week2/artifacts/`; `md check artifacts` exits 0 with all three bounds PASS (drift < 2e-3, |T_speed − 0.5| < 0.05, chi2_22 < 2) — unchanged.

- [ ] **Step 3: Commit only if a fix was needed** (expected: none).

---

### Task 9: Reinstall the current `md` binary on PATH

**Files:** none (environment step).

- [ ] **Step 1: Install**

From `week2/`:

```bash
cargo install --path md
```

Expected: `md` on PATH is now the current build (with `--ramp-to`).

---

### Task 10: Official heating trajectory run + REPRODUCE

**Files:** none (the heated output is separate from the contract artifacts).

- [ ] **Step 1: Run the official course heating command**

From the repo root (after the `cd ..` below; `--out docs` is the course-specified output name, written relative to the invocation directory and kept separate from `week2/artifacts/`):

```bash
cd ..
md run --n 400 --temperature 0.2 --ramp-to 1.2 \
  --steps 20000 --sample-every 100 --out docs
```

- [ ] **Step 2: Verify the heated run**

- `docs/run.json` exists and contains `"ramp_to": 1.2` (and the Part 4 keys).
- `docs/traj.jsonl` exists with `20000 / 100 = 200` frames.
- Do NOT claim energy conservation for this heated run (energy is injected by design).

- [ ] **Step 3: Confirm the unheated REPRODUCE path is untouched**

```bash
cd week2
make reproduce
cargo run --manifest-path md/Cargo.toml --release -- check artifacts
```
Expected: exit 0, all bounds PASS — the unheated contract run is unchanged.

---

## Self-Review

**1. Spec coverage** — each SPEC requirement maps to a task: optional flag (T1, T6); no-ramp preservation (T4 loop unchanged when None, T1 no-ramp test, T8 contract check); linear target formula (T2 + tests); per-step [Suggestion] schedule with step ordering integrate→rescale→save (T4); run.json recording [Course Req] + Option/skip-if-none [Suggestion] (T5); energy injection → no drift change, no energy-conservation acceptance (Global Constraints, T10 "do not claim"); cells default / --force naive preserved (no force changes anywhere); [Need confirmation] cadence preserved in SPEC labels and implemented only as [Suggestion] (T4 comment).

**2. Placeholder scan** — every step has concrete code/commands; no TBD/TODO.

**3. Type consistency** — `ramp_target(t0,t1,s,s_total)` defined once (T2) and used in T4; `SimConfig.ramp_to: Option<f64>` (T3) consumed by T4/T5/T6; `RunConfig.ramp_to` with `skip_serializing_if` (T5); `RunArgs.ramp_to: Option<f64>` (T6). Test tolerance `1e-9` used consistently in heating schedule assertions; unit `1e-15` for the pure function.
