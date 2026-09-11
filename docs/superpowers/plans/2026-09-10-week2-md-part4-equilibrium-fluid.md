# Week 2 Part 4 Equilibrium-Fluid CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `md` crate in `week2/md/` with `md run`, `md check`, and
`md video` subcommands that simulate an equilibrium 2D Lennard-Jones fluid
with periodic boundaries and a shifted cutoff, independently verify it, and
render it to MP4 — while preserving all Part 2/3 behavior.

**Architecture:** All physics lives in new additive library modules
(`system`, `pair`, `fluid`, `thermostat`, `simulate`, `io`, `metrics`,
`checker`, `render`, `video`); the single clap-derived binary in `cli.rs`/
`main.rs` only wires subcommands to the library. The Part 3 dimer path
(`lj_accelerations`, open boundaries, plain LJ) is untouched. `md check`
recomputes all physics from raw saved positions/velocities via
`metrics`/`checker`; stored energies are cross-checks only. `md video`
rasterizes RGBA frames in Rust and pipes them to an external `ffmpeg`.

**Tech Stack:** Rust 2021, `serde` + `serde_json` (JSON artifacts),
`rand` + `rand_distr` (seeded Gaussians), `clap` (CLI), system `ffmpeg`
(external, video only).

**Spec:** `../specs/2026-09-10-week2-md-equilibrium-fluid-design.md`

## Global Constraints

- Run all cargo commands from `week2/` with
  `--manifest-path md/Cargo.toml` (equivalently from the repo root with
  `--manifest-path week2/md/Cargo.toml`). Final verification command:
  `cargo test --manifest-path md/Cargo.toml --release` from `week2/`.
- Do not modify `src/lib.rs`'s existing `lj_energy`, `lj_force`,
  `greeting`; do not modify `src/state.rs`, `src/integrator.rs`,
  `src/experiment.rs`, `tests/lj.rs`, `tests/dimer.rs`, or the examples
  except that `lib.rs` gains new `mod`/`pub use` lines.
- The dimer keeps plain LJ + open boundaries; the fluid path is separate
  code, never a replacement.
- Reduced units: sigma = epsilon = m = k_B = 1. Cutoff rc = 2.5
  (strict `r < rc` uses the potential/force; `r >= rc` gives zero).
- Lattice: `a = sqrt(2/(sqrt(3)*rho))`, `h = sqrt(3)/2*a`,
  `x = (i + 0.5*(j%2))*a`, `y = j*h`, `Lx = n*a`, `Ly = n*h`, n = sqrt(N).
- Thermostat Schedule B: rescale initially and after each equilibration
  step `s` with `s % 50 == 0` (41 events for eq_steps = 2000); COM
  subtraction only once after the initial draw; T_target is the run's
  `--temperature`, never hard-coded 0.5.
- Production: thermostat OFF, t starts at 0, save steps
  `sample_every..=steps` (never step 0), 200 frames for the defaults.
- Notation: F = saved frames, S = F*N pooled speed samples,
  k = max(1, floor(F/10)), E_b = S/24,
  `chi2_22 = (1/22) * sum_b (O_b - E_b)^2 / E_b` with Rayleigh edges
  `b_k = sqrt(-2*T_speed*ln(1 - k/24))`, b_24 = infinity.
- Temperature gate: `abs(T_speed - run_config.temperature) < 0.05`
  ([Suggestion] for general runs; Course Requirement at default T = 0.5).
- Stored-energy cross-check [Suggestion]:
  `abs(stored - recomputed) <= 1e-10 * max(1, abs(recomputed))`.
- Box cross-check [Suggestion]: `1e-10 * max(1, L)` per axis.
- No cell lists / optimization (that is Part 5). No Python/pip anywhere.
- ffmpeg is an external prerequisite for `md video` only; video tests skip
  when the `ffmpeg` executable is absent; `make reproduce` never uses it.
- Mark Course Requirement vs [Suggestion] in test comments wherever the
  design does.
- Workflow: DESIGN → PLAN → TEST → RED → IMPLEMENT → GREEN → VERIFY →
  REPRODUCE. Tests are committed before the implementation that turns
  them green, per task.

## File Structure

```
week2/Makefile                         NEW  reproduce target (md run only)
week2/md/Cargo.toml                    MOD  add serde, serde_json, rand,
                                            rand_distr, clap
week2/md/Cargo.lock                    MOD  committed, pins all versions
week2/md/src/lib.rs                    MOD  new mod + pub use lines only
week2/md/src/cli.rs                    NEW  clap definitions + dispatch
week2/md/src/main.rs                   MOD  calls md::cli::main()
week2/md/src/system.rs                 NEW  lattice, Box2, MIC, wrap
week2/md/src/pair.rs                   NEW  shifted cutoff energy/force
week2/md/src/fluid.rs                  NEW  periodic accelerations/energies
week2/md/src/thermostat.rs             NEW  Gaussian init, COM, rescale
week2/md/src/simulate.rs               NEW  equilibration + production driver
week2/md/src/io.rs                     NEW  run.json / traj.jsonl read+write
week2/md/src/metrics.rs                NEW  drift, T_speed, chi2_22, g(r)
week2/md/src/checker.rs                NEW  structural validation + report
week2/md/src/render.rs                 NEW  RGBA rasterizer (panels)
week2/md/src/video.rs                  NEW  ffmpeg pipe
week2/md/tests/pair.rs                 NEW  cutoff acceptance tests
week2/md/tests/fluid.rs                NEW  internal-force acceptance tests
week2/md/tests/cli.rs                  NEW  binary file/frame contract tests
week2/md/tests/contract.rs             NEW  full default-contract test
week2/md/tests/checker.rs              NEW  checker unit/negative tests
week2/md/.gitignore                    MOD  ignore /artifacts
```

Unit tests for pure functions live in `#[cfg(test)] mod tests` inside each
new module (repo pattern); acceptance tests live in `tests/*.rs` integration
files (also the repo pattern). All commands below run from `week2/`.

---

### Task 1: Dependencies + course-required acceptance tests (TEST / RED)

Strict test-first ordering: before the first RED commit only dependency
declarations and Cargo.lock setup are allowed. No CLI skeleton, no
implementation code. The CLI skeleton moves to Task 1b, after the RED
commit of the failing tests.

**Files:**
- Modify: `week2/md/Cargo.toml`, `week2/md/.gitignore`
- Create: `week2/md/tests/pair.rs`, `week2/md/tests/fluid.rs`,
  `week2/md/tests/cli.rs`, `week2/md/tests/contract.rs`

**Interfaces:**
- Consumes: existing crate `md`.
- Produces (public API referenced by the failing tests — later tasks
  implement these exact signatures):
  - `md::pair::{RC, shifted_energy, shifted_force}`
  - `md::system::{Box2, lattice_state, minimum_image, wrap, side}`
  - `md::fluid::{fluid_accelerations, fluid_potential_energy}`
  - `md::simulate::{SimConfig, Frame, run_simulation}`
  - `md::io::{RunConfig, read_artifacts, write_artifacts}`
  - `md::checker::check_artifacts`

- [ ] **Step 1: Add dependencies [Suggestion per design]**

Verify `StandardNormal` availability first: `rand_distr` provides the
Gaussian. With rustc 1.75, pin compatible versions:

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = "0.8"
rand_distr = "0.4"
clap = { version = "4.4", features = ["derive"] }
```

Run `cargo build --manifest-path md/Cargo.toml` to regenerate
`Cargo.lock`; commit the lockfile together with the failing tests in
Step 5 (nothing else). Do NOT create any source file yet.

- [ ] **Step 2: Write the four course-required acceptance test files**

`week2/md/tests/pair.rs` (Course Requirement: shifted cutoff potential):

```rust
//! Acceptance tests for the shifted-cutoff Lennard-Jones pair interaction.
//!
//! Course Requirement: U_cut(r) = U(r) - U(rc) for r < rc, 0 for r >= rc;
//! the potential is continuous and approaches zero at rc; the force is the
//! original Part 2 LJ force below rc and zero at/above rc (it has a small
//! jump at rc).

use md::pair::{RC, shifted_energy, shifted_force};

#[test]
fn shifted_potential_is_continuous_approaching_rc_from_below() {
    // Probing just inside rc, the value approaches 0.
    for eps in [1e-1, 1e-2, 1e-3, 1e-6, 1e-9] {
        let u = shifted_energy(RC - eps);
        assert!(u.abs() < 1e-6, "U_cut({RC - eps}) = {u}, expected ~0");
    }
    assert_eq!(shifted_energy(RC), 0.0);
    assert_eq!(shifted_energy(RC + 1.0), 0.0);
}

#[test]
fn shifted_potential_minus_one_at_minimum() {
    let r_min = 2f64.powf(1.0 / 6.0);
    assert!((shifted_energy(r_min) - (md::lj_energy(r_min) - md::lj_energy(RC))).abs() < 1e-12);
}

#[test]
fn force_is_plain_lj_below_rc_and_zero_at_or_above() {
    let r_min = 2f64.powf(1.0 / 6.0);
    for r in [0.9, 1.0, r_min, 2.0, RC - 1e-9] {
        assert!((shifted_force(r) - md::lj_force(r)).abs() < 1e-12);
    }
    assert_eq!(shifted_force(RC), 0.0);
    assert_eq!(shifted_force(RC + 0.5), 0.0);
}

#[test]
fn force_matches_finite_difference_below_rc_with_non_straddling_stencil() {
    // Course Requirement: force is -dU_cut/dr below rc. Stencils must stay
    // strictly inside r < rc because the force jumps at rc.
    let h = 1e-6;
    for r in [1.0, 1.5, 2.0, 2.4] {
        let fd = -(shifted_energy(r + h) - shifted_energy(r - h)) / (2.0 * h);
        let tol = 1e-5 * f64::max(1.0, shifted_force(r).abs());
        assert!(
            (shifted_force(r) - fd).abs() < tol,
            "force({r}) disagrees with finite difference"
        );
    }
}
```

`week2/md/tests/fluid.rs` (Course Requirement: total internal force
vanishes within numerical tolerance):

```rust
//! Acceptance tests for the periodic fluid force loop.
//!
//! Course Requirement: Newton's third law makes the total internal force
//! analytically zero; numerically we require |sum_i F_i| < tolerance.

use md::fluid::fluid_accelerations;
use md::system::{Box2, lattice_state, minimum_image, wrap};
use md::State;

const FORCE_TOL: f64 = 1e-10;

fn total_force(state: &State, bx: &Box2) -> [f64; 2] {
    let acc = fluid_accelerations(state, bx);
    let mut sum = [0.0, 0.0];
    for a in acc {
        sum[0] += a[0];
        sum[1] += a[1];
    }
    sum
}

#[test]
fn total_internal_force_vanishes_on_lattice() {
    let state = lattice_state(100, 0.8);
    let bx = Box2::new(100, 0.8);
    let f = total_force(&state, &bx);
    assert!(f[0].abs() < FORCE_TOL, "sum Fx = {}", f[0]);
    assert!(f[1].abs() < FORCE_TOL, "sum Fy = {}", f[1]);
}

#[test]
fn total_internal_force_vanishes_on_perturbed_configuration() {
    // Deterministic perturbation (no RNG needed): displace every other atom.
    let mut state = lattice_state(100, 0.8);
    let bx = Box2::new(100, 0.8);
    for (i, p) in state.positions.iter_mut().enumerate() {
        if i % 2 == 0 {
            p[0] = wrap(p[0] + 0.37 * (i as f64 % 3.0), bx.lx);
            p[1] = wrap(p[1] + 0.11, bx.ly);
        }
    }
    let f = total_force(&state, &bx);
    assert!(f[0].abs() < FORCE_TOL, "sum Fx = {}", f[0]);
    assert!(f[1].abs() < FORCE_TOL, "sum Fy = {}", f[1]);
}

#[test]
fn minimum_image_displacement_matches_definition() {
    assert!((minimum_image(0.3, 1.0) - 0.3).abs() < 1e-12);
    assert!((minimum_image(0.6, 1.0) - (-0.4)).abs() < 1e-12);
    assert!((minimum_image(-0.6, 1.0) - 0.4).abs() < 1e-12);
    assert!((minimum_image(5.3, 1.0) - 0.3).abs() < 1e-12);
}

#[test]
fn wrap_puts_positions_in_half_open_box_and_leaves_velocities() {
    assert_eq!(wrap(-0.2, 1.0), 0.8);
    assert_eq!(wrap(1.2, 1.0), 0.2);
    assert_eq!(wrap(0.0, 1.0), 0.0);
    assert!(wrap(1.0, 1.0) < 1.0 && wrap(1.0, 1.0) >= 0.0);
}
```

`week2/md/tests/cli.rs` (Course Requirement: binary writes required
files/frames):

```rust
//! Integration tests for the `md` binary's file and frame contract.
//!
//! Course Requirement: `md run` writes run.json (required fields,
//! integrator = "velocity-verlet") and traj.jsonl (one object per saved
//! production frame); saved steps follow --sample-every; step 0 is not
//! saved; the generic frame count is steps / sample_every.

use std::path::PathBuf;
use std::process::Command;

fn md_binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_md"))
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("md-cli-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn run_md(args: &[&str]) -> std::process::Output {
    md_binary().args(args).output().expect("failed to run md binary")
}

#[test]
fn run_writes_required_files_and_frames_for_a_small_run() {
    let out = temp_dir("small");
    let output = run_md(&[
        "run", "--n", "16", "--rho", "0.8", "--temperature", "0.5",
        "--dt", "0.01", "--eq-steps", "100", "--steps", "200",
        "--sample-every", "25", "--seed", "2026",
        "--out", out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "md run failed: {}", String::from_utf8_lossy(&output.stderr));

    // run.json required fields [Course Requirement].
    let run_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).expect("run.json missing"),
    )
    .expect("run.json malformed");
    for field in ["n", "rho", "box", "dt", "temperature", "eq_steps",
                  "steps", "sample_every", "seed", "integrator"] {
        assert!(run_json.get(field).is_some(), "run.json missing field {field}");
    }
    assert_eq!(run_json["integrator"], "velocity-verlet");
    assert_eq!(run_json["n"], 16);

    // traj.jsonl frames [Course Requirement].
    let traj = std::fs::read_to_string(out.join("traj.jsonl")).expect("traj.jsonl missing");
    let frames: Vec<serde_json::Value> = traj
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("traj line malformed"))
        .collect();
    // Generic rule: frame count = steps / sample_every (200/25 = 8).
    assert_eq!(frames.len(), 8);
    // Step 0 is never saved; steps follow sample_every.
    for (idx, frame) in frames.iter().enumerate() {
        let expected_step = (idx + 1) * 25;
        assert_eq!(frame["step"].as_u64().unwrap(), expected_step as u64);
        assert!((frame["t"].as_f64().unwrap() - expected_step as f64 * 0.01).abs() < 1e-12);
        assert_eq!(frame["pos"].as_array().unwrap().len(), 16);
        assert_eq!(frame["vel"].as_array().unwrap().len(), 16);
        assert!(frame["E_pot"].is_f64());
        assert!(frame["E_kin"].is_f64());
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn run_rejects_non_square_n_and_non_divisible_steps() {
    // [Suggestion] per design: clear nonzero-exit errors.
    let out = temp_dir("bad");
    let output = run_md(&["run", "--n", "50", "--out", out.to_str().unwrap()]);
    assert!(!output.status.success());
    let output = run_md(&["run", "--n", "16", "--steps", "100", "--sample-every", "30",
                          "--out", out.to_str().unwrap()]);
    assert!(!output.status.success());
    let _ = std::fs::remove_dir_all(&out);
}
```

Note: `tests/cli.rs` needs `serde_json` as a dev-dependency — add to
`Cargo.toml` in Step 1:

```toml
[dev-dependencies]
serde_json = "1"
```

`week2/md/tests/contract.rs` (Course Requirement: the default contract run
passes the three physics bounds):

```rust
//! The full default-contract acceptance test (ordinary, non-ignored).
//!
//! Course Requirement: with the contract defaults (N = 100, rho = 0.8,
//! T = 0.5, dt = 0.01, eq 2000, steps 10000, sample 50, seed 2026) the run
//! produces exactly 200 frames and `md check` passes all three bounds:
//! secular drift < 2e-3, |T_speed - 0.5| < 0.05, chi2_22 < 2.

use md::checker::check_artifacts;
use md::io::read_artifacts;
use md::simulate::{SimConfig, run_simulation};

#[test]
fn default_contract_run_passes_the_three_physics_bounds() {
    let config = SimConfig::default();
    let frames = run_simulation(&config);

    // Exactly 200 frames: steps/sample_every [Course Requirement].
    assert_eq!(frames.len(), 10000 / 50);
    // Saved steps are 50, 100, ..., 10000; step 0 never saved.
    for (idx, frame) in frames.iter().enumerate() {
        assert_eq!(frame.step, (idx + 1) * 50);
        assert!((frame.t - frame.step as f64 * config.dt).abs() < 1e-12);
    }

    // Write to a temp dir and run the real checker end to end.
    let out = std::env::temp_dir().join(format!("md-contract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    md::io::write_artifacts(&out, &config.into(), &frames).expect("write artifacts");

    let (run_config, traj) = read_artifacts(&out).expect("read artifacts");
    assert_eq!(run_config.integrator, "velocity-verlet");
    assert_eq!(traj.len(), 200);

    let report = check_artifacts(&out).expect("check must succeed on valid artifacts");
    assert!(report.drift_pass, "drift = {}", report.drift);
    assert!(report.temperature_pass, "T_speed = {}", report.t_speed);
    assert!(report.chi2_pass, "chi2_22 = {}", report.chi2_22);

    let _ = std::fs::remove_dir_all(&out);
}
```

- [ ] **Step 3: Run the tests to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --no-run`
Expected: compilation FAILS with unresolved imports
(`md::pair`, `md::system`, `md::fluid`, `md::simulate`, `md::io`,
`md::checker` do not exist). This is the first RED stage and it contains
all four course-required tests. (The existing hello-world `main.rs` still
compiles, so `CARGO_BIN_EXE_md` resolves; only the new library modules are
missing.)

- [ ] **Step 4: Commit the failing tests**

```bash
git add week2/md/Cargo.toml week2/md/Cargo.lock week2/md/.gitignore \
        week2/md/tests/pair.rs week2/md/tests/fluid.rs week2/md/tests/cli.rs \
        week2/md/tests/contract.rs
git commit -m "test: add failing Part 4 acceptance tests"
```

Strict ordering: this RED commit of the failing tests happens BEFORE any
CLI skeleton or other implementation code is created.

---

### Task 1b: CLI skeleton (after the RED commit)

**Files:**
- Create: `week2/md/src/cli.rs`
- Modify: `week2/md/src/main.rs`, `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: clap (Task 1 Step 1).
- Produces: `md::cli::{Cli, Command, RunArgs}` with `pub fn main() -> i32`
  dispatching to `todo!()` stubs; the contract flag defaults live here and
  later tasks replace the stubs (`Run` in Task 8, `Check` in Task 10,
  `Video` in Task 13).

- [ ] **Step 1: Write the CLI skeleton**

`week2/md/src/cli.rs`:

```rust
//! Command-line interface: one `md` binary with run / check / video.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Parameters accepted by `md run`. Defaults are the course contract values.
#[derive(Debug, Parser)]
pub struct RunArgs {
    /// Number of atoms; must be a perfect square.
    #[arg(long, default_value_t = 100)]
    pub n: usize,
    /// Number density in reduced units.
    #[arg(long, default_value_t = 0.8)]
    pub rho: f64,
    /// Target temperature in reduced units.
    #[arg(long, default_value_t = 0.5)]
    pub temperature: f64,
    /// Time step.
    #[arg(long, default_value_t = 0.01)]
    pub dt: f64,
    /// Equilibration steps (thermostat on).
    #[arg(long, default_value_t = 2000)]
    pub eq_steps: usize,
    /// Production steps (thermostat off).
    #[arg(long, default_value_t = 10000)]
    pub steps: usize,
    /// Save every this many production steps.
    #[arg(long, default_value_t = 50)]
    pub sample_every: usize,
    /// RNG seed for the initial Gaussian velocities.
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Output directory (relative to the working directory).
    #[arg(long, default_value = "artifacts")]
    pub out: PathBuf,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Simulate and record an equilibrium Lennard-Jones fluid.
    Run(RunArgs),
    /// Independently recompute the physics of a saved run.
    Check {
        /// Directory containing run.json and traj.jsonl.
        artifacts: PathBuf,
    },
    /// Render the trajectory and g(r) to an MP4 via ffmpeg.
    Video {
        /// Directory containing run.json and traj.jsonl.
        artifacts: PathBuf,
        /// Output MP4 path (required explicitly).
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Debug, Parser)]
#[command(name = "md", about = "Week 2 molecular dynamics CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// CLI entry point. Returns the process exit code.
pub fn main() -> i32 {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(_) => todo!("md run (Task 8)"),
        Command::Check { .. } => todo!("md check (Task 10)"),
        Command::Video { .. } => todo!("md video (Task 13)"),
    }
}
```

`week2/md/src/main.rs` becomes:

```rust
fn main() {
    std::process::exit(md::cli::main());
}
```

`week2/md/src/lib.rs` gains (and only gains) these lines:

```rust
mod cli;
pub use cli::{Cli, Command, RunArgs};
```

- [ ] **Step 2: Verify the crate still builds and existing tests pass**

Run: `cargo test --manifest-path md/Cargo.toml --lib --test lj --test dimer`
Expected: PASS (the new `tests/*.rs` files from Task 1 remain RED — that
is expected and correct; they are turned green by Tasks 2–11.)

- [ ] **Step 3: Commit**

```bash
git add week2/md/src/cli.rs week2/md/src/main.rs week2/md/src/lib.rs
git commit -m "feat: add md CLI skeleton with contract subcommands"
```

---

### Task 2: Periodic geometry and triangular lattice (system.rs)

**Files:**
- Create: `week2/md/src/system.rs`
- Modify: `week2/md/src/lib.rs` (add `mod system; pub use system::...;`)

**Interfaces:**
- Consumes: `md::State`, `md::Vec2` (existing).
- Produces:
  - `pub struct Box2 { pub lx: f64, pub ly: f64 }`
  - `impl Box2 { pub fn new(n: usize, rho: f64) -> Box2 }`
  - `pub fn side(n: usize) -> Option<usize>` — Some(sqrt) if perfect square
  - `pub fn lattice_state(n: usize, rho: f64) -> State` — triangular
    lattice, zero velocities
  - `pub fn minimum_image(d: f64, l: f64) -> f64`
  - `pub fn wrap(p: f64, l: f64) -> f64`
  - `pub fn lattice_constants(rho: f64) -> (f64, f64)` — (a, h)

- [ ] **Step 1: Write the failing unit tests** (in `system.rs`'s
  `#[cfg(test)] mod tests`; they fail to compile until the module is
  declared, which is the RED):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn lattice_constants_match_the_sheet() {
        let (a, h) = lattice_constants(0.8);
        let a_expected = (2.0 / (3f64.sqrt() * 0.8)).sqrt();
        assert!((a - a_expected).abs() < EPS);
        assert!((h - 3f64.sqrt() / 2.0 * a).abs() < EPS);
        // rho = 1/(a*h).
        assert!((1.0 / (a * h) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn default_box_is_ten_by_ten_lattice() {
        let bx = Box2::new(100, 0.8);
        let (a, h) = lattice_constants(0.8);
        assert!((bx.lx - 10.0 * a).abs() < EPS);
        assert!((bx.ly - 10.0 * h).abs() < EPS);
    }

    #[test]
    fn lattice_coordinates_follow_the_sheet_formula() {
        let state = lattice_state(100, 0.8);
        let (a, h) = lattice_constants(0.8);
        // Atom (i=0,j=0): (0,0). Index = j*10 + i.
        assert!((state.positions[0][0] - 0.0).abs() < EPS);
        // Atom (i=0,j=1): x = 0.5*a, y = h.
        assert!((state.positions[10][0] - 0.5 * a).abs() < EPS);
        assert!((state.positions[10][1] - h).abs() < EPS);
        // Atom (i=3,j=2): x = 3*a (j even), y = 2*h.
        assert!((state.positions[23][0] - 3.0 * a).abs() < EPS);
        assert!((state.positions[23][1] - 2.0 * h).abs() < EPS);
        assert_eq!(state.positions.len(), 100);
        assert!(state.velocities.iter().all(|v| *v == [0.0, 0.0]));
    }

    #[test]
    fn side_detects_perfect_squares() {
        assert_eq!(side(100), Some(10));
        assert_eq!(side(16), Some(4));
        assert_eq!(side(50), None);
        assert_eq!(side(2), None);
    }

    #[test]
    fn minimum_image_and_wrap_properties() {
        // d - L*round(d/L).
        assert!((minimum_image(2.7, 5.0) - 2.7).abs() < EPS);
        assert!((minimum_image(-2.7, 5.0) + 2.7).abs() < EPS);
        assert!((minimum_image(2.6, 5.0) + 2.4).abs() < EPS);
        // 0 <= wrap(p) < L for a sweep of values.
        for p in [-0.3, 0.0, 2.5, 5.0, 7.3, 12.9] {
            let w = wrap(p, 5.0);
            assert!((0.0..5.0).contains(&w), "wrap({p}) = {w}");
        }
    }
}
```

- [ ] **Step 2: RED**

Run: `cargo test --manifest-path md/Cargo.toml system`
Expected: compile error — `system` module not declared yet. (Declare the
module in `lib.rs` with an empty body first if you prefer a runtime-fail
RED; either way, commit only after Step 4.)

- [ ] **Step 3: Implement `system.rs`**

```rust
//! Periodic geometry: triangular lattice, box, minimum image, wrapping.

use crate::state::State;
use crate::Vec2;

/// Triangular-lattice constants (a, h) at density `rho`.
/// a = sqrt(2/(sqrt(3)*rho)), h = sqrt(3)/2 * a  [Course Requirement].
pub fn lattice_constants(rho: f64) -> (f64, f64) {
    let a = (2.0 / (3f64.sqrt() * rho)).sqrt();
    let h = 3f64.sqrt() / 2.0 * a;
    (a, h)
}

/// Side length of the sqrt(N) x sqrt(N) lattice, or None if N is not a
/// perfect square. [Suggestion] non-square N is rejected.
pub fn side(n: usize) -> Option<usize> {
    let s = (n as f64).sqrt() as usize;
    if s * s == n { Some(s) } else { None }
}

/// Rectangular periodic box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box2 {
    pub lx: f64,
    pub ly: f64,
}

impl Box2 {
    /// Box for n atoms at density rho: Lx = sqrt(N)*a, Ly = sqrt(N)*h.
    pub fn new(n: usize, rho: f64) -> Box2 {
        let s = side(n).expect("n must be a perfect square");
        let (a, h) = lattice_constants(rho);
        Box2 { lx: s as f64 * a, ly: s as f64 * h }
    }
}

/// Initial triangular-lattice state: x = (i + 0.5*(j%2))*a, y = j*h,
/// i,j = 0..sqrt(N)-1; zero velocities. Atom index = j*n_side + i.
pub fn lattice_state(n: usize, rho: f64) -> State {
    let s = side(n).expect("n must be a perfect square");
    let (a, h) = lattice_constants(rho);
    let mut positions = Vec::with_capacity(n);
    for j in 0..s {
        for i in 0..s {
            positions.push(Vec2::from([
                (i as f64 + 0.5 * ((j % 2) as f64)) * a,
                j as f64 * h,
            ]));
        }
    }
    State { positions, velocities: vec![[0.0, 0.0]; n] }
}

/// Minimum-image displacement on one axis: d - L*round(d/L).
pub fn minimum_image(d: f64, l: f64) -> f64 {
    d - l * (d / l).round()
}

/// Wrap a coordinate into [0, L). Velocities are never wrapped.
pub fn wrap(p: f64, l: f64) -> f64 {
    let w = p - l * p.div_euclid(l);
    if w < 0.0 { w + l } else { w }
}
```

Add to `lib.rs`: `mod system; pub use system::{Box2, lattice_constants, lattice_state, minimum_image, side, wrap};`

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml system`
Expected: PASS (5 tests). Also run
`cargo test --manifest-path md/Cargo.toml` — `tests/fluid.rs` still fails
to compile (expected; it is RED from Task 1) — use
`cargo test --manifest-path md/Cargo.toml --lib` plus the integration
files that already compile (`tests/pair.rs` still RED) to confirm no
regression in `tests/lj.rs`/`tests/dimer.rs`.

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/system.rs week2/md/src/lib.rs
git commit -m "feat: add periodic box, triangular lattice, minimum image, wrapping"
```

---

### Task 3: Shifted-cutoff pair interaction (pair.rs)

**Files:**
- Create: `week2/md/src/pair.rs`
- Modify: `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `md::lj_energy`, `md::lj_force` (existing, unchanged).
- Produces:
  - `pub const RC: f64 = 2.5;`
  - `pub fn shifted_energy(r: f64) -> f64`
  - `pub fn shifted_force(r: f64) -> f64`

- [ ] **Step 1: RED already exists** — `tests/pair.rs` from Task 1 is the
failing acceptance suite. Run it:

Run: `cargo test --manifest-path md/Cargo.toml --test pair`
Expected: compile FAIL (`md::pair` unresolved).

- [ ] **Step 2: Implement `pair.rs`**

```rust
//! Shifted-cutoff Lennard-Jones pair functions (rc = 2.5).
//!
//! U_cut(r) = lj_energy(r) - lj_energy(rc) for r < rc, else 0: continuous,
//! zero at rc. The force is the plain Part 2 LJ force below rc and zero
//! at/above rc — it has a small jump at rc [Course Requirement].

use crate::{lj_energy, lj_force};

/// Cutoff distance in reduced units [Course Requirement].
pub const RC: f64 = 2.5;

/// Shifted pair energy.
pub fn shifted_energy(r: f64) -> f64 {
    if r < RC {
        lj_energy(r) - lj_energy(RC)
    } else {
        0.0
    }
}

/// Cut pair force: plain LJ below rc, zero at and above rc.
pub fn shifted_force(r: f64) -> f64 {
    if r < RC {
        lj_force(r)
    } else {
        0.0
    }
}
```

Add to `lib.rs`: `mod pair; pub use pair::{RC, shifted_energy, shifted_force};`

- [ ] **Step 3: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test pair`
Expected: 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/pair.rs week2/md/src/lib.rs
git commit -m "feat: add shifted-cutoff Lennard-Jones pair energy and force"
```

---

### Task 4: Periodic fluid accelerations and energies (fluid.rs)

**Files:**
- Create: `week2/md/src/fluid.rs`
- Modify: `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `md::pair::{shifted_energy, shifted_force}`,
  `md::system::{Box2, minimum_image}`, `md::State`, `md::kinetic_energy`.
- Produces:
  - `pub fn fluid_accelerations(state: &State, bx: &Box2) -> Vec<Vec2>`
  - `pub fn fluid_potential_energy(state: &State, bx: &Box2) -> f64`

- [ ] **Step 1: RED already exists** — `tests/fluid.rs` from Task 1.

Run: `cargo test --manifest-path md/Cargo.toml --test fluid`
Expected: compile FAIL (`md::fluid` unresolved).

- [ ] **Step 2: Implement `fluid.rs`**

```rust
//! Periodic-boundary fluid forces and energies with the shifted cutoff.
//!
//! O(N^2) pair loop; no cell lists (Part 5).

use crate::pair::{shifted_energy, shifted_force};
use crate::state::State;
use crate::system::{Box2, minimum_image};
use crate::Vec2;

/// Accelerations (mass 1) with minimum-image displacements and the
/// shifted-cutoff force. Pair contributions are antisymmetric.
pub fn fluid_accelerations(state: &State, bx: &Box2) -> Vec<Vec2> {
    let n = state.positions.len();
    let mut acc = vec![[0.0, 0.0]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = minimum_image(state.positions[i][0] - state.positions[j][0], bx.lx);
            let dy = minimum_image(state.positions[i][1] - state.positions[j][1], bx.ly);
            let r = (dx * dx + dy * dy).sqrt();
            let f_over_r = shifted_force(r) / r;
            let fx = f_over_r * dx;
            let fy = f_over_r * dy;
            acc[i][0] += fx;
            acc[i][1] += fy;
            acc[j][0] -= fx;
            acc[j][1] -= fy;
        }
    }
    acc
}

/// Sum of shifted pair energies over minimum-image pairs.
pub fn fluid_potential_energy(state: &State, bx: &Box2) -> f64 {
    let mut energy = 0.0;
    for i in 0..state.positions.len() {
        for j in (i + 1)..state.positions.len() {
            let dx = minimum_image(state.positions[i][0] - state.positions[j][0], bx.lx);
            let dy = minimum_image(state.positions[i][1] - state.positions[j][1], bx.ly);
            energy += shifted_energy((dx * dx + dy * dy).sqrt());
        }
    }
    energy
}
```

Add to `lib.rs`: `mod fluid; pub use fluid::{fluid_accelerations, fluid_potential_energy};`

Note: `r = 0` cannot occur for distinct lattice atoms at rho = 0.8 and
never arises at runtime (repulsion prevents overlap); no special case is
added (YAGNI).

- [ ] **Step 3: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test fluid`
Expected: 5 PASS. Also confirm Part 2/3 regression:
`cargo test --manifest-path md/Cargo.toml --lib --test lj --test dimer` → PASS.

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/fluid.rs week2/md/src/lib.rs
git commit -m "feat: add periodic fluid accelerations and shifted energies"
```

---

### Task 5: Gaussian initialization, COM removal, thermostat (thermostat.rs)

**Files:**
- Create: `week2/md/src/thermostat.rs`
- Modify: `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `md::State`, `md::kinetic_energy`.
- Produces:
  - `pub fn gaussian_velocities(n: usize, temperature: f64, seed: u64) -> Vec<Vec2>`
  - `pub fn remove_com_velocity(velocities: &mut [Vec2])`
  - `pub fn thermodynamic_temperature(velocities: &[Vec2]) -> f64`
  - `pub fn rescale_to(velocities: &mut [Vec2], target: f64)`

- [ ] **Step 1: Write the failing unit tests** (in `thermostat.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn gaussian_velocities_are_seeded_deterministic_with_variance_t() {
        // Course Requirement: seeded, independent, mean 0, variance T.
        let v1 = gaussian_velocities(100, 0.5, 2026);
        let v2 = gaussian_velocities(100, 0.5, 2026);
        assert_eq!(v1, v2, "same seed must reproduce identical velocities");
        let v_other = gaussian_velocities(100, 0.5, 7);
        assert_ne!(v1, v_other, "different seed must differ");
        // Empirical variance over 200 components ~ T (loose bound).
        let mean: f64 = v1.iter().flat_map(|v| v.iter()).sum::<f64>() / 200.0;
        assert!(mean.abs() < 0.2, "mean too far from 0: {mean}");
        let var: f64 = v1.iter().flat_map(|v| v.iter()).map(|x| (x - mean) * (x - mean)).sum::<f64>() / 200.0;
        assert!((var - 0.5).abs() < 0.2, "variance {var} far from 0.5");
    }

    #[test]
    fn com_removal_zeroes_the_mean() {
        let mut v = vec![[1.0, 2.0], [3.0, -1.0], [-2.0, 5.0]];
        remove_com_velocity(&mut v);
        for axis in 0..2 {
            let m = v.iter().map(|p| p[axis]).sum::<f64>() / v.len() as f64;
            assert!(m.abs() < EPS, "axis {axis} mean {m}");
        }
    }

    #[test]
    fn thermodynamic_temperature_uses_two_n_minus_two() {
        // 4 atoms at rest except one: E_kin = 0.5*2 = 1, T = 2*1/(8-2).
        let v = vec![[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        assert!((thermodynamic_temperature(&v) - 2.0 / 6.0).abs() < EPS);
    }

    #[test]
    fn rescale_sets_the_thermodynamic_temperature() {
        let mut v = vec![[1.0, 0.0]; 10];
        rescale_to(&mut v, 0.5);
        assert!((thermodynamic_temperature(&v) - 0.5).abs() < 1e-12);
        // Uniform rescale preserves zero COM.
        let mx = v.iter().map(|p| p[0]).sum::<f64>() / 10.0;
        assert!(mx.abs() < EPS);
    }
}
```

- [ ] **Step 2: RED**

Run: `cargo test --manifest-path md/Cargo.toml thermostat`
Expected: compile FAIL (module undeclared).

- [ ] **Step 3: Implement `thermostat.rs`**

```rust
//! Seeded Gaussian initialization, COM removal, and the simple thermostat.
//!
//! RNG details are [Suggestion]; the Course Requirement is seeded
//! independent Gaussian components with variance = run temperature.

use crate::Vec2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Independent Gaussian velocity components, mean 0, variance `temperature`,
/// drawn atom-major (vx, vy per atom), seeded by `seed`.
pub fn gaussian_velocities(n: usize, temperature: f64, seed: u64) -> Vec<Vec2> {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigma = temperature.sqrt();
    (0..n)
        .map(|_| {
            let vx: f64 = rng.sample(rand_distr::Normal::new(0.0, sigma).unwrap());
            let vy: f64 = rng.sample(rand_distr::Normal::new(0.0, sigma).unwrap());
            [vx, vy]
        })
        .collect()
}

/// Subtract the component-wise mean velocity (once, per the schedule).
pub fn remove_com_velocity(velocities: &mut [Vec2]) {
    let n = velocities.len() as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for v in velocities.iter() {
        mx += v[0];
        my += v[1];
    }
    mx /= n;
    my /= n;
    for v in velocities.iter_mut() {
        v[0] -= mx;
        v[1] -= my;
    }
}

/// T_thermo = 2*E_kin/(2N - 2) [Course Requirement].
pub fn thermodynamic_temperature(velocities: &[Vec2]) -> f64 {
    let e_kin: f64 = velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum();
    let n = velocities.len() as f64;
    2.0 * e_kin / (2.0 * n - 2.0)
}

/// Rescale every component by sqrt(target / T_thermo) [Course Requirement].
pub fn rescale_to(velocities: &mut [Vec2], target: f64) {
    let scale = (target / thermodynamic_temperature(velocities)).sqrt();
    for v in velocities.iter_mut() {
        v[0] *= scale;
        v[1] *= scale;
    }
}
```

Add to `lib.rs`: `mod thermostat; pub use thermostat::{gaussian_velocities, remove_com_velocity, rescale_to, thermodynamic_temperature};`

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml thermostat`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/thermostat.rs week2/md/src/lib.rs
git commit -m "feat: add seeded Gaussian velocities, COM removal, thermostat"
```

---

### Task 6: Equilibration / production simulation driver (simulate.rs)

**Files:**
- Create: `week2/md/src/simulate.rs`
- Modify: `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `lattice_state`, `Box2`, `fluid_accelerations`,
  `fluid_potential_energy`, `gaussian_velocities`, `remove_com_velocity`,
  `rescale_to`, `kinetic_energy`, `VelocityVerlet`, `Integrator`, `wrap`.
- Produces:
  - `pub struct SimConfig { pub n: usize, pub rho: f64, pub temperature: f64,
    pub dt: f64, pub eq_steps: usize, pub steps: usize, pub sample_every:
    usize, pub seed: u64 }` with `Default` = contract values
    (100, 0.8, 0.5, 0.01, 2000, 10000, 50, 2026)
  - `pub struct Frame { pub step: usize, pub t: f64, pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>, pub e_pot: f64, pub e_kin: f64 }`
  - `pub fn run_simulation(config: &SimConfig) -> Vec<Frame>`

- [ ] **Step 1: Write the failing unit tests** (in `simulate.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> SimConfig {
        SimConfig { n: 16, rho: 0.8, temperature: 0.5, dt: 0.01,
                    eq_steps: 100, steps: 200, sample_every: 25, seed: 2026 }
    }

    #[test]
    fn defaults_are_the_course_contract_values() {
        let c = SimConfig::default();
        assert_eq!(c.n, 100);
        assert_eq!(c.rho, 0.8);
        assert_eq!(c.temperature, 0.5);
        assert_eq!(c.dt, 0.01);
        assert_eq!(c.eq_steps, 2000);
        assert_eq!(c.steps, 10000);
        assert_eq!(c.sample_every, 50);
        assert_eq!(c.seed, 2026);
    }

    #[test]
    fn frames_follow_sample_every_and_skip_step_zero() {
        let frames = run_simulation(&small_config());
        assert_eq!(frames.len(), 8);
        for (idx, f) in frames.iter().enumerate() {
            assert_eq!(f.step, (idx + 1) * 25);
            assert!((f.t - f.step as f64 * 0.01).abs() < 1e-12);
        }
    }

    #[test]
    fn positions_stay_wrapped_and_energies_finite() {
        let frames = run_simulation(&small_config());
        let bx = Box2::new(16, 0.8);
        for f in &frames {
            for p in &f.pos {
                assert!((0.0..bx.lx).contains(&p[0]), "x = {}", p[0]);
                assert!((0.0..bx.ly).contains(&p[1]), "y = {}", p[1]);
            }
            assert!(f.e_pot.is_finite());
            assert!(f.e_kin.is_finite());
        }
    }

    #[test]
    fn thermostat_targets_the_run_temperature_and_zero_com() {
        // After equilibration (ending with a rescale at step eq_steps)
        // the thermodynamic temperature is the target.
        let frames = run_simulation(&small_config());
        let last = frames.last().unwrap();
        let t = crate::thermostat::thermodynamic_temperature(&last.vel);
        assert!((t - 0.5).abs() < 1e-9, "T after final rescale = {t}");
        // COM stays zero (uniform rescales preserve it).
        let mx = last.vel.iter().map(|v| v[0]).sum::<f64>() / last.vel.len() as f64;
        let my = last.vel.iter().map(|v| v[1]).sum::<f64>() / last.vel.len() as f64;
        assert!(mx.abs() < 1e-10 && my.abs() < 1e-10);
    }

    #[test]
    fn thermostat_event_count_is_schedule_b() {
        // [Course Requirement: Schedule B] initial rescale + one after every
        // 50th equilibration step: 1 + eq_steps/50 events; none in production.
        assert_eq!(thermostat_events(2000), 1 + 2000 / 50);
        assert_eq!(thermostat_events(100), 1 + 100 / 50);
    }
}
```

- [ ] **Step 2: RED**

Run: `cargo test --manifest-path md/Cargo.toml simulate`
Expected: compile FAIL.

- [ ] **Step 3: Implement `simulate.rs`**

```rust
//! Equilibration + production driver for the periodic fluid.

use crate::fluid::{fluid_accelerations, fluid_potential_energy};
use crate::integrator::{Integrator, VelocityVerlet};
use crate::state::State;
use crate::system::{Box2, lattice_state, wrap};
use crate::thermostat::{gaussian_velocities, remove_com_velocity, rescale_to};
use crate::Vec2;

/// Number of equilibration steps between thermostat events [Course Req.].
pub const THERMOSTAT_INTERVAL: usize = 50;

/// Run parameters; Default is the course contract run.
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig { n: 100, rho: 0.8, temperature: 0.5, dt: 0.01,
                    eq_steps: 2000, steps: 10000, sample_every: 50, seed: 2026 }
    }
}

/// One saved production frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    pub e_pot: f64,
    pub e_kin: f64,
}

/// Schedule B thermostat event count (doc/test helper).
pub fn thermostat_events(eq_steps: usize) -> usize {
    1 + eq_steps / THERMOSTAT_INTERVAL
}

/// Run equilibration (thermostat every 50 steps, Schedule B) then
/// thermostat-free production, saving every `sample_every` steps
/// (step 0 never saved).
pub fn run_simulation(config: &SimConfig) -> Vec<Frame> {
    let bx = Box2::new(config.n, config.rho);
    let mut state = lattice_state(config.n, config.rho);
    state.velocities = gaussian_velocities(config.n, config.temperature, config.seed);
    remove_com_velocity(&mut state.velocities);
    rescale_to(&mut state.velocities, config.temperature);

    let accelerations = |s: &State| fluid_accelerations(s, &bx);
    let mut verlet = VelocityVerlet::default();
    verlet.initialize(&state, &accelerations);

    // Equilibration: rescale after each step s with s % 50 == 0.
    for _ in 0..config.eq_steps {
        verlet.step(&mut state, config.dt, &accelerations);
        wrap_state(&mut state, &bx);
        if verlet_steps_done % THERMOSTAT_INTERVAL == 0 { /* see below */ }
    }
    // (The loop counter is tracked explicitly in the implementation;
    //  the pseudocode above is expanded in the real code with a
    //  `for step in 1..=config.eq_steps` counter and
    //  `if step % THERMOSTAT_INTERVAL == 0 { rescale_to(...) }`.)

    // Production: thermostat OFF, time starts at zero.
    let mut frames = Vec::with_capacity(config.steps / config.sample_every);
    for step in 1..=config.steps {
        verlet.step(&mut state, config.dt, &accelerations);
        wrap_state(&mut state, &bx);
        if step % config.sample_every == 0 {
            frames.push(Frame {
                step,
                t: step as f64 * config.dt,
                pos: state.positions.clone(),
                vel: state.velocities.clone(),
                e_pot: fluid_potential_energy(&state, &bx),
                e_kin: crate::kinetic_energy(&state),
            });
        }
    }
    frames
}

fn wrap_state(state: &mut State, bx: &Box2) {
    for p in state.positions.iter_mut() {
        p[0] = wrap(p[0], bx.lx);
        p[1] = wrap(p[1], bx.ly);
    }
}
```

Implementation note: write the equilibration loop concretely as
`for step in 1..=config.eq_steps { verlet.step(...); wrap_state(...); if step % THERMOSTAT_INTERVAL == 0 { rescale_to(&mut state.velocities, config.temperature); } }`
(the snippet above shows it once expanded and once condensed — use the
concrete form; remove the condensed duplicate).

Add to `lib.rs`: `mod simulate; pub use simulate::{Frame, SimConfig, THERMOSTAT_INTERVAL, run_simulation, thermostat_events};`

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml simulate`
Expected: 5 PASS. Note `thermostat_targets_the_run_temperature_and_zero_com`
holds because Schedule B ends with a rescale at step `eq_steps` (100 is a
multiple of 50).

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/simulate.rs week2/md/src/lib.rs
git commit -m "feat: add Schedule-B equilibration and production driver"
```

---

### Task 7: run.json / traj.jsonl I/O (io.rs)

**Files:**
- Create: `week2/md/src/io.rs`
- Modify: `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `SimConfig`, `Frame`, `Box2`.
- Produces:
  - `pub struct RunConfig { pub n: usize, pub rho: f64, pub box_dim: [f64; 2],
    pub dt: f64, pub temperature: f64, pub eq_steps: usize, pub steps: usize,
    pub sample_every: usize, pub seed: u64, pub integrator: String }`
    (serde Serialize/Deserialize; JSON keys exactly `n, rho, box, dt,
    temperature, eq_steps, steps, sample_every, seed, integrator`)
  - `impl From<SimConfig> for RunConfig`
  - `pub fn write_artifacts(dir: &Path, run: &RunConfig, frames: &[Frame]) -> std::io::Result<()>`
  - `pub fn read_artifacts(dir: &Path) -> Result<(RunConfig, Vec<Frame>), String>`

- [ ] **Step 1: Write the failing unit tests** (in `io.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate::SimConfig;
    use std::path::PathBuf;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("md-io-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn sample_frames() -> Vec<crate::simulate::Frame> {
        vec![
            crate::simulate::Frame { step: 50, t: 0.5,
                pos: vec![[0.1, 0.2]], vel: vec![[0.3, 0.4]],
                e_pot: -1.5, e_kin: 0.25 },
            crate::simulate::Frame { step: 100, t: 1.0,
                pos: vec![[0.5, 0.6]], vel: vec![[0.7, 0.8]],
                e_pot: -1.4, e_kin: 0.30 },
        ]
    }

    #[test]
    fn round_trip_preserves_run_fields_and_frames() {
        let dir = temp("rt");
        let run = RunConfig::from(SimConfig::default());
        write_artifacts(&dir, &run, &sample_frames()).unwrap();
        let (run2, frames2) = read_artifacts(&dir).unwrap();
        assert_eq!(run2.integrator, "velocity-verlet");
        assert_eq!(run2.n, 100);
        assert_eq!(run2.sample_every, 50);
        assert_eq!(run2.box_dim, run.box_dim);
        assert_eq!(frames2.len(), 2);
        assert_eq!(frames2[0].step, 50);
        assert!((frames2[1].t - 1.0).abs() < 1e-12);
        assert_eq!(frames2[0].pos, vec![[0.1, 0.2]]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_json_keys_are_exactly_the_course_contract() {
        let dir = temp("keys");
        let run = RunConfig::from(SimConfig::default());
        write_artifacts(&dir, &run, &sample_frames()).unwrap();
        let text = std::fs::read_to_string(dir.join("run.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        for key in ["n", "rho", "box", "dt", "temperature", "eq_steps",
                    "steps", "sample_every", "seed", "integrator"] {
            assert!(keys.contains(&key), "missing key {key}");
        }
        // traj.jsonl: one JSON object per line with the frame fields.
        let traj = std::fs::read_to_string(dir.join("traj.jsonl")).unwrap();
        let line: serde_json::Value =
            serde_json::from_str(traj.lines().next().unwrap()).unwrap();
        for key in ["step", "t", "pos", "vel", "E_pot", "E_kin"] {
            assert!(line.get(key).is_some(), "traj frame missing {key}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_errors_on_missing_or_malformed_files() {
        let dir = temp("missing");
        assert!(read_artifacts(&dir).is_err());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("run.json"), "{ not json").unwrap();
        std::fs::write(dir.join("traj.jsonl"), "").unwrap();
        assert!(read_artifacts(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
```

- [ ] **Step 2: RED**

Run: `cargo test --manifest-path md/Cargo.toml io`
Expected: compile FAIL.

- [ ] **Step 3: Implement `io.rs`**

```rust
//! run.json / traj.jsonl artifact reading and writing.

use crate::simulate::{Frame, SimConfig};
use crate::system::Box2;
use crate::Vec2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;

/// Metadata written to run.json; field order matches the course contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
}

impl From<SimConfig> for RunConfig {
    fn from(c: SimConfig) -> Self {
        let bx = Box2::new(c.n, c.rho);
        RunConfig {
            n: c.n, rho: c.rho, box_dim: [bx.lx, bx.ly], dt: c.dt,
            temperature: c.temperature, eq_steps: c.eq_steps, steps: c.steps,
            sample_every: c.sample_every, seed: c.seed,
            integrator: "velocity-verlet".to_string(),
        }
    }
}

impl From<&SimConfig> for RunConfig {
    fn from(c: &SimConfig) -> Self { RunConfig::from(*c) }
}

/// Serialized frame schema for traj.jsonl.
#[derive(Serialize, Deserialize)]
struct FrameJson<'a> {
    step: usize,
    t: f64,
    pos: &'a [Vec2],
    vel: &'a [Vec2],
    #[serde(rename = "E_pot")]
    e_pot: f64,
    #[serde(rename = "E_kin")]
    e_kin: f64,
}

#[derive(Deserialize)]
struct FrameJsonOwned {
    step: usize,
    t: f64,
    pos: Vec<Vec2>,
    vel: Vec<Vec2>,
    #[serde(rename = "E_pot")]
    e_pot: f64,
    #[serde(rename = "E_kin")]
    e_kin: f64,
}

/// Write run.json and traj.jsonl into `dir` (created as needed).
pub fn write_artifacts(dir: &Path, run: &RunConfig, frames: &[Frame]) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join("run.json"), serde_json::to_string_pretty(run).unwrap())?;
    let mut file = fs::File::create(dir.join("traj.jsonl"))?;
    for f in frames {
        let line = serde_json::to_string(&FrameJson {
            step: f.step, t: f.t, pos: &f.pos, vel: &f.vel,
            e_pot: f.e_pot, e_kin: f.e_kin,
        }).unwrap();
        writeln!(file, "{line}")?;
    }
    Ok(())
}

/// Read run.json and traj.jsonl from `dir`; Err(msg) on any malformed input.
pub fn read_artifacts(dir: &Path) -> Result<(RunConfig, Vec<Frame>), String> {
    let run_text = fs::read_to_string(dir.join("run.json"))
        .map_err(|e| format!("cannot read run.json: {e}"))?;
    let run: RunConfig = serde_json::from_str(&run_text)
        .map_err(|e| format!("malformed run.json: {e}"))?;
    let traj_text = fs::read_to_string(dir.join("traj.jsonl"))
        .map_err(|e| format!("cannot read traj.jsonl: {e}"))?;
    let mut frames = Vec::new();
    for (i, line) in traj_text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let f: FrameJsonOwned = serde_json::from_str(line)
            .map_err(|e| format!("malformed traj.jsonl line {}: {e}", i + 1))?;
        frames.push(Frame { step: f.step, t: f.t, pos: f.pos, vel: f.vel,
                            e_pot: f.e_pot, e_kin: f.e_kin });
    }
    Ok((run, frames))
}
```

Add to `lib.rs`: `mod io; pub use io::{RunConfig, read_artifacts, write_artifacts};`

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml io`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/io.rs week2/md/src/lib.rs
git commit -m "feat: add run.json and traj.jsonl artifact I/O"
```

---

### Task 8: `md run` CLI wiring

**Files:**
- Modify: `week2/md/src/cli.rs`

**Interfaces:**
- Consumes: `RunArgs` (Task 1), `SimConfig`, `run_simulation`,
  `write_artifacts`, `RunConfig`, `side`.
- Produces: a working `md run` subcommand (exit code semantics).

- [ ] **Step 1: RED already exists** — `tests/cli.rs` from Task 1 (it
currently fails at the `todo!()` panic). Run it to confirm:

Run: `cargo test --manifest-path md/Cargo.toml --test cli`
Expected: FAIL (panics on `todo!("md run")`).

- [ ] **Step 2: Implement the `Run` arm in `cli.rs`**

```rust
fn run_command(args: &RunArgs) -> Result<(), String> {
    // [Suggestion] validation: perfect-square n; steps % sample_every == 0.
    if md::system::side(args.n).is_none() {
        return Err(format!("--n must be a perfect square, got {}", args.n));
    }
    if args.sample_every == 0 || args.steps % args.sample_every != 0 {
        return Err(format!(
            "--steps ({}) must be a positive multiple of --sample-every ({})",
            args.steps, args.sample_every
        ));
    }
    let config = SimConfig {
        n: args.n, rho: args.rho, temperature: args.temperature, dt: args.dt,
        eq_steps: args.eq_steps, steps: args.steps,
        sample_every: args.sample_every, seed: args.seed,
    };
    let frames = md::simulate::run_simulation(&config);
    md::io::write_artifacts(&args.out, &RunConfig::from(&config), &frames)
        .map_err(|e| format!("cannot write artifacts: {e}"))?;
    println!(
        "wrote {} frames to {}",
        frames.len(),
        args.out.join("traj.jsonl").display()
    );
    Ok(())
}
```

Replace the `Command::Run(_)` arm with:

```rust
Command::Run(args) => match run_command(&args) {
    Ok(()) => 0,
    Err(msg) => { eprintln!("error: {msg}"); 2 }
},
```

- [ ] **Step 3: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test cli`
Expected: 2 PASS (the small run writes files; validation errors exit
nonzero). `md run --out artifacts` is now equivalent to the full flag
form because every flag has the contract default.

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/cli.rs
git commit -m "feat: wire md run subcommand with contract defaults"
```

---

### Task 9: Checker metrics (metrics.rs)

**Files:**
- Create: `week2/md/src/metrics.rs`
- Modify: `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `Frame`, `Box2`, `fluid_potential_energy`, `minimum_image`.
- Produces:
  - `pub fn frame_total_energies(frames: &[Frame], bx: &Box2) -> Vec<f64>`
    (recomputed E_pot + stored-independent E_kin from raw vel)
  - `pub fn secular_drift(totals: &[f64]) -> f64`
  - `pub fn pooled_speeds(frames: &[Frame]) -> Vec<f64>`
  - `pub fn t_speed(speeds: &[f64]) -> f64`
  - `pub fn chi2_22(speeds: &[f64], t: f64) -> f64`
  - `pub fn bin_edges(t: f64) -> [f64; 25]` (b_0..b_23 finite, b_24 = ∞)
  - `pub fn radial_distribution(pos_frames: &[&[Vec2]], bx: &Box2, bins: usize) -> Vec<(f64, f64)>`

- [ ] **Step 1: Write the failing unit tests** (in `metrics.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec2;

    #[test]
    fn secular_drift_uses_k_is_floor_of_tenth() {
        // F = 10 frames -> k = 1: drift = |E_last - E_first| / |E_first|.
        let totals: Vec<f64> = (0..10).map(|i| 10.0 + 0.001 * i as f64).collect();
        let expected = (10.009 - 10.0) / 10.0;
        assert!((secular_drift(&totals) - expected).abs() < 1e-15);
        // F = 200 -> k = 20: means over first and last 20.
        let mut t = vec![5.0; 200];
        for (i, e) in t.iter_mut().enumerate() { *e += if i < 20 { 0.0 } else { 0.0 }; }
        assert!(secular_drift(&t).abs() < 1e-15);
    }

    #[test]
    fn t_speed_is_half_the_mean_squared_speed() {
        let speeds = vec![1.0, 2.0, 3.0]; // mean(v^2) = 14/3
        assert!((t_speed(&speeds) - 14.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn bin_edges_are_equal_probability_rayleigh() {
        let t = 0.5;
        let edges = bin_edges(t);
        assert_eq!(edges[0], 0.0);
        assert_eq!(edges[24], f64::INFINITY);
        // b_k = sqrt(-2 t ln(1 - k/24)).
        let expected = (-2.0 * t * (1.0 - 12.0 / 24.0).ln()).sqrt();
        assert!((edges[12] - expected).abs() < 1e-15);
    }

    #[test]
    fn chi2_22_is_small_for_rayleigh_samples_and_reduced() {
        // Synthetic speeds drawn from Rayleigh(T): chi2_22 should be ~1.
        // Deterministic construction via inverse CDF on a regular grid.
        let t = 0.5;
        let n = 24000;
        let speeds: Vec<f64> = (0..n)
            .map(|i| (-2.0 * t * (1.0 - (i as f64 + 0.5) / n as f64).ln()).sqrt())
            .collect();
        let ts = t_speed(&speeds);
        let c = chi2_22(&speeds, ts);
        assert!(c < 2.0, "chi2_22 = {c}");
        assert!(c > 1e-6, "chi2_22 suspiciously zero: {c}");
    }

    #[test]
    fn chi2_22_is_large_for_non_maxwell_speeds() {
        // All speeds in one place: the statistic blows up.
        let speeds = vec![0.5; 2400];
        let ts = t_speed(&speeds);
        assert!(chi2_22(&speeds, ts) > 2.0);
    }

    #[test]
    fn radial_distribution_of_a_uniform_box_is_about_one() {
        // Hand-built near-uniform positions on a coarse grid, no pairs in
        // the first tiny bin; middle bins of a uniform box average ~1.
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let mut pos = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                pos.push([(i as f64 + 0.5), (j as f64 + 0.5)]);
            }
        }
        let frames: Vec<&[Vec2]> = vec![&pos];
        let g = radial_distribution(&frames, &bx, 50);
        assert_eq!(g.len(), 50);
        // Mean over bins between r = 1 and r = 4 should be near 1.
        let mid: Vec<f64> = g.iter()
            .filter(|(r, _)| *r > 1.5 && *r < 4.0)
            .map(|(_, v)| *v)
            .collect();
        let mean: f64 = mid.iter().sum::<f64>() / mid.len() as f64;
        assert!((mean - 1.0).abs() < 0.2, "mean g = {mean}");
    }
}
```

- [ ] **Step 2: RED**

Run: `cargo test --manifest-path md/Cargo.toml metrics`
Expected: compile FAIL.

- [ ] **Step 3: Implement `metrics.rs`**

```rust
//! Independent physics metrics recomputed from raw trajectory data.

use crate::fluid::fluid_potential_energy;
use crate::simulate::Frame;
use crate::state::State;
use crate::system::{Box2, minimum_image};
use crate::Vec2;

/// Recomputed total energies per frame: shifted E_pot from raw wrapped
/// positions (minimum image) + E_kin from raw velocities.
pub fn frame_total_energies(frames: &[Frame], bx: &Box2) -> Vec<f64> {
    frames
        .iter()
        .map(|f| {
            let state = State { positions: f.pos.clone(), velocities: f.vel.clone() };
            fluid_potential_energy(&state, bx)
                + f.vel.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum::<f64>()
        })
        .collect()
}

/// |mean(E_last_k) - mean(E_first_k)| / |E0| with k = max(1, floor(F/10)).
pub fn secular_drift(totals: &[f64]) -> f64 {
    let f = totals.len();
    let k = std::cmp::max(1, f / 10);
    let first: f64 = totals[..k].iter().sum::<f64>() / k as f64;
    let last: f64 = totals[f - k..].iter().sum::<f64>() / k as f64;
    (last - first).abs() / totals[0].abs()
}

/// All speeds v = |velocity| pooled over frames and atoms.
pub fn pooled_speeds(frames: &[Frame]) -> Vec<f64> {
    frames
        .iter()
        .flat_map(|f| f.vel.iter())
        .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
        .collect()
}

/// T_speed = mean(v^2)/2 [Course Requirement].
pub fn t_speed(speeds: &[f64]) -> f64 {
    let n = speeds.len() as f64;
    speeds.iter().map(|s| s * s).sum::<f64>() / n / 2.0
}

/// 24 equal-probability Rayleigh bin edges at temperature t; b_24 = inf.
pub fn bin_edges(t: f64) -> [f64; 25] {
    let mut edges = [0.0; 25];
    for k in 1..24 {
        edges[k] = (-2.0 * t * (1.0 - k as f64 / 24.0).ln()).sqrt();
    }
    edges[24] = f64::INFINITY;
    edges
}

/// Reduced chi-square: (1/22) * sum_b (O_b - E_b)^2 / E_b with E_b = S/24.
/// Bin edges use the passed temperature (callers pass T_speed) [Course Req].
pub fn chi2_22(speeds: &[f64], t: f64) -> f64 {
    let edges = bin_edges(t);
    let s = speeds.len() as f64;
    let expected = s / 24.0;
    let mut stat = 0.0;
    for b in 0..24 {
        let observed = speeds
            .iter()
            .filter(|&&v| v >= edges[b] && v < edges[b + 1])
            .count() as f64;
        stat += (observed - expected).powi(2) / expected;
    }
    stat / 22.0
}

/// g(r) averaged over atoms and frames: unordered minimum-image pairs,
/// g_k = 2*H_k / (N*F*rho*pi*(r_out^2 - r_in^2)) [Course Requirement].
/// Returns (r_center, g) per bin.
pub fn radial_distribution(pos_frames: &[&[Vec2]], bx: &Box2, bins: usize) -> Vec<(f64, f64)> {
    let r_max = 0.5 * bx.lx.min(by(bx));
    let dr = r_max / bins as f64;
    let n = pos_frames.first().map(|f| f.len()).unwrap_or(0) as f64;
    let f_count = pos_frames.len() as f64;
    let rho = n * f_count / (f_count * bx.lx * bx.ly); // n / area
    let mut hist = vec![0u64; bins];
    for frame in pos_frames {
        for i in 0..frame.len() {
            for j in (i + 1)..frame.len() {
                let dx = minimum_image(frame[i][0] - frame[j][0], bx.lx);
                let dy = minimum_image(frame[i][1] - frame[j][1], bx.ly);
                let r = (dx * dx + dy * dy).sqrt();
                if r < r_max {
                    hist[(r / dr) as usize] += 1;
                }
            }
        }
    }
    (0..bins)
        .map(|b| {
            let r_in = b as f64 * dr;
            let r_out = r_in + dr;
            let area = std::f64::consts::PI * (r_out * r_out - r_in * r_in);
            let g = 2.0 * hist[b] as f64 / (n * f_count * rho * area);
            (r_in + 0.5 * dr, g)
        })
        .collect()
}

fn by(bx: &Box2) -> f64 { bx.ly }
```

Note `r_max = 0.5 * min(Lx, Ly)` [Course Requirement]; `rho = N/(Lx*Ly)`.

Add to `lib.rs`: `mod metrics; pub use metrics::{bin_edges, chi2_22, frame_total_energies, pooled_speeds, radial_distribution, secular_drift, t_speed};`

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml metrics`
Expected: 6 PASS. If `chi2_22_is_small_for_rayleigh_samples` marginally
exceeds 2 due to the deterministic inverse-CDF grid, keep the grid (it is
exact-equal-probability by construction, so chi2 is driven only by T_speed
fitting and stays ≪ 2); investigate before loosening anything.

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/metrics.rs week2/md/src/lib.rs
git commit -m "feat: add drift, T_speed, chi2_22, and g(r) metrics"
```

---

### Task 10: Structural validation, check report, `md check` CLI (checker.rs)

**Files:**
- Create: `week2/md/src/checker.rs`, `week2/md/tests/checker.rs`
- Modify: `week2/md/src/cli.rs`, `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `read_artifacts`, `RunConfig`, `Frame`, `Box2`,
  `frame_total_energies`, `secular_drift`, `pooled_speeds`, `t_speed`,
  `chi2_22`.
- Produces:
  - `pub struct CheckReport { pub drift: f64, pub drift_pass: bool,
    pub t_speed: f64, pub temperature_pass: bool, pub chi2_22: f64,
    pub chi2_pass: bool, pub max_energy_mismatch: f64 }`
  - `pub struct CheckError(pub String);`
  - `pub fn check_artifacts(dir: &Path) -> Result<CheckReport, CheckError>`
    (Err = malformed input; Ok(report) carries PASS/FAIL per bound)
  - `pub fn run_check(dir: &Path) -> i32` (prints the report; 0 iff all
    pass; used by the CLI and tests)

- [ ] **Step 1: Write the failing tests** `week2/md/tests/checker.rs`:

```rust
//! Checker tests: metrics recomputed from raw data, structural validation,
//! and negative controls [Suggestion].

use md::checker::{check_artifacts, run_check};
use md::io::{RunConfig, write_artifacts};
use md::simulate::{Frame, SimConfig};
use std::path::PathBuf;

fn temp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("md-check-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Two frames of synthetic data: two atoms far apart (E_pot = 0) with
/// Maxwell-ish velocities ignored here; used for structural tests.
fn synthetic_frames(vel: Vec<[f64; 2]>) -> Vec<Frame> {
    (1..=2)
        .map(|s| Frame {
            step: s * 50,
            t: s as f64 * 0.01,
            pos: vec![[0.5, 0.5], [5.0, 5.0], [9.0, 1.0], [1.0, 9.0]],
            vel: vel.clone(),
            e_pot: 0.0,
            e_kin: 0.5 * vel.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum::<f64>(),
        })
        .collect()
}

fn write_synthetic(tag: &str, frames: &[Frame], run: Option<RunConfig>) -> PathBuf {
    let dir = temp(tag);
    let run = run.unwrap_or_else(|| {
        let mut r = RunConfig::from(SimConfig { n: 4, rho: 0.8, temperature: 0.5,
            dt: 0.01, eq_steps: 0, steps: 100, sample_every: 50, seed: 2026 });
        r.box_dim = [10.0, 10.0]; // hand-built box for the synthetic data
        r
    });
    write_artifacts(&dir, &run, frames).unwrap();
    dir
}

#[test]
fn malformed_missing_files_fail() {
    assert!(check_artifacts(&temp("none")).is_err());
}

#[test]
fn wrong_integrator_fails() {
    let mut run = RunConfig::from(SimConfig { n: 4, rho: 0.8, temperature: 0.5,
        dt: 0.01, eq_steps: 0, steps: 100, sample_every: 50, seed: 2026 });
    run.integrator = "euler".into();
    let dir = write_synthetic("integrator", &synthetic_frames(vec![[0.1, 0.1]; 4]), Some(run));
    assert!(check_artifacts(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wrong_frame_count_or_steps_fail() {
    let mut run = RunConfig::from(SimConfig { n: 4, rho: 0.8, temperature: 0.5,
        dt: 0.01, eq_steps: 0, steps: 150, sample_every: 50, seed: 2026 });
    run.box_dim = [10.0, 10.0];
    // 2 frames but steps says 3 should be saved.
    let dir = write_synthetic("count", &synthetic_frames(vec![[0.1, 0.1]; 4]), Some(run));
    assert!(check_artifacts(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn out_of_box_position_fails() {
    let mut frames = synthetic_frames(vec![[0.1, 0.1]; 4]);
    frames[0].pos[0] = [10.5, 0.5]; // outside [0, 10)
    let dir = write_synthetic("oob", &frames, None);
    assert!(check_artifacts(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stored_energy_mismatch_fails() {
    let mut frames = synthetic_frames(vec![[0.1, 0.1]; 4]);
    frames[0].e_kin += 1.0; // stored value disagrees with raw velocities
    let dir = write_synthetic("energy", &frames, None);
    assert!(check_artifacts(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn run_check_exit_codes() {
    // Valid structure but wrong temperature (all velocities tiny): the
    // temperature bound must FAIL -> nonzero exit. Malformed -> nonzero.
    let dir = write_synthetic("exit", &synthetic_frames(vec![[0.01, 0.01]; 4]), None);
    assert_ne!(run_check(&dir), 0);
    assert_ne!(run_check(&temp("none2")), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn end_to_end_corrupted_trajectory_fails() {
    // [Suggestion] end-to-end negative control: take a real small run and
    // corrupt one raw position, then run the real check path.
    let dir = temp("e2e");
    let config = SimConfig { n: 16, rho: 0.8, temperature: 0.5, dt: 0.01,
                             eq_steps: 100, steps: 100, sample_every: 50, seed: 2026 };
    let frames = md::simulate::run_simulation(&config);
    write_artifacts(&dir, &RunConfig::from(&config), &frames).unwrap();
    // Corrupt: NaN a velocity in the first frame's raw data.
    let traj = std::fs::read_to_string(dir.join("traj.jsonl")).unwrap();
    let mut lines: Vec<String> = traj.lines().map(str::to_string).collect();
    let mut v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    v["vel"][0][0] = serde_json::json!("not-a-number");
    lines[0] = serde_json::to_string(&v).unwrap();
    std::fs::write(dir.join("traj.jsonl"), lines.join("\n") + "\n").unwrap();
    assert_ne!(run_check(&dir), 0, "corrupted run must FAIL check");
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: RED**

Run: `cargo test --manifest-path md/Cargo.toml --test checker`
Expected: compile FAIL (`md::checker` unresolved).

- [ ] **Step 3: Implement `checker.rs`**

```rust
//! Independent verification of saved artifacts.

use crate::io::{read_artifacts, RunConfig};
use crate::metrics::{chi2_22, frame_total_energies, pooled_speeds, secular_drift, t_speed};
use crate::system::Box2;
use std::path::Path;

/// Result of the physics checks (structure already validated).
#[derive(Clone, Copy, Debug)]
pub struct CheckReport {
    pub drift: f64,
    pub drift_pass: bool,
    pub t_speed: f64,
    pub temperature_pass: bool,
    pub chi2_22: f64,
    pub chi2_pass: bool,
    pub max_energy_mismatch: f64,
}

pub struct CheckError(pub String);

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Structural validation [Suggestion implementing the sheet's
/// malformed-file Course Requirement] + physics checks.
pub fn check_artifacts(dir: &Path) -> Result<CheckReport, CheckError> {
    let (run, frames): (RunConfig, Vec<crate::simulate::Frame>) =
        read_artifacts(dir).map_err(CheckError)?;

    // ---- structural ----
    if run.integrator != "velocity-verlet" {
        return Err(CheckError(format!("integrator must be velocity-verlet, got {}", run.integrator)));
    }
    if run.sample_every == 0 || run.steps % run.sample_every != 0 {
        return Err(CheckError("steps must be a multiple of sample_every".into()));
    }
    let expected_frames = run.steps / run.sample_every;
    if frames.len() != expected_frames {
        return Err(CheckError(format!(
            "expected {expected_frames} frames, found {}", frames.len()
        )));
    }
    let bx = Box2 { lx: run.box_dim[0], ly: run.box_dim[1] };
    let bx_rebuilt = Box2::new(run.n, run.rho);
    // Box cross-check [Suggestion].
    if (bx.lx - bx_rebuilt.lx).abs() > 1e-10 * f64::max(1.0, bx.lx)
        || (bx.ly - bx_rebuilt.ly).abs() > 1e-10 * f64::max(1.0, bx.ly)
    {
        return Err(CheckError("stored box disagrees with n and rho".into()));
    }
    for (idx, frame) in frames.iter().enumerate() {
        let expected_step = (idx + 1) * run.sample_every;
        if frame.step != expected_step {
            return Err(CheckError(format!(
                "frame {idx}: step {} != {}", frame.step, expected_step
            )));
        }
        if (frame.t - frame.step as f64 * run.dt).abs() > 1e-10 * f64::max(1.0, frame.t) {
            return Err(CheckError(format!("frame {idx}: t inconsistent with step*dt")));
        }
        if frame.pos.len() != run.n || frame.vel.len() != run.n {
            return Err(CheckError(format!("frame {idx}: wrong array length")));
        }
        for p in &frame.pos {
            if !p[0].is_finite() || !p[1].is_finite()
                || !(0.0..bx.lx).contains(&p[0]) || !(0.0..bx.ly).contains(&p[1])
            {
                return Err(CheckError(format!("frame {idx}: position outside box or non-finite")));
            }
        }
        for v in &frame.vel {
            if !v[0].is_finite() || !v[1].is_finite() {
                return Err(CheckError(format!("frame {idx}: non-finite velocity")));
            }
        }
        if !frame.e_pot.is_finite() || !frame.e_kin.is_finite() {
            return Err(CheckError(format!("frame {idx}: non-finite energy")));
        }
    }

    // ---- physics, recomputed from raw data ----
    let totals = frame_total_energies(&frames, &bx);
    let drift = secular_drift(&totals);
    let speeds = pooled_speeds(&frames);
    let ts = t_speed(&speeds);
    let chi = chi2_22(&speeds, ts);

    // Stored-energy cross-check [Suggestion]; never used above.
    let mut max_mismatch = 0.0;
    for (frame, total) in frames.iter().zip(totals.iter()) {
        let recomputed_pot = total - frame.e_kin_raw_independent();
        // (see implementation note below — compute recomputed E_pot and
        //  E_kin separately inside frame_total_energies' helper)
        let _ = recomputed_pot;
        let ek_recomputed: f64 = frame.vel.iter()
            .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum();
        let mismatch_pot = 0.0; // filled in from the split helper
        let _ = mismatch_pot;
        max_mismatch = max_mismatch.max(
            (frame.e_kin - ek_recomputed).abs() / f64::max(1.0, ek_recomputed.abs()),
        );
    }
    if max_mismatch > 1e-10 {
        return Err(CheckError(format!(
            "stored energies disagree with recomputed values (max {max_mismatch})"
        )));
    }

    Ok(CheckReport {
        drift,
        drift_pass: drift < 2e-3,
        t_speed: ts,
        temperature_pass: (ts - run.temperature).abs() < 0.05,
        chi2_22: chi,
        chi2_pass: chi < 2.0,
        max_energy_mismatch: max_mismatch,
    })
}
```

Implementation note: implement the cross-check concretely by splitting the
recompute helper into `recompute_energies(frame, bx) -> (e_pot, e_kin)`
(use it in `frame_total_energies` too, keeping that public signature) and
cross-checking both `E_pot` and `E_kin` with
`abs(stored - recomputed) <= 1e-10 * max(1, abs(recomputed))`. Remove the
placeholder lines shown above — the real code has no `let _ =` stubs.

`run_check`:

```rust
/// Print the report and return the process exit code (0 iff all pass).
pub fn run_check(dir: &Path) -> i32 {
    match check_artifacts(dir) {
        Err(e) => { eprintln!("check FAILED: {e}"); 1 }
        Ok(r) => {
            println!("frames cross-check: max energy mismatch = {:.3e}", r.max_energy_mismatch);
            println!("secular drift   = {:.6e}  (bound 2e-3)  {}", r.drift,
                     if r.drift_pass { "PASS" } else { "FAIL" });
            println!("T_speed         = {:.6}    (target {:.2}, bound 0.05)  {}", r.t_speed, "?", "");
            // (print run temperature via the stored RunConfig as needed)
            println!("chi2_22         = {:.4}    (bound 2)  {}", r.chi2_22,
                     if r.chi2_pass { "PASS" } else { "FAIL" });
            if r.drift_pass && r.temperature_pass && r.chi2_pass { 0 } else { 1 }
        }
    }
}
```

(Concrete version: carry `run.temperature` into the report or re-read it in
`run_check`; no `"?"` placeholders in the final code.)

Wire the CLI `Check` arm:

```rust
Command::Check { artifacts } => md::checker::run_check(&artifacts),
```

Add to `lib.rs`: `mod checker; pub use checker::{CheckError, CheckReport, check_artifacts, run_check};`

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test checker`
Expected: 7 PASS (malformed, wrong integrator, wrong frame count,
out-of-box, energy mismatch, exit codes, end-to-end corruption).

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/checker.rs week2/md/src/cli.rs week2/md/src/lib.rs \
        week2/md/tests/checker.rs
git commit -m "feat: add md check with independent recomputation and negative controls"
```

---

### Task 11: The full default contract test goes green

**Files:**
- Modify: none (the test exists since Task 1).

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Run the contract test (it should now pass)**

Run: `cargo test --manifest-path md/Cargo.toml --test contract`
Expected: PASS — the default run produces 200 frames, steps 50..=10000,
and `check_artifacts` returns all three bounds passing.

There is exactly one set of physics acceptance criteria, identical in
debug and release: secular drift < 2e-3; |T_speed - 0.5| < 0.05 for the
default contract; chi2_22 < 2. The bounds are never loosened, changed,
or re-derived per build profile.

If a bound fails, do NOT loosen the bound. Investigate in this order:
(a) drift — verify the thermostat is off in production and that
`fluid_potential_energy` uses minimum image; (b) T_speed — verify the last
equilibration rescale happens at step `eq_steps`; (c) chi2 — verify bin
edges use T_speed, not 0.5.

- [ ] **Step 2: Run the whole suite**

Run: `cargo test --manifest-path md/Cargo.toml`
Expected: everything PASS (Part 2/3 regression, all Part 4 unit and
integration tests, including the already-green `tests/cli.rs` and
`tests/checker.rs`). (A debug-build contract run may take tens of
seconds; that is acceptable mid-development and does not alter the
acceptance criteria.)

- [ ] **Step 3: Commit (if any fix was needed)**

```bash
git add -A week2/md
git commit -m "fix: make the default contract run pass all physics bounds"
```

(If no fix was needed, no commit — Task 6–10 commits already cover it.)

---

### Task 12: Renderer and ffmpeg pipe (render.rs, video.rs)

**Files:**
- Create: `week2/md/src/render.rs`, `week2/md/src/video.rs`
- Modify: `week2/md/src/lib.rs`

**Interfaces:**
- Consumes: `Frame`, `RunConfig`, `radical_distribution`, `Box2`.
- Produces:
  - `pub struct Canvas { pub width: usize, pub height: usize, pub rgba: Vec<u8> }`
  - `impl Canvas { pub fn new(w, h) -> Canvas; pub fn fill(...);
    pub fn filled_disc(...); pub fn line(...); pub fn into_pixels(self) -> Vec<u8> }`
  - `pub fn render_frame(frame: &Frame, bx: &Box2, g: &[(f64, f64)], canvas_size: (usize, usize)) -> Vec<u8>`
    (left panel particles, right panel static g(r) with g = 1 reference)
  - `pub fn encode_video(dir: &Path, out_mp4: &Path) -> Result<(), String>`
    (computes g(r) once over all frames, renders every frame, pipes RGBA
    to ffmpeg)
  - `pub fn ffmpeg_available() -> bool`

- [ ] **Step 1: Write the failing unit tests** (in `render.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_starts_opaque_black_and_supports_disc() {
        let mut c = Canvas::new(16, 16);
        assert_eq!(c.rgba.len(), 16 * 16 * 4);
        assert!(c.rgba.chunks(4).all(|px| px[3] == 255));
        c.filled_disc(8, 8, 3, [255, 0, 0, 255]);
        assert_eq!(c.rgba[(8 * 16 + 8) * 4], 255);
        // A pixel outside the disc radius is untouched.
        assert_eq!(c.rgba[(0 * 16 + 0) * 4], 0);
    }

    #[test]
    fn line_draws_straight_horizontal() {
        let mut c = Canvas::new(16, 4);
        c.line(2, 2, 13, 2, [255; 4]);
        assert_eq!(c.rgba[(2 * 16 + 2) * 4], 255);
        assert_eq!(c.rgba[(2 * 16 + 13) * 4], 255);
    }

    #[test]
    fn render_frame_produces_full_size_rgba() {
        let frame = crate::simulate::Frame {
            step: 50, t: 0.5,
            pos: vec![[0.0, 0.0], [2.0, 2.0], [4.0, 1.0], [1.0, 4.0]],
            vel: vec![[0.0; 2]; 4], e_pot: 0.0, e_kin: 0.0,
        };
        let bx = crate::system::Box2 { lx: 5.0, ly: 5.0 };
        let g = vec![(0.5, 1.0); 10];
        let pixels = render_frame(&frame, &bx, &g, (320, 160));
        assert_eq!(pixels.len(), 320 * 160 * 4);
    }
}
```

(video tests, in `video.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_available_detects_binary() {
        // Either outcome is fine; it must not panic. On this machine ffmpeg
        // is expected absent, so video rendering tests elsewhere must skip.
        let _ = ffmpeg_available();
    }

    #[test]
    fn encode_video_of_tiny_run_produces_small_mp4_when_ffmpeg_present() {
        // Runtime availability handling: if ffmpeg is unavailable the
        // external-video portion is skipped gracefully; if available the
        // MP4 is actually generated and size-checked. The < 2 MB bound
        // itself is unconditional — only the external-binary step is
        // conditional.
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }
        let dir = std::env::temp_dir().join(format!("md-video-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = crate::simulate::SimConfig {
            n: 16, rho: 0.8, temperature: 0.5, dt: 0.01,
            eq_steps: 50, steps: 100, sample_every: 50, seed: 2026,
        };
        let frames = crate::simulate::run_simulation(&config);
        crate::io::write_artifacts(&dir, &crate::io::RunConfig::from(&config), &frames).unwrap();
        let mp4 = dir.join("run.mp4");
        encode_video(&dir, &mp4).expect("encode");
        let size = std::fs::metadata(&mp4).unwrap().len();
        assert!(size < 2_000_000, "mp4 is {size} bytes, must be < 2 MB");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
```

Design note: the mp4-size test is a normal (non-`#[ignore]`) test with
runtime availability handling — when ffmpeg is absent it skips the
external-video portion gracefully; when ffmpeg is installed (the manual
verification environment) it actually runs the MP4 generation and the
< 2 MB size check. The course does not require ffmpeg for the release
test suite (`make reproduce` excludes video).

- [ ] **Step 2: RED**

Run: `cargo test --manifest-path md/Cargo.toml render video`
Expected: compile FAIL.

- [ ] **Step 3: Implement `render.rs` and `video.rs`**

`render.rs` — a minimal RGBA rasterizer (no image crate):

```rust
//! Minimal RGBA rasterizer for the video panels.

pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Canvas {
        Canvas { width, height, rgba: vec![0u8; width * height * 4] }
    }
    pub fn fill(&mut self, [r, g, b, a]: [u8; 4]) {
        for px in self.rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&[r, g, b, a]);
        }
    }
    pub fn set(&mut self, x: usize, y: usize, color: [u8; 4]) {
        if x < self.width && y < self.height {
            let i = (y * self.width + x) * 4;
            self.rgba[i..i + 4].copy_from_slice(&color);
        }
    }
    /// Filled disc via integer radius scan (small radii only).
    pub fn filled_disc(&mut self, cx: usize, cy: usize, radius: usize, color: [u8; 4]) {
        let r = radius as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let x = cx as isize + dx;
                    let y = cy as isize + dy;
                    if x >= 0 && y >= 0 {
                        self.set(x as usize, y as usize, color);
                    }
                }
            }
        }
    }
    /// Bresenham-ish line by linear interpolation.
    pub fn line(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, color: [u8; 4]) {
        let steps = usize::max(x1.abs_diff(x0), y1.abs_diff(y0)).max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let x = (x0 as f64 + t * (x1 as f64 - x0 as f64)) as usize;
            let y = (y0 as f64 + t * (y1 as f64 - y0 as f64)) as usize;
            self.set(x, y, color);
        }
    }
    pub fn into_pixels(self) -> Vec<u8> { self.rgba }
}
```

`render_frame(frame, bx, g, (w, h))`: create a `Canvas`, background fill,
left half = box panel (map `p -> pixel`, disc per atom, radius ~3 px,
discs drawn in a fixed color; periodic box border as a rectangle of
`line` calls), right half = g(r) panel (map the 50 `(r, g)` points into
the panel, draw a polyline plus a horizontal g = 1 reference line, simple
linear y-scale with `g_max = max(2.0, data max)`), return `into_pixels()`.

`video.rs`:

```rust
//! ffmpeg pipe: render every saved frame and encode to MP4.

use crate::io::read_artifacts;
use crate::metrics::radial_distribution;
use crate::render::render_frame;
use crate::system::Box2;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// True iff an `ffmpeg` executable is on PATH.
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

/// Render the whole trajectory (with whole-run g(r) as a static panel)
/// and encode via ffmpeg. One video frame per saved frame [Course Req].
pub fn encode_video(dir: &Path, out_mp4: &Path) -> Result<(), String> {
    if !ffmpeg_available() {
        return Err("ffmpeg executable not found; install ffmpeg to use md video".into());
    }
    let (run, frames) = read_artifacts(dir).map_err(|e| format!("cannot read artifacts: {e}"))?;
    let bx = Box2 { lx: run.box_dim[0], ly: run.box_dim[1] };
    let pos_refs: Vec<&[crate::Vec2]> = frames.iter().map(|f| f.pos.as_slice()).collect();
    let g = radial_distribution(&pos_refs, &bx, 50);

    let (w, h) = (640usize, 320usize); // [Suggestion] layout/size
    let fps = "10";                    // [Suggestion]
    let mut child = Command::new("ffmpeg")
        .args(["-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", &format!("{w}x{h}"), "-r", fps,
               "-i", "pipe:0",
               "-c:v", "libx264", "-preset", "fast", "-crf", "28",
               "-pix_fmt", "yuv420p", "-y"])
        .arg(out_mp4)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot spawn ffmpeg: {e}"))?;
    let stdin = child.stdin.as_mut().expect("piped stdin");
    for frame in &frames {
        let pixels = render_frame(frame, &bx, &g, (w, h));
        stdin.write_all(&pixels).map_err(|e| format!("ffmpeg pipe: {e}"))?;
    }
    drop(stdin);
    let status = child.wait().map_err(|e| format!("ffmpeg wait: {e}"))?;
    if status.success() { Ok(()) } else { Err("ffmpeg exited with an error".into()) }
}
```

Add to `lib.rs`: `mod render; pub use render::{Canvas, render_frame}; mod video; pub use video::{encode_video, ffmpeg_available};`

- [ ] **Step 4: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml render video`
Expected: render tests PASS; `ffmpeg_available_detects_binary` PASS
(either branch); the mp4 test skips gracefully when ffmpeg is absent.

- [ ] **Step 5: Commit**

```bash
git add week2/md/src/render.rs week2/md/src/video.rs week2/md/src/lib.rs
git commit -m "feat: add RGBA renderer and ffmpeg video pipe"
```

---

### Task 13: `md video` CLI wiring

**Files:**
- Modify: `week2/md/src/cli.rs`

**Interfaces:**
- Consumes: `encode_video` (Task 12).
- Produces: working `md video <ARTIFACTS> --out <PATH>`.

- [ ] **Step 1: RED** — there is no dedicated acceptance test beyond the
mp4-size test with runtime availability handling (ffmpeg is an external
prerequisite). Behavior to
verify by hand in Step 3; the missing-ffmpeg error path is testable:

Add to `tests/cli.rs`:

```rust
#[test]
fn video_reports_missing_ffmpeg_or_succeeds() {
    // Course Requirement: one binary, md video artifacts --out PATH.
    // If ffmpeg is absent, exit nonzero with a clear message [design].
    let out = temp_dir("video");
    let artifacts = temp_dir("video-src");
    let run = run_md(&["run", "--n", "16", "--eq-steps", "50", "--steps", "100",
                       "--sample-every", "50", "--out", artifacts.to_str().unwrap()]);
    assert!(run.status.success());
    let video = run_md(&["video", artifacts.to_str().unwrap(),
                         "--out", out.join("run.mp4").to_str().unwrap()]);
    if md::video::ffmpeg_available() {
        assert!(video.status.success());
        assert!(out.join("run.mp4").exists());
    } else {
        assert!(!video.status.success());
    }
    // --out is required: omitting it must fail to parse.
    let no_out = run_md(&["video", artifacts.to_str().unwrap()]);
    assert!(!no_out.status.success());
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&artifacts);
}
```

Run: `cargo test --manifest-path md/Cargo.toml --test cli`
Expected: the new test FAILS at the `todo!("md video")` panic.

- [ ] **Step 2: Implement the `Video` arm**

```rust
Command::Video { artifacts, out } => match md::video::encode_video(&artifacts, &out) {
    Ok(()) => 0,
    Err(msg) => { eprintln!("error: {msg}"); 3 }
},
```

- [ ] **Step 3: GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test cli`
Expected: 3 PASS (ffmpeg absent on this machine → error-path branch).

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/cli.rs week2/md/tests/cli.rs
git commit -m "feat: wire md video subcommand with explicit --out"
```

---

### Task 14: `week2/Makefile`, full VERIFY, and REPRODUCE

**Files:**
- Create: `week2/Makefile`

**Interfaces:**
- Consumes: the `md` binary built from the crate.

- [ ] **Step 1: Write the Makefile [Course Requirement: week2/Makefile]**

```make
# Week 2 Part 4: regenerate the default contract run.
# Test verification:    cargo test --manifest-path md/Cargo.toml --release
# Physics verification: cargo run --manifest-path md/Cargo.toml --release -- check artifacts
# Video (needs ffmpeg): cargo run --manifest-path md/Cargo.toml --release -- video artifacts --out artifacts/run.mp4

reproduce:
	cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts

.PHONY: reproduce
```

- [ ] **Step 2: VERIFY — the full mandated release suite**

Run (from `week2/`):

```bash
cargo test --manifest-path md/Cargo.toml --release
```

Expected: every test PASS, including:
- Part 2/3 regression: `tests/lj.rs`, `tests/dimer.rs`, module unit tests;
- `tests/pair.rs` (shifted cutoff: continuity at rc, force identity,
  non-straddling finite differences);
- `tests/fluid.rs` (total internal force vanishes within 1e-10, MIC, wrap);
- `tests/cli.rs` (binary writes required files/frames; validation);
- `tests/contract.rs` (default contract run: 200 frames, three bounds);
- `tests/checker.rs` (negative controls + end-to-end corruption).

- [ ] **Step 3: REPRODUCE — fresh-clone contract run**

Run (from `week2/`):

```bash
make reproduce
cargo run --manifest-path md/Cargo.toml --release -- check artifacts
```

Expected: `make reproduce` writes `week2/artifacts/run.json` and
`week2/artifacts/traj.jsonl` (200 frames); `md check artifacts` prints
PASS on all three bounds and exits 0. Confirm `artifacts/` is not tracked:
`git status --short` shows nothing for `week2/artifacts`.

- [ ] **Step 4: Commit**

```bash
git add week2/Makefile
git commit -m "feat: add week2 Makefile reproduce target for the contract run"
```

---

## Self-Review

**Spec coverage check** (spec section → tasks):
- Physical model (lattice, PBC, MIC, wrap) → Task 2; interaction (shifted
  cutoff, force jump) → Task 3; periodic fluid forces/energies → Task 4.
- Velocity init, COM removal, Schedule B thermostat, T_target from config →
  Task 5 (math) + Task 6 (schedule; `thermostat_events` test pins 41).
- Production timing (no step 0, 200 frames, t = step·dt) → Task 6 + contract
  test (Task 1/11).
- CLI forms and defaults, non-square-N and divisibility rejection → Tasks 1,
  8, 13.
- run.json/traj.jsonl contract → Task 7 (+ Task 1 acceptance).
- Independent recomputation, structural validation, F/S notation, three
  bounds, energy/box cross-checks → Tasks 9, 10.
- g(r) definition, r_max, normalization → Task 9; whole-trajectory static
  panel + one frame per saved frame + < 2 MB → Tasks 12, 13.
- week2/Makefile scope (run only) → Task 14.
- Part 2/3 preservation → Global Constraints + regression runs in Tasks 4,
  14.
- No cell lists, no Python, ffmpeg external → Global Constraints.
- Course-required tests present in the first RED stage → Task 1 Steps 2–4
  (tests written, RED observed, failing tests committed; CLI skeleton only
  afterwards in Task 1b).
- ffmpeg/video test uses runtime availability handling, not `#[ignore]`
  (Task 12).
- One set of physics acceptance criteria, identical in debug and release
  (Task 11).

**Placeholder scan:** the two intentional "expanded pseudocode" notes in
Tasks 6 and 10 explicitly instruct replacing condensed snippets with the
concrete loop / split helper; no TBD/TODO remains elsewhere.

**Type consistency:** `Box2 { lx, ly }`, `Frame { step, t, pos, vel, e_pot,
e_kin }`, `SimConfig` fields, `RunConfig::from(&SimConfig)`,
`check_artifacts(&Path) -> Result<CheckReport, CheckError>`,
`radial_distribution(&[&[Vec2]], &Box2, usize)`, and CLI flag names are
identical across tasks.
