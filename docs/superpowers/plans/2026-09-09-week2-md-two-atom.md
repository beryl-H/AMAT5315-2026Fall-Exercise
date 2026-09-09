# Week 2 Two-Atom MD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved two-atom molecular-dynamics experiment in
`week2/md` so that one shared driver runs the same dimer with either Forward
Euler or velocity-Verlet through a single `Integrator` trait.

**Architecture:** `lib.rs` exposes a small `State` type, a shared
`Integrator` trait with `Euler` and caching `VelocityVerlet` implementations,
and an `experiment` module containing Lennard-Jones accelerations, energy
measurement, and the generic `run_experiment` driver. Integration tests in
`tests/dimer.rs` assert the course acceptance criteria through public APIs
only.

**Tech Stack:** Rust 2021 (existing `md` crate at `week2/md/Cargo.toml`), no
new dependencies.

**Spec:** `../specs/2026-09-09-week2-md-two-atom-design.md`

## Global Constraints

- Work only under `week2/md/`; do not touch other weeks.
- Keep `tests/lj.rs`, `src/main.rs`, and `Cargo.toml` unchanged.
- Keep `lj_energy` and `lj_force` unchanged in `src/lib.rs`.
- Run all cargo commands from the repository root with
  `--manifest-path week2/md/Cargo.toml`.
- Forward Euler, `dt = 0.01`, 500 steps: final relative energy error `> 0.5`.
- Velocity-Verlet, `dt = 0.01`, 500 steps: maximum absolute relative energy
  error `< 1e-3`.
- Velocity-Verlet also runs 5000 steps for later verification/plotting.
- Forward Euler uses old positions, old velocities, and old accelerations:
  `x_{n+1} = x_n + v_n * dt`, `v_{n+1} = v_n + a_n * dt`.
- Velocity-Verlet order is exact: `v_half = v_n + 0.5 * dt * a_n`,
  `x_{n+1} = x_n + dt * v_half`, `a_{n+1} = acceleration(x_{n+1})`,
  `v_{n+1} = v_half + 0.5 * dt * a_{n+1}`; `a_{n+1}` is retained.
- The dimer acceptance test is created before any simulator implementation.
- Workflow: design → plan → TEST → RED → IMPLEMENT → GREEN. No sabotage
  commit.

## File Structure

- `week2/md/tests/dimer.rs` — create; integration tests derived from the
  course criteria and the shared-driver requirement.
- `week2/md/src/state.rs` — create; `Vec2`, `State`, and
  `two_atom_initial_state`.
- `week2/md/src/integrator.rs` — create; `Integrator`, `Euler`,
  `VelocityVerlet`.
- `week2/md/src/experiment.rs` — create; `lj_accelerations`, energy
  functions, `ExperimentResult`, `run_experiment`.
- `week2/md/src/lib.rs` — modify only to declare modules and re-export the
  public API.

---

### Task 1: Dimer acceptance tests (TEST / RED)

**Files:**
- Create: `week2/md/tests/dimer.rs`

**Interfaces:**
- Consumes: nothing yet; this task intentionally references the public API
  that later tasks provide.
- Produces: `tests/dimer.rs`, the spec-derived acceptance suite that later
  tasks must make green.

- [ ] **Step 1: Write the failing dimer acceptance tests**

Create `week2/md/tests/dimer.rs` with exactly this content:

```rust
//! Integration tests for the Week 2 two-atom molecular-dynamics experiment.
//!
//! All assertions are derived from the course specification, not from the
//! implementation.

use std::cell::Cell;

use md::{
    Euler, Integrator, State, VelocityVerlet, lj_accelerations, run_experiment,
    two_atom_initial_state,
};

const DT: f64 = 0.01;
const STEPS_500: usize = 500;
const STEPS_5000: usize = 5000;

fn max_abs(errors: &[f64]) -> f64 {
    errors.iter().fold(0.0, |max, error| max.max(error.abs()))
}

#[test]
fn forward_euler_final_relative_energy_error_exceeds_half() {
    let result = run_experiment(&mut Euler::default(), STEPS_500, DT);
    let final_error = *result.relative_energy_errors.last().unwrap();
    assert!(
        final_error > 0.5,
        "Euler final relative energy error was {final_error}, expected > 0.5"
    );
}

#[test]
fn velocity_verlet_max_absolute_relative_energy_error_is_below_one_thousandth() {
    let result = run_experiment(&mut VelocityVerlet::default(), STEPS_500, DT);
    let max_error = max_abs(&result.relative_energy_errors);
    assert!(
        max_error < 1e-3,
        "velocity-Verlet max |relative energy error| was {max_error}, expected < 1e-3"
    );
}

#[test]
fn velocity_verlet_5000_step_run_is_available_and_finite() {
    let result = run_experiment(&mut VelocityVerlet::default(), STEPS_5000, DT);
    assert_eq!(result.relative_energy_errors.len(), STEPS_5000 + 1);
    assert_eq!(*result.steps.last().unwrap(), STEPS_5000);
    for error in &result.relative_energy_errors {
        assert!(error.is_finite(), "non-finite error in long velocity-Verlet run");
    }
}

#[test]
fn shared_driver_handles_both_integrators() {
    let euler = run_experiment(&mut Euler::default(), STEPS_500, DT);
    let verlet = run_experiment(&mut VelocityVerlet::default(), STEPS_500, DT);
    assert_eq!(euler.steps, verlet.steps);
    assert_eq!(euler.times, verlet.times);
    assert_eq!(euler.relative_energy_errors[0], 0.0);
    assert_eq!(verlet.relative_energy_errors[0], 0.0);
}

#[test]
fn velocity_verlet_evaluates_acceleration_once_per_step_after_initialization() {
    let mut state = two_atom_initial_state();
    let mut verlet = VelocityVerlet::default();
    let calls = Cell::new(0usize);
    let counting_acceleration = |s: &State| {
        calls.set(calls.get() + 1);
        lj_accelerations(s)
    };

    verlet.initialize(&state, &counting_acceleration);
    assert_eq!(calls.get(), 1, "initial acceleration must be evaluated once");

    verlet.step(&mut state, DT, &counting_acceleration);
    assert_eq!(calls.get(), 2, "each step must evaluate acceleration once");

    verlet.step(&mut state, DT, &counting_acceleration);
    assert_eq!(calls.get(), 3, "each step must evaluate acceleration once");
}
```

- [ ] **Step 2: Run the test suite to verify RED**

Run:

```bash
cargo test --manifest-path week2/md/Cargo.toml
```

Expected: compilation of `week2/md/tests/dimer.rs` fails because the public
API (`md::State`, `md::run_experiment`, etc.) does not exist yet. This is the
required RED state.

- [ ] **Step 3: Commit the failing test only**

```bash
git add week2/md/tests/dimer.rs
git commit -m "test: add failing Week 2 two-atom MD acceptance tests"
```

---

### Task 2: State module and initial configuration

**Files:**
- Create: `week2/md/src/state.rs`
- Modify: `week2/md/src/lib.rs` (module declaration and re-export only)
- Test: `week2/md/src/state.rs` (module unit test)

**Interfaces:**
- Consumes: nothing.
- Produces: `md::Vec2` (`[f64; 2]`), `md::State` with public `positions`
  and `velocities` fields, and `md::two_atom_initial_state() -> State`.

- [ ] **Step 1: Write the state implementation with its unit test**

Create `week2/md/src/state.rs` with exactly this content:

```rust
//! Minimal dynamical state for point atoms moving in 2D.

/// A 2D vector stored as `[x, y]`.
pub type Vec2 = [f64; 2];

/// Complete dynamical state of the simulated atoms.
///
/// Mass is implicitly `1` throughout this crate, matching the Week 2 spec.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
}

/// The fixed initial state from the Week 2 spec:
/// atoms at `(0.0, 0.0)` and `(1.2, 0.0)`, both at rest.
pub fn two_atom_initial_state() -> State {
    State {
        positions: vec![[0.0, 0.0], [1.2, 0.0]],
        velocities: vec![[0.0, 0.0], [0.0, 0.0]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_matches_spec() {
        let state = two_atom_initial_state();
        assert_eq!(state.positions, vec![[0.0, 0.0], [1.2, 0.0]]);
        assert_eq!(state.velocities, vec![[0.0, 0.0], [0.0, 0.0]]);
    }
}
```

- [ ] **Step 2: Wire the module into the library**

In `week2/md/src/lib.rs`, immediately after the `lj_force` function and before
the `#[cfg(test)]` module, add:

```rust
mod state;
pub use state::{State, Vec2, two_atom_initial_state};
```

Leave every existing Lennard-Jones function and the existing test module
unchanged.

- [ ] **Step 3: Run the library unit tests**

Run:

```bash
cargo test --lib --manifest-path week2/md/Cargo.toml
```

Expected: PASS, including the existing `greeting_says_hello_world` unit test
and the new `initial_state_matches_spec`.

The full `cargo test` command is still RED because `tests/dimer.rs` needs the
remaining API; that is expected until Task 5.

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/state.rs week2/md/src/lib.rs
git commit -m "feat: add MD state and two-atom initial configuration"
```

---

### Task 3: Lennard-Jones accelerations and energy measurement

**Files:**
- Create: `week2/md/src/experiment.rs` (acceleration and energy part only;
  the driver is added in Task 5)
- Modify: `week2/md/src/lib.rs` (module declaration and re-export only)
- Test: `week2/md/src/experiment.rs` (module unit tests)

**Interfaces:**
- Consumes: `State`, `Vec2`, `lj_energy`, `lj_force`.
- Produces:
  - `md::lj_accelerations(&State) -> Vec<Vec2>`
  - `md::kinetic_energy(&State) -> f64`
  - `md::potential_energy(&State) -> f64`
  - `md::total_energy(&State) -> f64`
  - `md::relative_energy_error(&State, f64) -> f64`

- [ ] **Step 1: Write the acceleration and energy implementation with unit tests**

Create `week2/md/src/experiment.rs` with exactly this content:

```rust
//! Lennard-Jones accelerations, energy measurement, and (in Task 5) the
//! shared experiment driver.

use crate::state::{State, Vec2};
use crate::{lj_energy, lj_force};

/// Accelerations from the plain Lennard-Jones potential.
///
/// Mass is 1, so acceleration equals force. For each pair `i < j`:
///
/// ```text
/// d = r_i - r_j
/// acceleration_i += lj_force(|d|) * d / |d|
/// acceleration_j -= lj_force(|d|) * d / |d|
/// ```
pub fn lj_accelerations(state: &State) -> Vec<Vec2> {
    let n = state.positions.len();
    let mut accelerations = vec![[0.0, 0.0]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = state.positions[i][0] - state.positions[j][0];
            let dy = state.positions[i][1] - state.positions[j][1];
            let r = (dx * dx + dy * dy).sqrt();
            let f_over_r = lj_force(r) / r;

            accelerations[i][0] += f_over_r * dx;
            accelerations[i][1] += f_over_r * dy;
            accelerations[j][0] -= f_over_r * dx;
            accelerations[j][1] -= f_over_r * dy;
        }
    }

    accelerations
}

/// Kinetic energy with mass 1: `0.5 * sum_i |v_i|^2`.
pub fn kinetic_energy(state: &State) -> f64 {
    state
        .velocities
        .iter()
        .map(|velocity| 0.5 * (velocity[0] * velocity[0] + velocity[1] * velocity[1]))
        .sum()
}

/// Potential energy: `sum_{i<j} U(r_ij)` with the existing `lj_energy`.
pub fn potential_energy(state: &State) -> f64 {
    let mut energy = 0.0;
    for i in 0..state.positions.len() {
        for j in (i + 1)..state.positions.len() {
            let dx = state.positions[i][0] - state.positions[j][0];
            let dy = state.positions[i][1] - state.positions[j][1];
            energy += lj_energy((dx * dx + dy * dy).sqrt());
        }
    }
    energy
}

/// Total mechanical energy measured with the Lennard-Jones potential.
pub fn total_energy(state: &State) -> f64 {
    kinetic_energy(state) + potential_energy(state)
}

/// Relative energy error against a fixed reference energy `e0`.
pub fn relative_energy_error(state: &State, e0: f64) -> f64 {
    (total_energy(state) - e0) / e0.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::two_atom_initial_state;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn initial_total_energy_is_lj_energy_at_spec_separation() {
        let state = two_atom_initial_state();
        let expected = 4.0 * (1.2f64.powi(-12) - 1.2f64.powi(-6));
        assert!((total_energy(&state) - expected).abs() < EPSILON);
        assert_eq!(relative_energy_error(&state, total_energy(&state)), 0.0);
    }

    #[test]
    fn pair_acceleration_follows_spec_sign_convention() {
        let state = State {
            positions: vec![[0.0, 0.0], [1.2, 0.0]],
            velocities: vec![[0.0, 0.0], [0.0, 0.0]],
        };
        let accelerations = lj_accelerations(&state);
        let scalar_force = crate::lj_force(1.2);
        // d / r for atom 0 relative to atom 1 is (-1, 0).
        let expected_first_x = scalar_force * (-1.0);

        assert!((accelerations[0][0] - expected_first_x).abs() < EPSILON);
        assert_eq!(accelerations[0][1], 0.0);
        assert!((accelerations[1][0] + accelerations[0][0]).abs() < EPSILON);
        assert_eq!(accelerations[1][1], 0.0);
    }
}
```

- [ ] **Step 2: Wire the module into the library**

In `week2/md/src/lib.rs`, after the `state` module declarations added in Task
2, add:

```rust
mod experiment;
pub use experiment::{
    kinetic_energy, lj_accelerations, potential_energy, relative_energy_error, total_energy,
};
```

`run_experiment` and `ExperimentResult` are intentionally not exported yet;
they are added in Task 5.

- [ ] **Step 3: Run the library unit tests**

Run:

```bash
cargo test --lib --manifest-path week2/md/Cargo.toml
```

Expected: PASS for the existing tests plus the two new acceleration/energy
tests.

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/experiment.rs week2/md/src/lib.rs
git commit -m "feat: add LJ accelerations and energy measurement"
```

---

### Task 4: Integrator trait, Forward Euler, and velocity-Verlet

**Files:**
- Create: `week2/md/src/integrator.rs`
- Modify: `week2/md/src/lib.rs` (module declaration and re-export only)
- Test: `week2/md/src/integrator.rs` (module unit tests)

**Interfaces:**
- Consumes: `State`, `Vec2`.
- Produces:
  - `md::Integrator` with `initialize` and `step`
  - `md::Euler`
  - `md::VelocityVerlet`

- [ ] **Step 1: Write the integrators with their unit tests**

Create `week2/md/src/integrator.rs` with exactly this content:

```rust
//! Shared integration interface and the two Week 2 integrators.

use crate::state::{State, Vec2};

/// Common interface for advancing a molecular-dynamics state by one step.
pub trait Integrator {
    /// Optional one-time setup before the first step.
    ///
    /// The default does nothing. Velocity-Verlet overrides this to cache the
    /// initial acceleration.
    fn initialize(&mut self, _state: &State, _acceleration: &dyn Fn(&State) -> Vec<Vec2>) {}

    /// Advance `state` by exactly one step of size `dt`.
    fn step(&mut self, state: &mut State, dt: f64, acceleration: &dyn Fn(&State) -> Vec<Vec2>);
}

/// Forward Euler: both updates use values from the old state.
#[derive(Clone, Debug, Default)]
pub struct Euler;

impl Integrator for Euler {
    fn step(&mut self, state: &mut State, dt: f64, acceleration: &dyn Fn(&State) -> Vec<Vec2>) {
        // Cache the old velocities so the position update cannot accidentally
        // use the updated velocity (which would be semi-implicit Euler).
        let v_n = state.velocities.clone();
        let a_n = acceleration(state);

        // x_{n+1} = x_n + v_n * dt
        for (i, position) in state.positions.iter_mut().enumerate() {
            position[0] += dt * v_n[i][0];
            position[1] += dt * v_n[i][1];
        }

        // v_{n+1} = v_n + a_n * dt
        for (velocity, acceleration_i) in state.velocities.iter_mut().zip(a_n.iter()) {
            velocity[0] += dt * acceleration_i[0];
            velocity[1] += dt * acceleration_i[1];
        }
    }
}

/// Velocity-Verlet with the initial acceleration cached before the first step.
#[derive(Clone, Debug, Default)]
pub struct VelocityVerlet {
    previous_acceleration: Option<Vec<Vec2>>,
}

impl Integrator for VelocityVerlet {
    fn initialize(&mut self, state: &State, acceleration: &dyn Fn(&State) -> Vec<Vec2>) {
        self.previous_acceleration = Some(acceleration(state));
    }

    fn step(&mut self, state: &mut State, dt: f64, acceleration: &dyn Fn(&State) -> Vec<Vec2>) {
        let a_n = self
            .previous_acceleration
            .take()
            .expect("VelocityVerlet::initialize must be called before step");
        let half_dt = 0.5 * dt;

        // v_half = v_n + 0.5 * dt * a_n
        let mut v_half = state.velocities.clone();
        for (velocity, a) in v_half.iter_mut().zip(a_n.iter()) {
            velocity[0] += half_dt * a[0];
            velocity[1] += half_dt * a[1];
        }

        // x_{n+1} = x_n + dt * v_half
        for (i, position) in state.positions.iter_mut().enumerate() {
            position[0] += dt * v_half[i][0];
            position[1] += dt * v_half[i][1];
        }

        // a_{n+1} = acceleration(x_{n+1}); one new evaluation per step.
        let a_next = acceleration(state);

        // v_{n+1} = v_half + 0.5 * dt * a_{n+1}
        for (velocity, a) in v_half.iter_mut().zip(a_next.iter()) {
            velocity[0] += half_dt * a[0];
            velocity[1] += half_dt * a[1];
        }

        state.velocities = v_half;
        self.previous_acceleration = Some(a_next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_updates_positions_with_old_velocities() {
        let mut state = State {
            positions: vec![[1.0, 0.0], [0.0, 0.0]],
            velocities: vec![[1.0, 0.0], [0.0, 0.0]],
        };
        let constant_acceleration = |_: &State| vec![[2.0, 0.0], [2.0, 0.0]];
        let mut euler = Euler::default();

        euler.step(&mut state, 1.0, &constant_acceleration);

        // Old-velocity update: x1 = 1 + 1 = 2, not 3.
        assert_eq!(state.positions[0], [2.0, 0.0]);
        assert_eq!(state.velocities[0], [3.0, 0.0]);
    }

    #[test]
    fn velocity_verlet_uses_half_velocity_and_caches_next_acceleration() {
        let mut state = State {
            positions: vec![[1.0, 0.0], [0.0, 0.0]],
            velocities: vec![[1.0, 0.0], [0.0, 0.0]],
        };
        let mut verlet = VelocityVerlet::default();
        let first_acceleration = |_: &State| vec![[2.0, 0.0], [2.0, 0.0]];

        verlet.initialize(&state, &first_acceleration);
        assert!(verlet.previous_acceleration.is_some());

        let second_acceleration = |_: &State| vec![[4.0, 0.0], [4.0, 0.0]];
        verlet.step(&mut state, 1.0, &second_acceleration);

        // v_half = [2, 0], x1 = 1 + 2 = 3, v1 = 2 + 2 = 4.
        assert_eq!(state.positions[0], [3.0, 0.0]);
        assert_eq!(state.velocities[0], [4.0, 0.0]);
        assert!(verlet.previous_acceleration.is_some());
    }
}
```

- [ ] **Step 2: Wire the module into the library**

In `week2/md/src/lib.rs`, after the `experiment` module declarations added in
Task 3, add:

```rust
mod integrator;
pub use integrator::{Euler, Integrator, VelocityVerlet};
```

- [ ] **Step 3: Run the library unit tests**

Run:

```bash
cargo test --lib --manifest-path week2/md/Cargo.toml
```

Expected: PASS for the existing unit tests plus the two new integrator tests.

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/integrator.rs week2/md/src/lib.rs
git commit -m "feat: add Integrator trait with Euler and velocity-Verlet"
```

---

### Task 5: Shared experiment driver (GREEN)

**Files:**
- Modify: `week2/md/src/experiment.rs` (add imports, `ExperimentResult`,
  `run_experiment`, and one driver unit test)
- Modify: `week2/md/src/lib.rs` (export the new driver API)
- Test: `week2/md/tests/dimer.rs` (created in Task 1)

**Interfaces:**
- Consumes: `State`, `Vec2`, `two_atom_initial_state`, `lj_accelerations`,
  the energy functions, and `Integrator` implementations.
- Produces: `md::ExperimentResult` and
  `md::run_experiment<I: Integrator>(&mut I, usize, f64) -> ExperimentResult`.

- [ ] **Step 1: Add the driver implementation**

In `week2/md/src/experiment.rs`, replace the import block at the top with:

```rust
use crate::integrator::Integrator;
use crate::state::{two_atom_initial_state, State, Vec2};
use crate::{lj_energy, lj_force};
```

Then insert this code immediately after the `relative_energy_error` function
and before the existing `#[cfg(test)]` module:

```rust
/// Time series produced by the shared two-atom experiment driver.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentResult {
    pub steps: Vec<usize>,
    pub times: Vec<f64>,
    pub relative_energy_errors: Vec<f64>,
}

/// Run the fixed Week 2 dimer with any `Integrator`.
///
/// `steps` counts integrated steps. The returned series includes step 0 with
/// error 0.0, so its length is `steps + 1`.
pub fn run_experiment<I: Integrator>(
    integrator: &mut I,
    steps: usize,
    dt: f64,
) -> ExperimentResult {
    let mut state = two_atom_initial_state();
    let e0 = total_energy(&state);
    let accelerations = |s: &State| lj_accelerations(s);

    integrator.initialize(&state, &accelerations);

    let mut step_indices = Vec::with_capacity(steps + 1);
    let mut times = Vec::with_capacity(steps + 1);
    let mut errors = Vec::with_capacity(steps + 1);

    for step in 0..=steps {
        if step > 0 {
            integrator.step(&mut state, dt, &accelerations);
        }
        step_indices.push(step);
        times.push(step as f64 * dt);
        errors.push(relative_energy_error(&state, e0));
    }

    ExperimentResult {
        steps: step_indices,
        times,
        relative_energy_errors: errors,
    }
}
```

Finally, inside the existing `#[cfg(test)] mod tests` in
`week2/md/src/experiment.rs`, add this unit test:

```rust
#[test]
fn experiment_records_step_zero_and_every_integrated_step() {
    let result = run_experiment(&mut crate::Euler::default(), 2, 0.5);
    assert_eq!(result.steps, vec![0, 1, 2]);
    assert_eq!(result.times, vec![0.0, 0.5, 1.0]);
    assert_eq!(result.relative_energy_errors[0], 0.0);
}
```

- [ ] **Step 2: Export the driver API**

In `week2/md/src/lib.rs`, change the `pub use experiment::{...};` line from
Task 3 so it also exports `ExperimentResult` and `run_experiment`:

```rust
pub use experiment::{
    kinetic_energy, lj_accelerations, potential_energy, relative_energy_error,
    run_experiment, total_energy, ExperimentResult,
};
```

- [ ] **Step 3: Run the full test suite to verify GREEN**

Run:

```bash
cargo test --manifest-path week2/md/Cargo.toml
```

Expected: all tests pass — the pre-existing `tests/lj.rs` suite, all library
unit tests, and every test in the Task 1 `tests/dimer.rs` acceptance suite,
including:

- Euler final relative energy error `> 0.5`;
- velocity-Verlet maximum absolute relative energy error `< 1e-3`;
- the 5000-step velocity-Verlet run;
- shared-driver coverage of both integrators;
- the acceleration-evaluation-count check.

This is the required GREEN state.

- [ ] **Step 4: Commit**

```bash
git add week2/md/src/experiment.rs week2/md/src/lib.rs
git commit -m "feat: add shared two-atom experiment driver"
```

---

### Task 6: Final verification

**Files:**
- None (verification only).

- [ ] **Step 1: Re-run the full suite from a clean state**

Run:

```bash
cargo test --manifest-path week2/md/Cargo.toml
```

Expected: PASS for all tests. Confirm `git status --short` shows no unexpected
changes outside the committed implementation and test files.

- [ ] **Step 2: Confirm the pre-existing LJ tests were untouched**

Check that `week2/md/tests/lj.rs` appears nowhere in the commits made during
execution:

```bash
git log --oneline -- week2/md/tests/lj.rs
```

Expected: only the original Week 2 commits that created and validated the LJ
tests; no new commit modifies that file.

