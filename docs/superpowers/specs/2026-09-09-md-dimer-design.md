# md Dimer Simulation Design

Date: 2026-09-09
Status: approved by user in brainstorming, pending spec review

## Purpose

Extend the `md` crate in `week2/` into a two-atom (dimer) molecular dynamics
simulator. Two integrators — forward Euler and velocity-Verlet — share one
`Integrator` trait so that one experiment driver can run either method from the
same initial state. The experiment's output is the relative total-energy error
measured at every step.

The pair potential is already implemented and tested: `energy(r) = 4 (r^-12 - r^-6)`
and `force(r) = 24/r (2 r^-12 - r^-6)` in reduced units (`m = sigma = epsilon = 1`),
free functions in `lib.rs`.

## Experiment specification

- Two atoms in 2D, masses 1, open boundaries, plain Lennard-Jones potential.
- Initial state: atoms at rest at `(0, 0)` and `(1.2, 0)`, separation 1.2.
- Timestep `dt = 0.01`.
- 500 steps for both methods; 5000 steps for velocity-Verlet alone.
- Energy: `E(t) = (1/2) sum_i |v_i|^2 + sum_{i<j} U(r_ij)`, `E0 = E(0) = U(1.2)`
  (both atoms start at rest).
- Output: the relative energy error `(E(t) - E0) / |E0|` at every step.
- Acceptance criteria on the shared initial state:
  - velocity-Verlet maximum error below `1e-3` on the 500-step run;
  - velocity-Verlet maximum error below `1e-3` on the 5000-step run;
  - Euler's final error above `0.5` on the 500-step run.

## Architecture

Two new modules with one responsibility each; `lib.rs` keeps the existing free
`energy`/`force` functions plus the experiment driver, initial state, and tests.

### `src/system.rs` — owned particle state

```rust
pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,   // cache, private
}
```

- `System::new(positions, velocities)` computes the initial accelerations.
- Accelerations come from the pair forces: with `d = x_i - x_j`, `r = |d|`,
  the force on atom `i` is `force(r) * d / r` (m = 1, so acceleration = force).
  Implemented as a general `i < j` pair loop (the same code serves larger N later).
- `accelerations` is a cache that integrators refresh at the positions they need.
- Reading methods (`pair_energy`, `kinetic_energy`, `n_atoms`) borrow `&self`;
  the acceleration update borrows `&mut self`.

### `src/integrator.rs` — the shared interface

```rust
pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}
```

Both implementations are stateless unit structs. The only difference between the
methods is the step body; all state lives in `System`.

- `pub struct Euler` — forward Euler:
  - `x_{n+1} = x_n + v_n * dt`
  - `v_{n+1} = v_n + a_n * dt`
  using the accelerations at the current positions.
- `pub struct VelocityVerlet` — one force evaluation per step:
  - `v_{n+1/2} = v_n + (dt/2) * a_n` (half kick with the cached `a_n`)
  - `x_{n+1} = x_n + dt * v_{n+1/2}` (drift)
  - `a_{n+1} = accelerations(x_{n+1})` (refresh the cache)
  - `v_{n+1} = v_{n+1/2} + (dt/2) * a_{n+1}` (half kick with the new `a_n`)

At the start of the first step the cache holds `a(x_0)` from `System::new`, so the
Verlet invariant "the cache equals the acceleration at the current positions"
holds inductively.

### `lib.rs` — driver, initial state, tests

```rust
pub fn run<I: Integrator>(integrator: &I, system: &mut System, dt: f64, steps: usize) -> Vec<f64>
```

- Records `e0 = E(0)` before stepping.
- Applies `integrator.step` `steps` times.
- Returns the relative energy error measured after each step.

```rust
pub fn dimer() -> System
```

- Two atoms at `(0, 0)` and `(1.2, 0)`, velocities zero.

## Testing

Tests live in `lib.rs` under `#[cfg(test)]`, following the existing pattern.
Existing tests (`greeting`, energy well depth, force-vs-derivative) stay
untouched.

One dimer test drives both methods through the same generic `run` from the same
`dimer()` state, so the comparison is fair by construction:

- `dimer_conservation`
  - 500 steps, `dt = 0.01`, velocity-Verlet: `max |error| < 1e-3`.
  - 500 steps, `dt = 0.01`, Euler: final error `> 0.5`.
  - 5000 steps, `dt = 0.01`, velocity-Verlet: `max |error| < 1e-3`.

Workflow order: write the failing dimer test first (red), then implement
`System`, `run`, and the two integrators until green, committing after each
stage.

## Out of scope

- Plotting the error series (a later step, separate request).
- Periodic boundary conditions, thermostat, N > 2 fluid runs, and the `md`
  command-line interface.
- The time-reversal experiment.
