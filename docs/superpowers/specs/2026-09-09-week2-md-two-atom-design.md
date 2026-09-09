# Week 2: Two-Atom Molecular Dynamics in `week2/md` — Design

Date: 2026-09-09
Status: Approved (design only; no implementation yet)

## Objective

Extend the existing `md` crate in `week2/md/` into a two-atom molecular
dynamics experiment. The experiment must be able to run with either of two
integrators — Forward Euler or velocity-Verlet — through one shared
`Integrator` trait and one common experiment driver. The measured output is
the relative total-energy error over time.

The only intended difference between the two experiment runs is which
integrator implementation is supplied to the shared driver.

## Physical and numerical specification

Initial state:

- two atoms in 2D
- mass `m = 1` (implicit everywhere, including the kinetic-energy formula)
- open boundaries
- plain Lennard-Jones potential from Part 2 (`lj_energy`, `lj_force`)
- atom `i` position: `(0.0, 0.0)`
- atom `j` position: `(1.2, 0.0)`
- both initial velocities: `(0.0, 0.0)`

Runs:

- `dt = 0.01`
- 500 steps with Forward Euler
- 500 steps with velocity-Verlet
- an additional 5000-step velocity-Verlet run

Measurement:

- total energy `E = E_kin + E_pot`
- `E_kin = 0.5 * sum_i |v_i|^2` (mass 1)
- `E_pot = sum_{i<j} U(r_ij)` using the existing `lj_energy`
- relative energy error at time `t`: `(E(t) - E0) / abs(E0)`,
  where `E0` is the total energy of the initial state

Acceptance criteria (unchanged from the course specification):

- Forward Euler, `dt = 0.01`, 500 steps: final relative energy error `> 0.5`
- velocity-Verlet, `dt = 0.01`, 500 steps: maximum absolute relative energy
  error `< 1e-3`

Reference check performed during design (Python probe, not a repo artifact):

| Run | Final relative error | Maximum | Minimum |
| --- | ---: | ---: | ---: |
| Euler, 500 steps | +1.932 | +1.956 | +0.0005 |
| Velocity-Verlet, 500 steps | −9.4e-6 | +1.8e-4 | −3.3e-4 |
| Velocity-Verlet, 5000 steps | −2.0e-5 | +1.8e-4 | −3.3e-4 |

These numbers assume the exact update orders specified below.

## Non-goals

- No periodic boundary conditions.
- No variable masses, cutoffs, thermostatting, or other ensembles.
- No changes to the existing `lj_energy` / `lj_force` implementations or tests.
- No plotting artifact in this task unless the Week 2 workflow later requests one.
- No sabotage commit: the required workflow for Part 3 is
  design → plan → test (RED) → implement (GREEN), not a sabotage step.

## Proposed code structure

Inside `week2/md/`:

- `src/lib.rs` — keeps and re-exports the existing Lennard-Jones functions and
  exposes the new modules below.
- `src/state.rs` — `Vec2`, `State`, and the fixed two-atom initial state.
- `src/integrator.rs` — the `Integrator` trait, `Euler`, and
  `VelocityVerlet`.
- `src/experiment.rs` — Lennard-Jones accelerations, energy measurement, and
  the shared `run_experiment` driver.
- `tests/dimer.rs` — integration tests derived from the specification.

## State representation

```rust
type Vec2 = [f64; 2];

struct State {
    positions:  Vec<Vec2>,
    velocities: Vec<Vec2>,
}
```

The mass `m = 1` is documented rather than stored; both the kinetic-energy
formula and the reduced-unit potential omit it. `Vec` keeps the code agnostic
to atom count even though this task always constructs two atoms.

## Integrator trait

```rust
trait Integrator {
    /// One-time setup before the first step (default: nothing).
    fn initialize(
        &mut self,
        state: &State,
        acceleration: &dyn Fn(&State) -> Vec<Vec2>,
    ) {}

    /// Advance the state by exactly one step of size dt.
    fn step(
        &mut self,
        state: &mut State,
        dt: f64,
        acceleration: &dyn Fn(&State) -> Vec<Vec2>,
    );
}
```

`Euler` is a unit struct. `VelocityVerlet` carries the cached acceleration:

```rust
struct VelocityVerlet {
    previous_acceleration: Option<Vec<Vec2>>,
}
```

The `Option` represents “not yet initialized”; `run_experiment` always calls
`initialize` before stepping, so normal use never takes the `None` path.

## Forward Euler update order (exact)

Both updates must use the old state:

```text
a_n = acceleration(x_n)
x_{n+1} = x_n + v_n * dt
v_{n+1} = v_n + a_n * dt
```

Implementation rule to prevent accidental semi-implicit Euler: compute the
full acceleration vector from the old positions first, then update every
position using the old velocity, then update every velocity using the already
computed old acceleration.

## Velocity-Verlet update order (exact)

The initial acceleration is computed once before the first step and retained.
Each later step performs exactly one new force/acceleration evaluation and
retains `a_{n+1}` for reuse as the next step’s input:

```text
v_half  = v_n + 0.5 * dt * a_n
x_{n+1} = x_n + dt * v_half
a_{n+1} = acceleration(x_{n+1})
v_{n+1} = v_half + 0.5 * dt * a_{n+1}
```

Over a full run: one acceleration evaluation in `initialize`, then one new
evaluation per step — never a recomputation of the old acceleration.

## Shared experiment driver

```rust
fn run_experiment<I: Integrator>(
    integrator: &mut I,
    steps: usize,
    dt: f64,
) -> ExperimentResult
```

The driver:

1. builds the fixed two-atom initial state;
2. computes `E0 = total_energy(&state)`;
3. calls `integrator.initialize(...)`;
4. steps `steps` times, recording the relative energy error after every step;
5. returns the recorded series.

`ExperimentResult` contains step indices, times, and the relative-energy-error
series. It includes step 0 with error 0.0, so a 500-step run has 501 entries
and the final entry corresponds to step 500.

## Force / acceleration calculation

`lj_accelerations(state)` reuses the existing `lj_force` without modification.
For each pair `i < j`:

```text
d = r_i - r_j
r = |d|
acceleration_i += lj_force(r) * d / r
acceleration_j -= lj_force(r) * d / r
```

Mass 1 means force and acceleration are numerically equal; the public function
is named for acceleration because integrators consume accelerations.

## Energy measurement

Small public functions:

- `kinetic_energy(state)` — `0.5 * sum_i |v_i|^2`
- `potential_energy(state)` — `sum_{i<j} lj_energy(|r_i - r_j|)`
- `total_energy(state)` — kinetic plus potential
- `relative_energy_error(state, e0)` — `(E - E0) / abs(E0)`

`E0` is computed once by the driver from the initial state.

## Test structure (planned)

Integration tests in `tests/dimer.rs` will be derived from the specification,
not from the implementation:

- Forward Euler over 500 steps: final relative energy error `> 0.5`.
- Velocity-Verlet over 500 steps: maximum absolute relative energy error
  `< 1e-3`.
- Velocity-Verlet over 5000 steps: full run with finite, stable output.
- The same generic `run_experiment` driver is exercised with both concrete
  integrators, verifying the shared-driver design requirement.
- An acceleration-call counter verifies velocity-Verlet performs one new
  evaluation per step after initialization and retains `a_{n+1}`.
- Small unit checks pin the exact update formulas on a constant-acceleration
  state.

Implementation workflow after this design is approved:
design → plan → test (RED) → implement (GREEN).
