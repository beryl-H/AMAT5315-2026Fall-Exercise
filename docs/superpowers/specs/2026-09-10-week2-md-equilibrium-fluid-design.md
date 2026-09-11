# Week 2 Part 4: Equilibrium Lennard-Jones Fluid CLI in `week2/md` — Design

Date: 2026-09-10
Status: Approved (design only; no implementation yet)

## Objective

Extend the existing `md` crate in `week2/md/` with a single CLI binary that
simulates, verifies, and visualizes an equilibrium 2D Lennard-Jones fluid:

- `md run` simulates and records an equilibrium Lennard-Jones fluid.
- `md check` independently recomputes the physics from the saved trajectory
  and reports PASS/FAIL.
- `md video` renders the molecular motion together with the radial
  distribution function g(r).
- `make reproduce` (in `week2/`) regenerates the default contract run from a
  fresh clone.

All Part 2 (plain Lennard-Jones primitives) and Part 3 (dimer experiment,
integrators) behavior and tests must keep working. In particular the isolated
dimer continues to use the plain Lennard-Jones potential and open boundaries.

Statements are labeled **[Course Requirement]** where they come from the
learning sheet and **[Suggestion]** where they are this design's choices.

## Physical model [Course Requirement]

- 2D reduced units: sigma = epsilon = m = k_B = 1.
- Default N = 100, rho = 0.8.
- Initial state is a triangular lattice; default is 10 x 10.
- Lattice geometry:
  - a = sqrt(2 / (sqrt(3) * rho))
  - h = sqrt(3)/2 * a
  - rho = 1/(a*h) (consistent with the above)
  - Lx = sqrt(N) * a, Ly = sqrt(N) * h
  - coordinates: x_ij = (i + 0.5*(j mod 2))*a, y_ij = j*h,
    for i, j = 0..sqrt(N)-1
- General square N: sqrt(N) rows x sqrt(N) columns at the same lattice
  spacing. Non-perfect-square N is rejected with a clear nonzero CLI error
  **[Suggestion]** (the sheet does not specify this case); the particle
  count is never padded or silently changed.

### Periodic boundaries [Course Requirement]

- Periodic boundary conditions on both axes.
- Minimum-image displacement per axis: `d_min = d - L * round(d/L)`.
- After each step wrap positions into `0 <= x < Lx`, `0 <= y < Ly`.
- Velocities are unchanged by wrapping.

### Interaction [Course Requirement]

- Cutoff rc = 2.5.
- For r < rc: `U_cut(r) = U(r) - U(rc)`, where U is the existing Part 2
  `lj_energy`; the force is the existing Part 2 `lj_force(r)`.
- For r >= rc: `U_cut = 0` and force = 0.
- The shifted potential is continuous and approaches zero at rc; the force
  has a small finite jump at rc (since `lj_force(rc) != 0`). Nothing is
  smoothed.
- The existing plain-LJ/open-boundary path (Part 3 dimer) is preserved
  untouched.

### Velocity initialization and equilibration [Course Requirement]

- Velocity components are independent Gaussian samples with mean 0 and
  variance equal to the run temperature T (default T = 0.5).
- seed = 2026 (default).
- Subtract the two component-wise mean velocities once after the initial
  draw, so the centre-of-mass velocity is zero.
- `T_thermo = 2 * E_kin / (2*N - 2)`.
- Rescale every velocity component by `sqrt(T_target / T_thermo)`, where
  T_target is the run's `--temperature` value (not hard-coded 0.5).
- Thermostat schedule (Schedule B, documented for reproducibility): rescale
  initially before equilibration step 1, then after each equilibration step
  s for which `s % 50 == 0` (recomputing E_kin/T_thermo from the current
  velocities each time). With eq_steps = 2000 this gives rescales after
  steps 50, 100, ..., 2000 plus the initial one: 41 events. The COM
  velocity is not re-subtracted at later thermostat events.
- Equilibration = 2000 steps (default).

### Production [Course Requirement]

- Thermostat OFF.
- Production steps = 10000 (default), dt = 0.01 (default).
- Save every 50 steps (`--sample-every`); do not save production step 0.
- Saved steps are 50, 100, ..., 10000: exactly 200 frames for the default
  run.
- Production time starts from zero (t = step * dt with production step
  numbering).

### Energy-conservation note

- During production the thermostat is OFF, so there is no intentional energy
  injection or removal.
- The shifted potential is continuous at rc but non-smooth there.
- The force has a finite jump at rc.
- A cutoff crossing therefore does not create a finite potential-energy
  discontinuity in the exact model, but the force non-smoothness can degrade
  numerical integration behavior.
- Therefore Part 4 measures secular energy drift directly rather than
  reusing the Part 3 smooth-potential energy argument.

## Non-goals

- No changes to Part 2/3 functions, tests, or examples (regression only).
- No 3D, no neighbor lists, no performance work beyond what the mandated
  release verification needs.
- No Python/pip tooling anywhere in the design.
- ffmpeg is an external prerequisite for `md video` only; nothing in this
  project installs it, and `make reproduce` does not use it.
- No sabotage commits modifying the Part 3 simulator.

## Dependencies and RNG

Add to `Cargo.toml`: `serde` (derive), `serde_json`, `rand`, `rand_distr`,
`clap` (derive). Commit `Cargo.lock` so the submitted dependency versions
are pinned for fresh-clone reproduction.

**Implementation note [Suggestion]:** before writing `Cargo.toml`, verify
where the Gaussian sampler (`StandardNormal`) is provided for the selected
crate versions and pin accordingly. In current `rand` releases the sampler
lives in the separate `rand_distr` crate; the design therefore plans on
depending on both `rand` and `rand_distr`, both pinned in `Cargo.lock`.

RNG convention **[Suggestion]** (Course Requirement is only: seeded
independent Gaussian components with the specified variance):

- `StdRng::seed_from_u64(seed)`;
- StandardNormal draws atom-major: atom 0 (vx, vy), atom 1 (vx, vy), ...;
- `v = sqrt(config.temperature) * z`;
- component-wise mean subtraction once, then the initial thermostat
  rescale.
- Reproducibility claim is scoped to this repository with its committed
  lockfile; no claim that `StdRng`/distribution output is stable across all
  future `rand` versions.

## Module layout

```
src/
  lib.rs                 re-exports; Part 2/3 items unchanged
  state.rs               unchanged
  integrator.rs          unchanged; VelocityVerlet reused
  experiment.rs          unchanged: Part 3 dimer path
  system.rs              NEW periodic geometry: lattice, box, MIC, wrapping
  pair.rs                NEW shifted-cutoff LJ pair energy + scalar force
  fluid.rs               NEW accelerations/energies on State with periodic box
  thermostat.rs          NEW COM removal, T_thermo, rescale
  simulate.rs            NEW equilibration + production driver, frames
  io.rs                  NEW RunConfig JSON schema + Frame JSONL read/write
  metrics.rs             NEW drift, T_speed, chi2_22, g(r)
  checker.rs             NEW structural validation + check report
  render.rs              NEW RGBA frame rasterizer
  video.rs               NEW ffmpeg pipe
  cli.rs / main.rs       NEW clap subcommands
```

Units live in the library and are re-exported so the binary and integration
tests share them. `md check`'s independent physics lives in
`metrics.rs`/`checker.rs`, which compute everything from raw positions and
velocities.

### system.rs

- `a = sqrt(2/(sqrt(3)*rho))`, `h = sqrt(3)/2 * a`.
- General square N: `n = sqrt(N)` integer, else CLI error [Suggestion];
  positions `x = (i + 0.5*(j mod 2))*a`, `y = j*h`, i,j in 0..n;
  box `Lx = n*a`, `Ly = n*h`.
- `minimum_image(d, L) = d - L*round(d/L)` per axis.
- `wrap(p, [Lx,Ly]) -> [0,L)` via `p - L*floor(p/L)`; applied after each
  step; velocities untouched.
- Default N=100, rho=0.8: n=10, Lx=10a, Ly=10h.

### pair.rs — shifted cutoff (rc = 2.5)

- `U_shifted(r) = if r < rc { lj_energy(r) - lj_energy(rc) } else { 0.0 }`.
- `force_shifted(r) = if r < rc { lj_force(r) } else { 0.0 }`.
- The potential is continuous and approaches zero at rc; the force has a
  small finite jump at rc. No smoothing.
- Fluid force loop applies minimum-image displacements; the vector force on
  atom i from pair (i,j) is `f(r) * d_hat` with d minimum-imaged; the pair
  contribution is antisymmetric (+F to i, -F to j).

### fluid.rs

- `fluid_accelerations(state, &box) -> Vec<Vec2>` and
  `fluid_potential_energy(state, &box) -> f64` over all pairs with MIC +
  cutoff; kinetic energy reused from Part 3 (`kinetic_energy`).
- Newton's third law makes the analytic total internal force zero by pair
  antisymmetry. Numerically this is tested, not assumed: on random periodic
  configurations assert `|sum_i F_i,x| < tol` and `|sum_i F_i,y| < tol`
  with an explicit floating-point tolerance; no claim of bitwise zero.
- The dimer continues to use plain `lj_accelerations` with open boundaries.

### thermostat.rs

- `remove_com_velocity(state)`: subtract per-component means, once only.
- `thermodynamic_temperature(state) = 2 * E_kin / (2N - 2)`.
- `rescale(state, T_target)`: multiply every velocity component by
  `sqrt(T_target / T_thermo)`.

### simulate.rs — driver

One shared step loop = one velocity-Verlet step + wrap positions:

1. Build lattice state; Gaussian velocities (seeded); COM removal; initial
   thermostat rescale (event 1).
2. Equilibrate `eq_steps` (default 2000): after each equilibration step s
   with `s % 50 == 0`, recompute E_kin/T_thermo and rescale (Schedule B;
   default run: 41 events including the initial one). No COM re-subtraction.
3. Production: thermostat OFF; `steps` (default 10000) at `dt` (default
   0.01); production time starts at zero; save at steps `sample_every`,
   `2*sample_every`, ..., `steps` (not step 0) — exactly 200 frames for the
   default run.
4. Each frame stores wrapped positions, velocities, shifted E_pot, E_kin,
   step, and `t = step * dt` (production step numbering).
5. Velocity-Verlet is initialized (cached acceleration) once, before
   equilibration step 1.

## CLI

One `md` binary with clap-derived subcommands **[Suggestion]**; the
user-facing forms are Course Requirements:

- `md run --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 --eq-steps 2000
  --steps 10000 --sample-every 50 --seed 2026 --out artifacts`
  This must be equivalent to `md run --out artifacts` (all defaults).
  The `--out` default is the relative path `artifacts` resolved against the
  current working directory.
- `md check <ARTIFACTS>`: positional artifacts directory.
- `md video <ARTIFACTS> --out <PATH>`: `--out` is required explicitly
  (e.g. `md video artifacts --out artifacts/run.mp4`); the sheet does not
  specify a video `--out` default, so none is claimed.

Non-perfect-square N, malformed input, failed checks, and a missing ffmpeg
executable produce a clear error message and a nonzero exit status.

`steps` must be a positive multiple of `sample_every` (so the saved sequence
`sample_every, 2*sample_every, ...` ends exactly on `steps`); violations are
rejected at argument parsing with a nonzero exit [Suggestion].

## File formats [Course Requirement]

`run.json` (single JSON object) must contain:

`n, rho, box [Lx,Ly], dt, temperature, eq_steps, steps, sample_every, seed,
integrator = "velocity-verlet"`.

`traj.jsonl` (one JSON object per saved production frame):

`step, t = step*dt, pos` (wrapped), `vel`, `E_pot` (shifted potential),
`E_kin`.

The output directory is created as needed. Generated `artifacts/` stays out
of git. Numbers are serialized with serde_json's default round-trip f64
formatting (the strict energy cross-check tolerates it).

## `md check` — independent verification

Reads `run.json` and `traj.jsonl`; rebuilds the box from `n, rho` and cross-checks
the stored `box` within `1e-10 * max(1, L)` per axis [Suggestion]; **recomputes** E_pot
(minimum image + shifted cutoff) and E_kin from the raw saved positions and
velocities; the stored energies are only cross-checked and are never inputs
to the physics metrics.

### Structural validation (malformed input -> nonzero exit)

[Course Requirement: check must exit with an error on malformed files or
failed checks; the detailed list is a design [Suggestion] implementing it.]

- all required run.json fields present;
- `integrator == "velocity-verlet"`;
- frame count equals `steps / sample_every`;
- saved production step sequence matches `sample_every, 2*sample_every, ...,
  steps` and `t == step * dt`;
- positions inside the periodic box `[0, Lx) x [0, Ly)` for every axis;
- finite positions, velocities, and energy values;
- per-frame array lengths equal `n`;
- stored-energy cross-check [Suggestion]:
  `abs(stored - recomputed) <= 1e-10 * max(1, abs(recomputed))`.
  No bitwise equality required; the stored energies are never used in the
  drift/temperature/statistical calculations.

### Notation

- **F** = number of saved frames (default F = 200).
- **S** = number of pooled speed samples = F * N (default S = 20,000).

### Physics metrics

1. **Secular drift** with `k = max(1, floor(F/10))`:
   `abs(mean(E_last_k) - mean(E_first_k)) / abs(E0) < 2e-3`,
   where E are recomputed frame total energies and E0 is the recomputed
   total energy of the first saved frame.
2. **Temperature:** pool all S saved-frame speeds;
   `T_speed = mean(v^2)/2`;
   gate for a general saved run **[Suggestion]**:
   `abs(T_speed - run_config.temperature) < 0.05`;
   the Course Requirement is the default contract run (temperature 0.5),
   where this is exactly `abs(T_speed - 0.5) < 0.05`.
3. **Speed shape** (course definition, verbatim):
   - 24 equal-probability 2D Maxwell-Boltzmann (Rayleigh) bins with edges
     `b_k = sqrt(-2 * T_speed * ln(1 - k/24))` for k = 0..23 and
     `b_24 = infinity`; bin edges use T_speed, never fixed 0.5;
   - expected count per bin `E_b = S / 24`;
   - `chi2_22 = (1/22) * sum_b ((O_b - E_b)^2 / E_b) < 2`.
   - This is a reduced chi-square with 22 = 24 - 1 - 1 degrees of freedom
     (one constraint fixes the total count; T_speed is fitted from the same
     pooled data). The `< 2` bound is a practical tolerance for correlated
     saved samples, not a formal significance test.

`md check` prints a readable report (recomputed vs stored energies, the
three metric values, PASS/FAIL per bound) and exits 0 only if every check
passes and the files are well formed.

## `md video`

- Loads run.json/traj.jsonl and computes g(r) **once** from the entire
  saved trajectory (presentation choice [Suggestion]; the averaging and
  normalization are Course Requirements):
  - minimum-image unordered-pair histogram H_k pooled over all F frames;
  - 50 radial bins from 0 to `r_max = 0.5 * min(Lx, Ly)` [Suggestion];
  - `g_k = 2 * H_k / (N * F * rho * pi * (r_outer^2 - r_inner^2))`,
    the factor 2 accounting for each unordered pair being a neighbour of
    both atoms; equivalently ring counts normalized per atom by
    `rho * pi * (r_outer^2 - r_inner^2)` and averaged over atoms and frames.
- Renders RGBA frames in Rust (own rasterizer, no image crate): one panel
  with the periodic box and particle discs, one static panel with the
  whole-trajectory g(r) curve (with a g=1 reference line). Layout,
  resolution, and fps are [Suggestion] implementation details.
- Exactly one video frame per saved trajectory frame; no saved frame is
  skipped [Course Requirement].
- Frames are piped to the system `ffmpeg` executable as rawvideo
  (e.g. `-f rawvideo -pix_fmt rgba -s WxH -i pipe:0 ... -f mp4 -y <out>`)
  which encodes the MP4 [Suggestion for the exact invocation]. Encoding
  settings are chosen so the final MP4 stays under 2 MB [Course
  Requirement].
- If the `ffmpeg` executable is missing, `md video` fails with a clear
  nonzero error; no system package is installed by the program. Tests that
  need ffmpeg skip gracefully when it is absent.

## `week2/Makefile`

[Course Requirement: the sheet requires `week2/Makefile`.]

```make
reproduce:
	cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts
```

- Run from `week2/`, writing to `week2/artifacts/` (paths relative to the
  working directory).
- Responsibilities are separate:
  - `make reproduce` regenerates the default run only;
  - test verification is `cargo test --manifest-path md/Cargo.toml
    --release`;
  - physics verification is `cargo run --manifest-path md/Cargo.toml
    --release -- check artifacts`;
  - video generation is a separate `md video` step.
- Generated `artifacts/` stays out of git.

## Tests (test-first: failing tests before implementation)

Fast unit tests (part of ordinary `cargo test`):

- lattice geometry: a, h, box dimensions, rho consistency, default 10x10
  coordinates; general square N; non-square N rejected;
- minimum-image displacement and wrapping (including velocities unchanged);
- total internal force: `|sum_i F_i| < numerical tolerance` on random
  periodic configurations (analytic zero, numerical test);
- shifted cutoff, as separate tests:
  - potential continuity: values approaching rc from below tend to zero,
    and `U_shifted(r >= rc) = 0`;
  - force/derivative consistency: finite differences of `U_shifted` only at
    points safely below rc, with stencils that never straddle rc (the force
    jumps there);
  - piecewise force identity: equals Part 2 `lj_force` below rc, zero at
    and above rc;
- COM removal; T_thermo and rescale math; thermostat event count (41 for
  the default run) and no rescaling during production;
- frame timing: no step 0 saved; steps are `sample_every, 2*sample_every,
  ...`; generic frame count = steps/sample_every;
- checker metric functions, including chi2_22 on synthetic Maxwell data
  (PASS) and non-Maxwell data (FAIL), using the F/S notation; temperature
  metric tested against both 0.5 and non-default run temperatures.

CLI integration tests (small run in a temp directory):

- required run.json fields and `integrator == "velocity-verlet"`;
- run.json and traj.jsonl exist and parse;
- saved steps follow `sample_every`;
- generic frame count = steps/sample_every.

Full default-contract test (ordinary integration test in the normal release
suite — NOT `#[ignore]`; the course VERIFY command is
`cargo test --manifest-path md/Cargo.toml --release`):

- runs the real default simulation;
- exactly 200 frames with steps 50..=10000 and t = step*dt;
- `md check` logic passes all three physics bounds (drift with
  k = max(1, floor(F/10)); `abs(T_speed - 0.5) < 0.05`; chi2_22 < 2 with
  E_b = S/24);
- structural validation passes;
- one end-to-end negative control: start from valid temporary artifacts,
  deliberately corrupt raw trajectory data, run the real check path, and
  verify a nonzero exit.

Negative controls [Suggestion] (synthetic trajectories plus the end-to-end
corruption test above; no sabotage of the actual simulator):

- secular energy drift must FAIL;
- wrong temperature must FAIL;
- non-Maxwell speed shape must FAIL;
- malformed/missing data must FAIL with nonzero exit.

Regression: existing `tests/lj.rs`, `tests/dimer.rs`, and examples keep
passing; the Part 3 plain-LJ/open-boundary path is asserted unchanged.

Video test: with ffmpeg present, render and assert the MP4 exists and is
under 2 MB; skip when ffmpeg is absent.

Development workflow may use targeted fast tests while iterating, but the
Part 4 GREEN/final verification must run the full release command above
with the contract test included.

## Risks / notes

- Debug-build performance: the O(N^2) x 12,000-step contract run is sized
  for the mandated release verification command.
- Cutoff is strict at `r < rc`, so equality at rc falls in the zero branch;
  the force jump at rc is intentional and unsmoothed.
- Reproducibility (identical artifacts from a fresh clone) holds for the
  committed Cargo.lock; RNG and sampler details are [Suggestion]-level.
