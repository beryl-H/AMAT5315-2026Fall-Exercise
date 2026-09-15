# Week 2: Heating / Melting (`md run --ramp-to`) — Spec

Date: 2026-09-15
Status: Draft (SPEC only; no implementation plan yet)

## Current status (observed)

Heating is not implemented yet. The flag is absent from `RunArgs`, so the
intended heating command

```
md run --n 400 --temperature 0.2 --ramp-to 1.2 --steps 20000 \
  --sample-every 100 --out docs
```

currently fails with

```
error: unexpected argument '--ramp-to' found
```

This failure is expected before implementation: this SPEC defines the target
behavior (flag accepted, linear heating ramp applied), and the flag is added
in a later implementation step.

## Scope and references

This document is the Week 2 spec. It extends the already-approved Week 2
work; the following remain in force unchanged and are not re-specified here:

- `docs/superpowers/specs/2026-09-10-week2-md-equilibrium-fluid-design.md`
  (Part 4 equilibrium-fluid contract: lattice, shifted LJ cutoff, Schedule B
  equilibration, thermostat-off production, run.json/traj.jsonl artifacts,
  `md check` bounds, `md video`, `make reproduce`).
- `docs/superpowers/specs/2026-09-15-week2-md-cell-list-design.md`
  (Part 5 cell-list force: `--force naive|cells`, `cells` is the default).

Statements are labeled **[Course Requirement]** where they come from the
course instructions and **[Suggestion]** where they are this spec's choices.
Points the course wording does not determine are marked **[Need
confirmation]** with a proposed interpretation given as **[Suggestion]**.

## Heating contract [Course Requirement]

1. `md run` gains an optional flag `--ramp-to <temperature>`.
2. Without `--ramp-to`, the current equilibrium-fluid behavior is preserved
   exactly: equilibration thermostat (Schedule B) unchanged; production
   thermostat OFF; the default contract run and its energy-drift check
   unchanged.
3. With `--ramp-to`, during PRODUCTION the velocities are periodically
   rescaled as a heating thermostat.
4. The production thermostat acts every 50 production steps.
5. Its target temperature rises linearly from `--temperature` at production
   step 0 to `--ramp-to` at the final production step.
6. `ramp_to` is recorded in run.json for heating runs.
7. Heating deliberately adds energy, so:
   - the existing unheated energy-conservation contract is not redefined;
   - the existing secular-drift bound is not weakened;
   - heating output is kept separate from the normal `artifacts/` contract
     run.
8. The final force-method behavior is preserved: `cells` is the default;
   `--force naive` remains available.
9. All existing Part 2–4 tests and scientific tolerances are preserved.

## Mathematical ramp schedule

Let `T0 = --temperature`, `T1 = --ramp-to`, `S = --steps` (production
integrates steps `1..=S`; "final production step" is therefore `S`).

### Target temperature at each rescale event [Course Requirement + this spec]

At production step `s` (`0 <= s <= S`) the ramp target is

```
T_target(s) = T0 + (T1 - T0) * s / S
```

so `T_target(0) = T0` and `T_target(S) = T1` — the ramp is linear and reaches
the requested endpoint at the final production step.

### Rescale event schedule [Course Requirement: every 50 production steps]

After production step `s` with `s % 50 == 0` and `s >= 50`, rescale the
velocities to `T_target(s)` (uniform rescale via the existing
`rescale_to`, mass = 1, COM preserved). There is no rescale at production
step 0 (the velocities already sit at `T0` from the end of equilibration).

- `S` divisible by 50: events occur at `s = 50, 100, ..., S`; the final
  event at `s = S` targets `T_target(S) = T1`, so the endpoint is applied.
- `S` not divisible by 50: events occur at `s = 50, 100, ...,
  50*floor(S/50)`; the last event targets
  `T_target(50*floor(S/50)) < T1`, so the endpoint value is approached but
  never exactly applied. **[Need confirmation]** whether a forced final
  rescale at step `S` (to `T1` exactly) is required for non-divisible `S`.
  **[Suggestion]** Keep the strict 50-step cadence (consistent with
  equilibration's Schedule B) and do not add a forced step-`S` event; the
  endpoint is reached exactly iff `S % 50 == 0`. Validate with tests on
  `S % 50 == 0` (endpoint reached) and a non-divisible `S` (last event at
  the largest multiple of 50).

### Degenerate and absent cases

- `--ramp-to == --temperature` (flat ramp): the schedule is constant `T0`;
  rescaling every 50 production steps to `T0` is allowed and behaves like a
  constant-temperature production thermostat. Not an error. **[Suggestion]**
- `--ramp-to` absent: production thermostat remains OFF and run.json carries
  no `ramp_to` key (see below).
- `--ramp-to <= 0`: rejected by the same positivity rule as
  `--temperature` **[Suggestion]**. Whether `--ramp-to < --temperature`
  (a cooling ramp) must be rejected is **[Need confirmation]**; **[Suggestion]**
  allow any positive `T1`, including cooling, since the course only says the
  feature "adds energy" for the intended heating use.
- Equilibration is never affected by `--ramp-to`; the ramp applies to
  production only **[Course Requirement: "during PRODUCTION"]**.

## run.json representation

- `SimConfig` gains `pub ramp_to: Option<f64>` (Default `None`).
- `RunConfig` (run.json) gains `ramp_to: Option<f64>` serialized as an
  **optional field present only when heating is used** — i.e.
  `#[serde(skip_serializing_if = "Option::is_none")]`. Deserialization of a
  missing `ramp_to` yields `None` (Option fields default to None in serde),
  so old unheated run.json files still read correctly.
- Heating runs: run.json contains `"ramp_to": <T1>` alongside the existing
  Part 4 keys.
- Unheated runs: run.json contains exactly the existing Part 4 key set
  (`n, rho, box, dt, temperature, eq_steps, steps, sample_every, seed,
  integrator`) — the existing contract is preserved byte-for-byte.

## Heating output separation

- The default contract run and `make reproduce` never pass `--ramp-to`;
  `artifacts/` remains the unheated contract output.
- A heating run must be directed to a separate `--out` directory
  **[Suggestion: documented convention, not a hard guard]**.
- `md check` is unchanged: recomputation and the three bounds are as in
  Part 4. On a heated run the secular-drift bound will legitimately FAIL
  (energy is injected); that is expected and is not an acceptance
  criterion. **[Need confirmation]** whether `md check` should skip the
  drift gate when `run.json.ramp_to` is present; **[Suggestion]** do NOT
  change the checker — the drift bound is a contract-run criterion, and
  heated runs are verified only for structural validity (frames, steps,
  energies, box) plus the ramp schedule itself.
- No energy-conservation acceptance condition is defined for heated runs
  [Course Requirement: heating adds energy; do not redefine the unheated
  contract].

## Testable correctness

Tests must independently detect (proposed locations: pure unit tests for the
schedule + `simulate`/`cli`/`io` integration tests):

1. **Wrong linear interpolation** — unit-test the schedule helper
   `ramp_target(t0, t1, s, S)`: `ramp_target(0.5, 1.5, 0, 100) == 0.5`,
   `== 1.0` at `s=50`, `== 1.5` at `s=100` (exact, within 1e-15).
2. **Off-by-one schedule errors** — test that the first production rescale
   is at step 50 (not 0, not 1), the cadence is exactly 50, and the event
   count for divisible `S` is `S/50`.
3. **Thermostat accidentally active when `--ramp-to` absent** — the existing
   `no_thermostat_rescaling_during_production` test stays; additionally
   assert run.json has no `ramp_to` key for a no-ramp run.
4. **Wrong 50-step cadence** — with a ramp, frames aligned at production
   steps 50, 100, ... have thermodynamic temperature pinned to
   `T_target(step)` (within tight tolerance), and an intermediate frame
   (e.g. step 25) is not pinned.
5. **Ramp endpoint not reaching the requested value** — with `S % 50 == 0`,
   the final frame (step `S`) has thermodynamic temperature equal to
   `--ramp-to` within the same tight tolerance (the step-`S` rescale applied
   `T1` exactly).
6. **run.json not recording `ramp_to`** — a heating run's run.json contains
   `"ramp_to": <value>`; a no-ramp run's run.json does not contain the key;
   `io` round-trip preserves `Option<f64>` (missing → None).

## Preservation guarantees [Course Requirement]

- Unheated behavior identical (production thermostat off, same artifacts,
  same bounds); all existing Part 2–4/5 tests and tolerances unchanged.
- Force methods unchanged: `cells` default, `--force naive` available.
- The ramp reuses the existing uniform `rescale_to` thermostat primitive;
  no new physics, no new integrator, no COM re-removal.
- run.json schema for unheated runs unchanged.

## Non-goals

- No implementation plan yet.
- No change to the unheated contract acceptance (drift bound etc.).
- No Verlet lists / further optimization.
