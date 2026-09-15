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

Statements are labeled **[Course Requirement]** only where the course
learning sheet states them. **[Suggestion]** marks this spec's choices, and
**[Need confirmation]** marks details the sheet does not determine.

## Heating contract

**[Course Requirement]**

1. `md run` gains an optional flag `--ramp-to <temperature>`.
2. Without `--ramp-to`, the current equilibrium-fluid behavior is preserved
   exactly: equilibration rescales velocities initially and every 50
   equilibration steps; ordinary production has the thermostat OFF; the
   default contract run and its energy-drift check are unchanged.
3. With `--ramp-to`, during PRODUCTION the velocities are rescaled as a
   heating thermostat toward a target that rises linearly with the
   production step.
4. The target runs from `--temperature` at production step 0 to `--ramp-to`
   at the last production step.
5. `ramp_to` is recorded in run.json for heating runs.
6. Heating deliberately injects energy, so the unheated energy-conservation
   criterion is not applied to heated runs; the existing secular-drift bound
   is not weakened; heating output is kept separate from the normal
   `artifacts/` contract run.
7. The final force-method behavior is preserved: `cells` is the default;
   `--force naive` remains available.
8. All existing Part 2–4 tests and scientific tolerances are preserved.

Note: the course does **not** say that the heating-production thermostat
rescales every 50 production steps. The production-heating cadence is
therefore **[Need confirmation]** and an interpretation is proposed below as
**[Suggestion]**. (The every-50 cadence in the sheet applies to
*equilibration*, which is unchanged.)

## Ramp target (the course-derived function) [Course Requirement]

Let `T0 = --temperature`, `T1 = --ramp-to`, `S = --steps` (production
integrates steps `1..=S`; "last production step" is `S`).

For the production coordinate `s` in `[0, S]` the ramp target is

```
T_target(s) = T0 + (T1 - T0) * s / S
```

with

```
T_target(0) = T0
T_target(S) = T1
```

This is the mathematical ramp target. It is separate from the thermostat
event schedule below (the schedule decides *when* the velocity rescale to
`T_target(s)` is applied).

## Thermostat event schedule

The cadence of the heating-production rescale is **[Need confirmation]**.

**[Suggestion] Per-step rescale.** For a heated production run, rescale after
every production integration step `s = 1, 2, ..., S`, applying
`T_target(s)` via the existing uniform `rescale_to` (mass = 1, COM
preserved). There is no production rescale at step 0: step 0 is only the
mathematical start of the ramp, and the system enters production after
equilibration already at `T0`.

Why this suggestion:

- It directly satisfies the course's stated final endpoint: the last
  rescale, at `s = S`, applies `T_target(S) = T1` exactly, for every value
  of `S`.
- It works for every `S`; no divisibility condition is involved.
- It avoids inventing a special divisibility-by-50 condition for production
  heating (the sheet specifies every-50 only for equilibration).

This replaces any earlier statement that a non-multiple-of-50 run "fails to
apply T1" — that behavior is not established by the course and is not part
of this spec. Under the suggested per-step schedule the endpoint `T1` is
always applied at step `S`.

## Degenerate and absent cases

- `--ramp-to == --temperature` (flat ramp): the schedule is constant `T0`;
  a flat production thermostat. Allowed. **[Suggestion]** (the sheet does
  not state this case).
- `--ramp-to < --temperature` (cooling ramp): whether this must be rejected
  is **[Need confirmation]**; **[Suggestion]** allow any positive `T1`,
  including cooling.
- `--ramp-to <= 0`: rejected by the same positivity rule as `--temperature`.
  **[Suggestion]** (the sheet does not state a rule).
- `--ramp-to` absent: production thermostat remains completely OFF and
  run.json carries no `ramp_to` key (see run.json below).
- Equilibration is never affected by `--ramp-to`; the ramp applies to
  production only **[Course Requirement: "during PRODUCTION"]**.

## run.json

- **[Course Requirement]** Heating runs record `ramp_to` in run.json.
- **[Suggestion]** Representation: `SimConfig` gains
  `pub ramp_to: Option<f64>` (Default `None`); `RunConfig` (run.json) gains
  `ramp_to: Option<f64>` with `#[serde(skip_serializing_if =
  "Option::is_none")]`. Heating runs therefore serialize `"ramp_to": <T1>`
  alongside the existing Part 4 keys; unheated runs omit the key entirely so
  the existing Part 4 run.json schema remains byte/schema compatible, and
  old unheated run.json files still deserialize (a missing Option field
  defaults to None).

## Heating output separation and verification

- The default contract run and `make reproduce` never pass `--ramp-to`;
  `artifacts/` remains the unheated contract output. A heating run is
  directed to a separate `--out` directory. **[Suggestion: documented
  convention, not a hard guard]**
- `md check` is unchanged and is not used as the heated-run
  energy-conservation verifier. On a heated run the secular-drift bound will
  legitimately FAIL (energy is injected); that is expected and not an
  acceptance criterion. Whether `md check` should skip the drift gate when
  `ramp_to` is present is **[Need confirmation]**; **[Suggestion]** do not
  change the checker — heated runs are verified structurally (frames,
  steps, energies, box) and for the ramp schedule only.
- No energy-conservation acceptance condition is defined for heated runs
  [Course Requirement: heating injects energy].

## Testable correctness

The tests below target the suggested per-step schedule. Pure unit tests
cover the ramp function; integration tests cover the schedule, run.json, and
the absent-flag path. Tests must independently detect:

1. **Wrong linear interpolation** — unit-test the schedule helper
   `ramp_target(t0, t1, s, S)`:
   - `ramp_target(T0, T1, 0, S) == T0` (exact);
   - `ramp_target(T0, T1, S, S) == T1` (exact);
   - an interior step is the correct linear interpolation, e.g.
     `ramp_target(0.2, 1.2, 100, 200) == 0.7` (exact, within 1e-15).
2. **Off-by-one endpoint errors** — `ramp_target(0, S) == T0` and
   `ramp_target(S, S) == T1` are exact endpoints; no off-by-one at the
   boundaries.
3. **Per-step rescale schedule** — with `--ramp-to`, every saved production
   frame at step `s` has thermodynamic temperature pinned to
   `T_target(s)` (within tight tolerance), including an intermediate frame
   (e.g. step 25), so an omitted or mis-cadenced rescale is detected.
4. **Final application uses exactly T1** — the last production frame (step
   `S`) has thermodynamic temperature equal to `--ramp-to` within the same
   tight tolerance, for `S` both divisible and not divisible by 50.
5. **Accidental production thermostat when `--ramp-to` absent** — the
   existing `no_thermostat_rescaling_during_production` test stays; no
   `ramp_to` key appears in the unheated run.json.
6. **Heating run.json records `ramp_to`** — a heated run's run.json contains
   `"ramp_to": <value>`.
7. **Unheated run.json omission (compatibility)** — **[Suggestion]** a
   no-ramp run's run.json does not contain the `ramp_to` key, preserving the
   Part 4 schema; the `io` round-trip preserves `Option<f64>` (missing →
   None).

Unheated scientific tolerances and the energy-drift tests are unchanged.

## Preservation guarantees [Course Requirement]

- Unheated behavior identical (production thermostat off, same artifacts,
  same bounds); all existing Part 2–4/5 tests and tolerances unchanged.
- Force methods unchanged: `cells` default, `--force naive` available.
- The ramp reuses the existing uniform `rescale_to` thermostat primitive; no
  new physics, no new integrator, no COM re-removal.
- run.json schema for unheated runs unchanged.

## Non-goals

- No implementation plan yet.
- No change to the unheated contract acceptance (drift bound etc.).
- No Verlet lists / further optimization.
