# Week 2 Part 5: Cell-List Force Evaluation — Design

Date: 2026-09-15
Status: **Approved** (architecture approved with four corrections; design frozen; implementation plan not yet written)

Scope: `week2/md/` only. Design for a cell-list (`cells`) force/energy path alongside the existing naive path, plus the `--force naive|cells` CLI switch.

## Goal

Speed up the O(N²) pair loop in `md::fluid::fluid_accelerations` (measured 97% inclusive sample share in the naive profile) with a cell list, while keeping the original naive implementation available as the reference path. No Verlet neighbor lists and no more advanced optimization — the course requires only a cell list.

## 1. Final default contract [Course Requirement]

The final state of the crate is:

- `--force cells` is the **default** on the CLI.
- `SimConfig::default()` uses `ForceMethod::Cells`.
- `--force naive` remains available as the reference path and is never removed.

The later implementation **plan** may stage the default flip (start at `Naive`, flip to `Cells` only after the naive/cells equality tests pass) for safety, but that staging is **not** the final design contract. The final contract is `cells` as default.

## 2. `ForceMethod` enum [Course Requirement: no run.json change]

- `ForceMethod` is a small physics/runtime-selection enum: `pub enum ForceMethod { Naive, Cells }`.
- It does **NOT** carry `Serialize`/`Deserialize` and is **NOT** recorded in `run.json`. The Part 4 run.json schema contains no `force_method` field, and Part 5 does not request recording it, so the artifact schema is unchanged.
- CLI parsing uses a minimal mechanism (`FromStr` + `Display`, or clap `ValueEnum` — either is fine; no serde).

## 3. Architecture overview

Everything lives in `fluid.rs` (plus a `SimConfig` field and a CLI flag); `io`, `metrics`, `checker`, `render`, `video` are method-agnostic and unchanged.

- `pub fn fluid_accelerations(state, bx) -> Vec<Vec2>` — **naive** reference (unchanged behavior; used by existing tests, metrics, checker).
- `pub fn fluid_potential_energy(state, bx) -> f64` — **naive** reference (unchanged).
- `impl ForceMethod`:
  - `pub fn accelerations(self, state, bx) -> Vec<Vec2>` — dispatches `Naive`/`Cells`.
  - `pub fn potential_energy(self, state, bx) -> f64` — dispatches `Naive`/`Cells`.
- `SimConfig` gains `pub force_method: ForceMethod`; `run_simulation` dispatches through `config.force_method` for both the integrator's accelerations and the saved-frame `e_pot`.
- Private helpers in `fluid.rs`:
  - `pair_force(dx_raw, dy_raw, bx) -> (fx, fy)` and `pair_energy(dx_raw, dy_raw, bx) -> f64` — the **single source of pair physics** (minimum-image displacement + `shifted_force`/`shifted_energy`). Both naive and cells call these, so they agree to rounding by construction.
  - `cell_bins(...)` — build `Vec<Vec<usize>>` of particle indices per cell (rebuilt each force evaluation).
  - `neighbor_cells(c, nx, ny) -> Vec<usize>` — wrapped + deduplicated 3×3 neighbors.
  - `for_each_pair(...)` — shared pair iterator (see §6) driving both cells force and cells energy with an identical pair set.

## 4. Cell geometry [Course Requirement]

- Cutoff `rc = 2.5` (existing `pair::RC`).
- `nx = floor(Lx / rc)`; cell width `wx = Lx / nx`. Similarly along y: `ny = floor(Ly / rc)`, `wy = Ly / ny`.
- Because `nx <= Lx / rc`, the construction guarantees `wx >= rc` and `wy >= rc`. This invariant is what makes the 3×3 neighborhood search complete, and it must not be broken.
- Binning: positions are wrapped into `[0, Lx) × [0, Ly)` by the driver (wrap every step) and by the checker's in-box validation, so `cx = floor(p[0] / wx)` (clamped to `nx-1`), `cy = floor(p[1] / wy)`; cell index `c = cy*nx + cx`.
- Default contract: `Lx ≈ 12.014, Ly ≈ 10.404` → `nx = 4, ny = 4`, `wx ≈ 3.004`, `wy ≈ 2.601`, 16 cells.
- Naive profile case (N=400): `Lx ≈ 24.028, Ly ≈ 20.809` → `nx = 9, ny = 8`, `wx ≈ 2.670`, `wy ≈ 2.601`, 72 cells.

### Small-box guard [Suggestion]

The stated course rule is exactly `nx = floor(Lx / rc)` with no special case. A box with `L < rc` would give `floor(L/rc) = 0` and divide by zero; this cannot occur for any valid course run (contract and profile boxes are far larger than rc). If defensive support for `L < rc` is retained (e.g. `nx = max(1, floor(Lx/rc))`), it is **[Suggestion]** and outside the stated course rule — it must not be presented as a course requirement.

## 5. Neighbor enumeration [Course Requirement]

- For each cell at `(cx, cy)`, enumerate the 9 offsets `dx, dy ∈ {-1, 0, 1}`.
- Wrap each axis periodically: `(cx + dx).rem_euclid(nx)`, `(cy + dy).rem_euclid(ny)`.
- **Deduplicate** the resulting wrapped cell indices (e.g. a small `Vec<usize>` with a linear `contains` — 9 candidates). This is what the two-cell-wide-box case exercises: e.g. `nx = 2, cx = 0` gives x-offsets {−1,0,1} → wrapped {1,0,1} → dedup {0,1}.
- Per the approved architecture the search is **per particle**: for each particle `i`, scan the 3×3 neighborhood of `i`'s cell.

## 6. Pair deduplication [Course Requirement: pairs evaluated only once]

- Recommended strategy (approved): per-particle enumeration with a global `j > i` guard.
  - For each particle `i`, for each deduplicated neighbor cell, for each `j` in that cell with `j > i`: evaluate the pair.
  - The `j > i` guard means a pair is applied only from the lower-index side even though it is seen from both particles' searches.
  - Cell-index dedup prevents the same `j` being revisited from one `i`.
  - Together these guarantee every unordered pair is applied **exactly once**.
- Energy uses the **identical** enumeration: one private `for_each_pair` drives both cells force and cells energy, so forces and energies agree and neither counts a pair twice. (The naive side uses the same iterator in all-`i<j` form.)
- The antisymmetric `+f / -f` accumulation (Newton's third law) is preserved in both paths.

## 7. Rebuild policy

- The cell bins are **rebuilt each force/energy evaluation** (approved architecture). Positions change every MD step, so incremental cell maintenance has no benefit here and is not required.
- Binning is O(N); the cost is negligible next to the pair loop.

## 8. CLI / SimConfig propagation [Course Requirement]

- `RunArgs` gains `--force <naive|cells>`; **default is `cells`** (final contract). `ForceMethod` parses via `FromStr`/clap `ValueEnum`.
- `run_command` copies `args.force` into `SimConfig.force_method`.
- `SimConfig::default()` sets `force_method = ForceMethod::Cells` (final contract; the plan may stage the flip).
- `run_simulation` uses `config.force_method.accelerations(...)` for the integrator and `config.force_method.potential_energy(...)` for saved `e_pot`; `e_kin` is unchanged.
- **run.json is unchanged** — no `force_method` field.

## 9. Checker / metrics [design decision]

- `metrics::frame_total_energies` and `checker::check_artifacts` recompute with the **naive** `fluid_potential_energy` regardless of the run's method.
- This is safe because naive and cells agree to rounding (~1e-15 relative), far inside the checker's stored-energy cross-check tolerance `1e-10 * max(1, |recomputed|)`. The physics bounds (`drift`, `T_speed`, `chi2_22`) use raw positions/velocities and are method-agnostic.

## 10. Required correctness tests (to be written in the implementation phase)

Course-required equality/coverage cases (compare cells vs naive within a tolerance):

1. Naive and cells **forces** agree within tolerance on the lattice.
2. Naive and cells **potential energies** agree within tolerance on the lattice.
3. **Perturbed** configurations (deterministic displacements, not only perfect symmetric lattices) — forces and energies agree.
4. Pairs interacting **across a periodic boundary** (atoms near `x=0` and `x=Lx` whose minimum-image separation < rc) agree.
5. Pairs **exactly at the cutoff** (`r == rc`) agree. Nuance: if such a pair sits in cells 2 apart (possible when `r = rc = w`), cells will not find it and naive will, but the contribution is zero either way (`shifted_force(rc) = 0`, `shifted_energy(rc) = 0`), so totals must still agree.
6. A **two-cell-wide periodic box** (small box forcing `nx = ny = 2`) where wrapped neighbor indices duplicate — no double counting: cells total ≈ naive total, and the Newton's-third-law sum ≈ 0.
7. Existing Part 2–4 suite stays green (the naive path in `tests/fluid.rs`, metrics, checker, contract, cli, video, etc. is unchanged).

Additional coverage:

- **[Suggestion] `r = rc - ε`** — at exactly `rc` both shifted energy and force are zero, so equality at `rc` alone could fail to detect a missed candidate pair. A pair just inside the cutoff (`r = rc - ε`) must be found by cells (it contributes nonzero energy/force) and agree with naive.

### Comparison tolerance [Suggestion]

The naive/cells comparison tolerance is **not** a course requirement; the plan must choose and fix a scale-aware rounding criterion before implementation. Proposed criterion (component-wise / scalar):

```
|a − b| <= atol + rtol * max(|a|, |b|)   with  atol = 1e-10,  rtol = 1e-9
```

This is scale-aware (allowed error grows with the compared magnitudes) and has a small absolute floor for near-zero components (cancellation). It is generous enough to absorb FP accumulation-order differences while still catching a genuinely missed pair (an O(1) or larger discrepancy).

## 11. Risks / edge cases

- **`nx = 0` for `L < rc`**: divide-by-zero; see §4 small-box guard ([Suggestion]).
- **Cell-width invariant** `wx, wy >= rc` is guaranteed by construction; must not be broken by any guard.
- **Unwrapped / out-of-box positions** in binning: contract is wrapped positions; clamp the upper index defensively. Not a course requirement.
- **Overlapping atoms** (`r = 0` → `shifted_force(0)/0 = NaN`): pre-existing in naive (YAGNI), unchanged in cells.
- **Accumulation order** differs between naive and cells → tolerance-based comparison, never bitwise equality.
- **Determinism**: cells enumeration is deterministic (fixed cell order).
- **Default-flip regression**: after flipping `SimConfig::default()` to `Cells`, re-run the full suite + contract to confirm bounds are unchanged (they are, to rounding).
- **Performance**: verify cells beats naive on the N=400 profile case; for tiny N (n=16) cells may not win on wall-clock but must still be correct.

## 12. Requirement labeling summary

**Course Requirement** (as stated by the course/spec):
- `--force naive` keeps the original implementation available.
- `--force cells` provides the cell-list implementation.
- `--force cells` is the default after the optimization is complete (final contract).
- `rc = 2.5`; `nx = floor(Lx/rc)`, `wx = Lx/nx`; `ny = floor(Ly/rc)`, `wy = Ly/ny`; cell width ≥ rc.
- 3×3 neighborhood search (own cell + eight neighbors) per particle/cell.
- Neighbor indices wrap periodically.
- Wrapped neighbor indices deduplicated.
- Pairs evaluated only once.
- Correctness coverage: naive/cells force equality; naive/cells energy equality; perturbed lattices; pairs across a periodic boundary; pairs exactly at the cutoff; two-cell-wide box with duplicate wrapped neighbors.
- Preserve all Part 2–4 behavior and existing tests.

**[Suggestion]** (our choices, not course-specified):
- Small-box (`L < rc`) guard `max(1, ...)` — outside the stated course rule.
- Naive/cells comparison tolerance and the scale-aware criterion in §10.
- The `r = rc − ε` test case.
- Binning implementation details (`Vec<Vec<usize>>`, clamp upper index, etc.).

## Non-goals

- No Verlet neighbor lists or more advanced optimization.
- No run.json schema change.
- No change to simulation sampling, physics bounds, checker/renderer/video, or the Part 4 artifact contract.
