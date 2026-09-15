# Week 2 Part 5: Cell-List Force Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cell-list (`cells`) force/energy path to `week2/md` alongside the existing naive path, wire a `--force naive|cells` switch, make `cells` the default, and verify correctness, profile, and scaling — without changing any Part 2–4 behavior.

**Architecture:** All cell-list physics lives in `fluid.rs`. A small `ForceMethod` enum (`Naive`/`Cells`) selects the path; `SimConfig` carries it into `run_simulation`; the CLI exposes `--force naive|cells`. Naive stays the reference path used by existing tests and the checker. Both paths share one minimum-image / shifted-LJ pair-physics helper. No `force_method` field is added to run.json.

**Tech Stack:** Rust 2021, existing crate only. Cell bins are rebuilt every evaluation (no incremental updates). No Verlet lists, no parallelism, no SIMD, no heating ramp.

**Spec:** `docs/superpowers/specs/2026-09-15-week2-md-cell-list-design.md`

## Global Constraints

- Run all cargo commands from `week2/` with `--manifest-path md/Cargo.toml` (equivalently from the repo root with `--manifest-path week2/md/Cargo.toml`).
- Do NOT modify `src/lib.rs`'s existing public functions; only extend the `pub use fluid::{...}` line (and later `simulate`/`cli` as specified). Preserve all Part 2–4 behavior and existing tests.
- Preserve the public naive reference functions `fluid_accelerations(state, bx)` and `fluid_potential_energy(state, bx)` unchanged; existing Part 2–4 tests keep using them.
- Final default contract [Course Requirement]: `--force cells` is the default; `SimConfig::default()` uses `ForceMethod::Cells`; `--force naive` remains available.
- `ForceMethod` carries **no** serde derives and is **not** written to run.json. The run.json key set stays exactly `n, rho, box, dt, temperature, eq_steps, steps, sample_every, seed, integrator`.
- Course geometry rule: `nx = floor(Lx / rc)`, `ny = floor(Ly / rc)`, `wx = Lx / nx`, `wy = Ly / ny`, cell width ≥ rc, 3×3 wrapped + deduplicated neighbor search, pairs evaluated once.
- Naive/cells comparison tolerance is [Suggestion] and is FIXED in Task 1 before implementation: `|a − b| <= 1e-10 + 1e-9 * max(|a|, |b|)`. Do not loosen it later to make cells pass.
- The checker recomputes through the naive reference path and stays method-independent.
- Post-optimization profiling/scaling numbers are measured only; never invent expected timings or speedups.
- Do not introduce Verlet neighbor lists, incremental cell updates, parallelism, SIMD-specific optimization, or the Part 5 heating ramp.

## Task Map (dependency order)

1. First RED — naive-vs-cells equality tests (tests/cells.rs)
2. Shared pair physics + naive refactor
3. `ForceMethod` enum (no serde) + export
4. Cell geometry + position→cell mapping
5. Wrapped + deduplicated 3×3 neighbor cells
6. Cell membership (binning)
7. Cells pair enumeration + accelerations + potential energy
8. Dispatch through `ForceMethod` → tests/cells.rs GREEN
9. `SimConfig` propagation (+ update all `SimConfig { .. }` literals)
10. CLI `--force naive|cells`
11. Final default flip to `Cells`
12. Correctness VERIFY (release suite + cells-run `md check`)
13. Profile `cells` (N=400) + `profile-cells.png` + compare to naive evidence
14. Scaling benchmarks N = 100, 400, 1600 (both methods, measured wall-clock)

---

### Task 1: First RED — naive-vs-cells equality tests (tests/cells.rs)

**Files:**
- Create: `week2/md/tests/cells.rs`

**Interfaces:**
- Consumes: `md::fluid::{fluid_accelerations, fluid_potential_energy}` (naive, exist), `md::system::{Box2, lattice_state, minimum_image, wrap}`, `md::State`.
- Produces: the equality/coverage test suite that later tasks turn green. It references `md::fluid::ForceMethod::Cells.accelerations(...)` and `ForceMethod::Cells.potential_energy(...)` — the API built in Tasks 3 and 8. It does NOT exist yet, which is the RED.

- [ ] **Step 1: Write the failing test file**

`week2/md/tests/cells.rs`:

```rust
//! Cell-list vs naive equality and coverage tests.
//!
//! Course Requirement: the cells implementation must reproduce the naive
//! reference to rounding on every required configuration.

use md::fluid::{fluid_accelerations, fluid_potential_energy, ForceMethod};
use md::system::{Box2, lattice_state, minimum_image, wrap};
use md::State;

/// Scale-aware rounding comparison [Suggestion]; fixed before implementation.
const ATOL: f64 = 1e-10;
const RTOL: f64 = 1e-9;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs())
}

fn assert_forces_close(naive: &[[f64; 2]], cells: &[[f64; 2]]) {
    assert_eq!(naive.len(), cells.len());
    for k in 0..naive.len() {
        assert!(close(naive[k][0], cells[k][0]),
            "atom {k} Fx: naive {} vs cells {}", naive[k][0], cells[k][0]);
        assert!(close(naive[k][1], cells[k][1]),
            "atom {k} Fy: naive {} vs cells {}", naive[k][1], cells[k][1]);
    }
}

fn assert_total_force_zero(acc: &[[f64; 2]]) {
    let mut s = [0.0, 0.0];
    for a in acc {
        s[0] += a[0];
        s[1] += a[1];
    }
    assert!(s[0].abs() < 1e-9, "sum Fx = {}", s[0]);
    assert!(s[1].abs() < 1e-9, "sum Fy = {}", s[1]);
}

fn assert_energy_close(naive: f64, cells: f64) {
    assert!(close(naive, cells), "energy: naive {naive} vs cells {cells}");
}

#[test]
fn forces_and_energy_agree_on_lattice() {
    let state = lattice_state(100, 0.8);
    let bx = Box2::new(100, 0.8);
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_total_force_zero(&cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_on_perturbed_configuration() {
    let mut state = lattice_state(100, 0.8);
    let bx = Box2::new(100, 0.8);
    for (i, p) in state.positions.iter_mut().enumerate() {
        if i % 2 == 0 {
            p[0] = wrap(p[0] + 0.05 * (i as f64 % 3.0), bx.lx);
            p[1] = wrap(p[1] + 0.11, bx.ly);
        }
    }
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_total_force_zero(&cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_across_periodic_boundary() {
    // Pairs straddling x = 0/Lx and y = 0/Ly with minimum-image r < rc.
    let bx = Box2 { lx: 10.0, ly: 10.0 };
    let state = State {
        positions: vec![
            [0.3, 5.0], [9.7, 5.0], [5.0, 0.3], [5.0, 9.7], [2.0, 2.0], [8.0, 8.0],
        ],
        velocities: vec![[0.0, 0.0]; 6],
    };
    // Sanity: the x-straddling pair really is within rc via minimum image.
    let dx = minimum_image(0.3 - 9.7, bx.lx);
    assert!(dx.abs() < 2.5, "boundary pair must interact, dx = {dx}");
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_at_exact_cutoff() {
    // A pair at exactly r = rc (zero contribution) plus a sub-rc pair, so the
    // test is not trivially all-zero.
    let bx = Box2 { lx: 10.0, ly: 10.0 };
    let state = State {
        positions: vec![[1.0, 5.0], [3.5, 5.0], [5.0, 1.0], [5.0, 8.0], [1.0, 3.0]],
        velocities: vec![[0.0, 0.0]; 5],
    };
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_at_cutoff_minus_epsilon() {
    // [Suggestion] A pair just inside the cutoff must actually be found by
    // cells (nonzero contribution) and agree with naive.
    let eps = 1e-6;
    let bx = Box2 { lx: 10.0, ly: 10.0 };
    let state = State {
        positions: vec![[1.0, 5.0], [1.0 + 2.5 - eps, 5.0], [5.0, 1.0], [5.0, 8.0]],
        velocities: vec![[0.0, 0.0]; 4],
    };
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}

#[test]
fn agree_in_two_cell_wide_box_with_dedup() {
    // nx = ny = 2: wrapped 3x3 neighbor indices duplicate and must be
    // deduplicated or pairs would be double-counted.
    let bx = Box2 { lx: 5.0, ly: 5.0 };
    let mut pos = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            pos.push([i as f64 + 0.6, j as f64 + 0.6]);
        }
    }
    let state = State {
        positions: pos,
        velocities: vec![[0.0, 0.0]; 16],
    };
    let naive = fluid_accelerations(&state, &bx);
    let cells = ForceMethod::Cells.accelerations(&state, &bx);
    assert_forces_close(&naive, &cells);
    assert_total_force_zero(&cells);
    assert_energy_close(
        fluid_potential_energy(&state, &bx),
        ForceMethod::Cells.potential_energy(&state, &bx),
    );
}
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --test cells`
Expected: compile FAIL — `error[E0432]: unresolved import md::fluid::ForceMethod` (neither `ForceMethod` nor `ForceMethod::Cells` exists yet).

- [ ] **Step 3: Commit the failing tests**

```bash
git add week2/md/tests/cells.rs
git commit -m "test: add failing naive-vs-cells equality and coverage tests"
```

---

### Task 2: Shared pair physics + naive refactor

**Files:**
- Modify: `week2/md/src/fluid.rs` (add `pair_force`, `pair_energy`; refactor the two naive public functions to use them; add a `#[cfg(test)] mod tests`).

**Interfaces:**
- Consumes: `crate::pair::{shifted_energy, shifted_force}`, `crate::system::{Box2, minimum_image}` (all exist).
- Produces (private to `fluid.rs`, shared by naive and cells):
  - `fn pair_force(dx_raw: f64, dy_raw: f64, bx: &Box2) -> (f64, f64)`
  - `fn pair_energy(dx_raw: f64, dy_raw: f64, bx: &Box2) -> f64`

- [ ] **Step 1: Write the failing unit test** (append to a new `#[cfg(test)] mod tests` in `fluid.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_helpers_use_minimum_image_and_shifted_lj() {
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        // raw dx = 9.5 wraps to -0.5 via minimum image (r = 0.5).
        let (fx, fy) = pair_force(9.5, 0.0, &bx);
        let f_over_r = crate::pair::shifted_force(0.5) / 0.5;
        assert!((fx - (f_over_r * -0.5)).abs() < 1e-15);
        assert!((fy - 0.0).abs() < 1e-15);
        assert!((pair_energy(9.5, 0.0, &bx) - crate::pair::shifted_energy(0.5)).abs() < 1e-15);
    }
}
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib pair_helpers`
Expected: compile FAIL — `error[E0425]: cannot find function pair_force` (and `pair_energy`).

- [ ] **Step 3: Implement the shared helpers and refactor the naive functions**

Replace the bodies of `fluid_accelerations` and `fluid_potential_energy` in `fluid.rs` with calls to the helpers (identical math, same accumulation order for the outer i<j loops):

```rust
/// Single source of pair physics: minimum-image displacement + shifted force.
fn pair_force(dx_raw: f64, dy_raw: f64, bx: &Box2) -> (f64, f64) {
    let dx = minimum_image(dx_raw, bx.lx);
    let dy = minimum_image(dy_raw, bx.ly);
    let r = (dx * dx + dy * dy).sqrt();
    let f_over_r = shifted_force(r) / r;
    (f_over_r * dx, f_over_r * dy)
}

/// Single source of pair physics: minimum-image displacement + shifted energy.
fn pair_energy(dx_raw: f64, dy_raw: f64, bx: &Box2) -> f64 {
    let dx = minimum_image(dx_raw, bx.lx);
    let dy = minimum_image(dy_raw, bx.ly);
    shifted_energy((dx * dx + dy * dy).sqrt())
}
```

`fluid_accelerations` becomes:

```rust
pub fn fluid_accelerations(state: &State, bx: &Box2) -> Vec<Vec2> {
    let n = state.positions.len();
    let mut acc = vec![[0.0, 0.0]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (fx, fy) = pair_force(
                state.positions[i][0] - state.positions[j][0],
                state.positions[i][1] - state.positions[j][1],
                bx,
            );
            acc[i][0] += fx;
            acc[i][1] += fy;
            acc[j][0] -= fx;
            acc[j][1] -= fy;
        }
    }
    acc
}
```

`fluid_potential_energy` becomes:

```rust
pub fn fluid_potential_energy(state: &State, bx: &Box2) -> f64 {
    let mut energy = 0.0;
    for i in 0..state.positions.len() {
        for j in (i + 1)..state.positions.len() {
            energy += pair_energy(
                state.positions[i][0] - state.positions[j][0],
                state.positions[i][1] - state.positions[j][1],
                bx,
            );
        }
    }
    energy
}
```

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib pair_helpers`
Expected: PASS.

- [ ] **Step 5: Regression — naive path unchanged**

Run: `cargo test --manifest-path md/Cargo.toml --lib --test lj --test dimer --test pair --test fluid`
Expected: all PASS (the refactored naive path is bit-identical in math; existing tests prove behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/fluid.rs
git commit -m "refactor: share pair physics between force and energy paths"
```

---

### Task 3: `ForceMethod` enum (no serde) + export

**Files:**
- Modify: `week2/md/src/fluid.rs` (add the enum), `week2/md/src/lib.rs` (export it).

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `pub enum ForceMethod { Naive, Cells }` (Clone, Copy, Debug, PartialEq, Eq, `clap::ValueEnum`, `Display`). No serde.
  - Re-exported as `md::fluid::ForceMethod` (and `md::ForceMethod`) so `tests/cells.rs` resolves.
  - The dispatch methods `.accelerations`/`.potential_energy` come in Task 8; do NOT add them here.

- [ ] **Step 1: Write the failing unit test** (append to the `fluid.rs` tests module):

```rust
    #[test]
    fn force_method_displays_and_supports_clap_values() {
        assert_eq!(format!("{}", ForceMethod::Naive), "naive");
        assert_eq!(format!("{}", ForceMethod::Cells), "cells");
        // ValueEnum possible values (used by --force) are the kebab names.
        let names: Vec<String> = ForceMethod::value_variants()
            .iter()
            .map(|m| m.to_possible_value().unwrap().get_name().to_string())
            .collect();
        assert_eq!(names, vec!["naive", "cells"]);
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib force_method`
Expected: compile FAIL — `error[E0425]: cannot find enum ForceMethod in this scope`.

- [ ] **Step 3: Implement the enum**

In `fluid.rs` (top of file, after the imports):

```rust
/// Force evaluation method: the naive O(N^2) reference or the cell list.
/// A pure runtime/performance selection; never serialized to run.json.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ForceMethod {
    Naive,
    Cells,
}

impl std::fmt::Display for ForceMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ForceMethod::Naive => "naive",
                ForceMethod::Cells => "cells",
            }
        )
    }
}
```

In `lib.rs`, change the fluid re-export to:

```rust
pub use fluid::{ForceMethod, fluid_accelerations, fluid_potential_energy};
```

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib force_method`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib`
Expected: PASS. Note: `cargo test --manifest-path md/Cargo.toml --test cells` is now RED in a NEW way — `error[E0599]: no method named 'accelerations' found for enum ForceMethod` — this is the expected staged compile-RED until Task 8. All other test targets compile and pass.

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/fluid.rs week2/md/src/lib.rs
git commit -m "feat: add ForceMethod runtime-selection enum"
```

---

### Task 4: Cell geometry + position→cell mapping

**Files:**
- Modify: `week2/md/src/fluid.rs`.

**Interfaces:**
- Consumes: `crate::pair::RC`.
- Produces (private):
  - `fn cell_geometry(bx: &Box2) -> (usize, usize, f64, f64)` → `(nx, ny, wx, wy)`
  - `fn cell_axis_index(coord: f64, w: f64, n: usize) -> usize`
  - `fn cell_index(x: f64, y: f64, wx: f64, wy: f64, nx: usize, ny: usize) -> usize`

- [ ] **Step 1: Write the failing unit tests** (append to `fluid.rs` tests):

```rust
    #[test]
    fn cell_geometry_matches_course_rule() {
        // Default contract: Lx ~ 12.014, Ly ~ 10.404.
        let bx = Box2::new(100, 0.8);
        let (nx, ny, wx, wy) = cell_geometry(&bx);
        assert_eq!((nx, ny), (4, 4));
        assert!(wx >= crate::pair::RC && wy >= crate::pair::RC);
        // Profile-sized box (N = 400): Lx ~ 24.028, Ly ~ 20.809.
        let bx400 = Box2::new(400, 0.8);
        let (nx400, ny400, _, _) = cell_geometry(&bx400);
        assert_eq!((nx400, ny400), (9, 8));
        // Two-cell-wide box (nx = ny = 2, w = 2.5).
        let small = Box2 { lx: 5.0, ly: 5.0 };
        let (n2x, n2y, w2x, w2y) = cell_geometry(&small);
        assert_eq!((n2x, n2y), (2, 2));
        assert!((w2x - 2.5).abs() < 1e-12 && (w2y - 2.5).abs() < 1e-12);
    }

    #[test]
    fn cell_index_maps_wrapped_positions() {
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let (nx, ny, wx, wy) = cell_geometry(&bx); // 4x4, w = 2.5
        assert_eq!(cell_index(0.0, 0.0, wx, wy, nx, ny), 0);
        assert_eq!(cell_index(2.4, 2.4, wx, wy, nx, ny), 0);
        assert_eq!(cell_index(2.5, 0.0, wx, wy, nx, ny), 1);
        assert_eq!(cell_index(0.0, 2.5, wx, wy, nx, ny), nx);
        assert_eq!(cell_index(9.99, 9.99, wx, wy, nx, ny), nx * ny - 1);
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib cell_geometry`
Expected: compile FAIL — `error[E0425]: cannot find function cell_geometry`.

- [ ] **Step 3: Implement**

```rust
/// Rectangular cell grid: nx = floor(Lx/rc), wx = Lx/nx, and similarly for y,
/// so wx, wy >= rc [Course Requirement]. The max(1, ...) guard handles a
/// sub-rc box and is [Suggestion] (outside the stated course rule).
fn cell_geometry(bx: &Box2) -> (usize, usize, f64, f64) {
    let nx = ((bx.lx / crate::pair::RC).floor() as usize).max(1);
    let ny = ((bx.ly / crate::pair::RC).floor() as usize).max(1);
    (nx, ny, bx.lx / nx as f64, bx.ly / ny as f64)
}

/// Axis cell index for a coordinate already wrapped into [0, n*w).
fn cell_axis_index(coord: f64, w: f64, n: usize) -> usize {
    ((coord / w).floor() as usize).min(n - 1)
}

/// Flat cell index c = cy * nx + cx for a wrapped position.
fn cell_index(x: f64, y: f64, wx: f64, wy: f64, nx: usize, ny: usize) -> usize {
    cell_axis_index(y, wy, ny) * nx + cell_axis_index(x, wx, nx)
}
```

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib cell_geometry`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/fluid.rs
git commit -m "feat: add cell grid geometry and position-to-cell mapping"
```

---

### Task 5: Wrapped + deduplicated 3×3 neighbor cells

**Files:**
- Modify: `week2/md/src/fluid.rs`.

**Interfaces:**
- Consumes: nothing new.
- Produces (private): `fn neighbor_cells(c: usize, nx: usize, ny: usize) -> Vec<usize>`

- [ ] **Step 1: Write the failing unit tests** (append to `fluid.rs` tests):

```rust
    #[test]
    fn neighbor_cells_wraps_periodically_and_deduplicates() {
        // 4x4 grid: cell 0 (cx=0, cy=0) wraps onto the far column/row.
        let n = neighbor_cells(0, 4, 4);
        assert_eq!(n.len(), 9);
        for d in [0usize, 3, 12, 15] {
            assert!(n.contains(&d), "cell {d} must be a wrapped neighbor of 0");
        }
        assert!(!n.contains(&5), "(1,1) is not in the 3x3 of (0,0)");
        // no duplicates
        let mut s = n.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), n.len());
    }

    #[test]
    fn neighbor_cells_deduplicates_in_two_cell_box() {
        // nx = ny = 2: the 3x3 window wraps onto only the 4 distinct cells.
        for c in 0..4 {
            let n = neighbor_cells(c, 2, 2);
            assert_eq!(n.len(), 4, "cell {c} must yield exactly 4 distinct cells");
            for d in 0..4 {
                assert!(n.contains(&d));
            }
        }
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib neighbor_cells`
Expected: compile FAIL — `error[E0425]: cannot find function neighbor_cells`.

- [ ] **Step 3: Implement**

```rust
/// Wrapped + deduplicated 3x3 neighbor cell indices of `c` [Course Requirement].
fn neighbor_cells(c: usize, nx: usize, ny: usize) -> Vec<usize> {
    let cx = c % nx;
    let cy = c / nx;
    let mut out = Vec::with_capacity(9);
    for dy in [-1isize, 0, 1] {
        for dx in [-1isize, 0, 1] {
            let wcx = ((cx as isize + dx).rem_euclid(nx as isize)) as usize;
            let wcy = ((cy as isize + dy).rem_euclid(ny as isize)) as usize;
            let idx = wcy * nx + wcx;
            if !out.contains(&idx) {
                out.push(idx);
            }
        }
    }
    out
}
```

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib neighbor_cells`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/fluid.rs
git commit -m "feat: add wrapped, deduplicated 3x3 neighbor-cell enumeration"
```

---

### Task 6: Cell membership (binning)

**Files:**
- Modify: `week2/md/src/fluid.rs`.

**Interfaces:**
- Consumes: `cell_index` (Task 4).
- Produces (private): `fn build_cells(positions: &[Vec2], nx: usize, ny: usize, wx: f64, wy: f64) -> Vec<Vec<usize>>`

- [ ] **Step 1: Write the failing unit test** (append to `fluid.rs` tests):

```rust
    #[test]
    fn build_cells_bins_every_particle_once() {
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let (nx, ny, wx, wy) = cell_geometry(&bx);
        let pos: Vec<Vec2> = vec![
            [0.1, 0.1], [2.6, 0.2], [7.7, 7.8], [9.9, 9.9], [5.0, 5.0],
        ];
        let cells = build_cells(&pos, nx, ny, wx, wy);
        assert_eq!(cells.iter().map(|c| c.len()).sum::<usize>(), pos.len());
        assert!(cells[cell_index(0.1, 0.1, wx, wy, nx, ny)].contains(&0));
        assert!(cells[cell_index(2.6, 0.2, wx, wy, nx, ny)].contains(&1));
        assert!(cells[cell_index(7.7, 7.8, wx, wy, nx, ny)].contains(&2));
        assert!(cells[cell_index(9.9, 9.9, wx, wy, nx, ny)].contains(&3));
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib build_cells`
Expected: compile FAIL — `error[E0425]: cannot find function build_cells`.

- [ ] **Step 3: Implement**

```rust
/// Particle-index lists per cell, rebuilt on every evaluation [design].
fn build_cells(positions: &[Vec2], nx: usize, ny: usize, wx: f64, wy: f64) -> Vec<Vec<usize>> {
    let mut cells = vec![Vec::new(); nx * ny];
    for (i, p) in positions.iter().enumerate() {
        cells[cell_index(p[0], p[1], wx, wy, nx, ny)].push(i);
    }
    cells
}
```

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib build_cells`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/fluid.rs
git commit -m "feat: add per-cell particle binning"
```

---

### Task 7: Cells pair enumeration + accelerations + potential energy

**Files:**
- Modify: `week2/md/src/fluid.rs`.

**Interfaces:**
- Consumes: `pair_force`/`pair_energy` (Task 2), `cell_geometry`/`cell_index` (Task 4), `neighbor_cells` (Task 5), `build_cells` (Task 6).
- Produces (private):
  - `fn for_each_pair_cells<F: FnMut(usize, usize, f64, f64)>(positions: &[Vec2], bx: &Box2, f: F)`
  - `fn accelerations_cells(state: &State, bx: &Box2) -> Vec<Vec2>`
  - `fn potential_energy_cells(state: &State, bx: &Box2) -> f64`

- [ ] **Step 1: Write the failing unit tests** (append to `fluid.rs` tests):

```rust
    #[test]
    fn for_each_pair_cells_counts_each_pair_once() {
        // In the two-cell box the wrapped neighbor indices duplicate; the
        // per-particle 3x3 with j > i must still visit each unordered pair once.
        let bx = Box2 { lx: 5.0, ly: 5.0 };
        let pos: Vec<Vec2> = (0..8)
            .map(|i| [0.7 + (i as f64 % 4.0), 0.7 + (i as f64 / 4.0)])
            .collect();
        let mut count = 0usize;
        for_each_pair_cells(&pos, &bx, |_i, _j, _dx, _dy| count += 1);
        assert_eq!(count, pos.len() * (pos.len() - 1) / 2);
    }

    #[test]
    fn cells_forces_sum_to_zero() {
        let bx = Box2 { lx: 5.0, ly: 5.0 };
        let state = State {
            positions: vec![[0.7, 0.7], [1.7, 1.7], [3.3, 3.3], [4.4, 4.4], [0.7, 4.4], [4.4, 0.7]],
            velocities: vec![[0.0, 0.0]; 6],
        };
        let acc = accelerations_cells(&state, &bx);
        let mut s = [0.0, 0.0];
        for a in &acc {
            s[0] += a[0];
            s[1] += a[1];
        }
        assert!(s[0].abs() < 1e-9 && s[1].abs() < 1e-9);
    }

    #[test]
    fn cells_spot_check_matches_naive() {
        // Quick spot check; the authoritative suite is tests/cells.rs (Task 8).
        let mut state = crate::system::lattice_state(16, 0.8);
        let bx = Box2::new(16, 0.8);
        for (i, p) in state.positions.iter_mut().enumerate() {
            if i % 3 == 0 {
                p[0] += 0.13;
            }
        }
        let na = fluid_accelerations(&state, &bx);
        let cl = accelerations_cells(&state, &bx);
        for k in 0..na.len() {
            assert!((na[k][0] - cl[k][0]).abs() < 1e-9 && (na[k][1] - cl[k][1]).abs() < 1e-9);
        }
        assert!((fluid_potential_energy(&state, &bx) - potential_energy_cells(&state, &bx)).abs() < 1e-9);
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib for_each_pair_cells`
Expected: compile FAIL — `error[E0425]: cannot find function for_each_pair_cells`.

- [ ] **Step 3: Implement**

```rust
/// Visit each unordered pair exactly once via the cell list [Course Req]:
/// per-particle 3x3 (wrapped + deduplicated) neighborhood with a j > i guard.
fn for_each_pair_cells<F: FnMut(usize, usize, f64, f64)>(
    positions: &[Vec2],
    bx: &Box2,
    mut f: F,
) {
    let (nx, ny, wx, wy) = cell_geometry(bx);
    let cells = build_cells(positions, nx, ny, wx, wy);
    for i in 0..positions.len() {
        let c = cell_index(positions[i][0], positions[i][1], wx, wy, nx, ny);
        for &n in &neighbor_cells(c, nx, ny) {
            for &j in &cells[n] {
                if j > i {
                    f(i, j, positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]);
                }
            }
        }
    }
}

/// Cell-list accelerations (mass 1); antisymmetric +f/-f per pair.
fn accelerations_cells(state: &State, bx: &Box2) -> Vec<Vec2> {
    let mut acc = vec![[0.0, 0.0]; state.positions.len()];
    for_each_pair_cells(&state.positions, bx, |i, j, dx, dy| {
        let (fx, fy) = pair_force(dx, dy, bx);
        acc[i][0] += fx;
        acc[i][1] += fy;
        acc[j][0] -= fx;
        acc[j][1] -= fy;
    });
    acc
}

/// Cell-list shifted potential energy over the same unordered pair set.
fn potential_energy_cells(state: &State, bx: &Box2) -> f64 {
    let mut energy = 0.0;
    for_each_pair_cells(&state.positions, bx, |_i, _j, dx, dy| {
        energy += pair_energy(dx, dy, bx);
    });
    energy
}
```

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib for_each_pair_cells`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib`
Expected: PASS. (`tests/cells.rs` is still the staged compile-RED — no `.accelerations` method yet.)

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/fluid.rs
git commit -m "feat: add cell-list accelerations and potential energy"
```

---

### Task 8: Dispatch through `ForceMethod` → tests/cells.rs GREEN

**Files:**
- Modify: `week2/md/src/fluid.rs`.

**Interfaces:**
- Consumes: `ForceMethod` (Task 3), `accelerations_cells`/`potential_energy_cells` (Task 7), naive references.
- Produces (public): `ForceMethod::accelerations(self, state: &State, bx: &Box2) -> Vec<Vec2>` and `ForceMethod::potential_energy(self, state: &State, bx: &Box2) -> f64`.

- [ ] **Step 1: Write the failing unit test** (append to `fluid.rs` tests):

```rust
    #[test]
    fn force_method_dispatch_matches_references() {
        let state = crate::system::lattice_state(36, 0.8);
        let bx = Box2::new(36, 0.8);
        assert_eq!(
            ForceMethod::Naive.accelerations(&state, &bx),
            fluid_accelerations(&state, &bx)
        );
        let cells = ForceMethod::Cells.accelerations(&state, &bx);
        assert_eq!(cells.len(), 36);
        assert!((ForceMethod::Naive.potential_energy(&state, &bx)
            - fluid_potential_energy(&state, &bx))
            .abs() < 1e-12);
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib force_method_dispatch`
Expected: compile FAIL — `error[E0599]: no method named 'accelerations' found for enum ForceMethod`.

- [ ] **Step 3: Implement the dispatch**

```rust
impl ForceMethod {
    /// Compute accelerations with the selected method (m = 1, a = F).
    pub fn accelerations(self, state: &State, bx: &Box2) -> Vec<Vec2> {
        match self {
            ForceMethod::Naive => fluid_accelerations(state, bx),
            ForceMethod::Cells => accelerations_cells(state, bx),
        }
    }

    /// Compute the shifted potential energy with the selected method.
    pub fn potential_energy(self, state: &State, bx: &Box2) -> f64 {
        match self {
            ForceMethod::Naive => fluid_potential_energy(state, bx),
            ForceMethod::Cells => potential_energy_cells(state, bx),
        }
    }
}
```

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test cells`
Expected: all 6 tests PASS (lattice, perturbed, boundary, exact cutoff, cutoff−ε, two-cell dedup).

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib --test lj --test dimer --test pair --test fluid`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/fluid.rs
git commit -m "feat: dispatch naive/cells through ForceMethod"
```

---

### Task 9: `SimConfig` propagation

**Files:**
- Modify: `week2/md/src/simulate.rs` (struct, Default, `run_simulation`; `small_config` test helper), `week2/md/src/cli.rs` (run_command literal), `week2/md/src/video.rs` (test literal), `week2/md/tests/checker.rs` (four `SimConfig { .. }` literals).

**Interfaces:**
- Consumes: `ForceMethod` (Task 3) and its dispatch (Task 8).
- Produces: `SimConfig { ..., pub force_method: ForceMethod }`. `run_simulation` dispatches via `config.force_method`. run.json is unchanged (`RunConfig::from(&config)` ignores `force_method`).

- [ ] **Step 1: Write the failing unit test** (append to the `simulate.rs` tests module):

```rust
    #[test]
    fn run_simulation_works_with_both_force_methods() {
        for m in [ForceMethod::Naive, ForceMethod::Cells] {
            let mut c = small_config();
            c.force_method = m;
            let frames = run_simulation(&c);
            assert_eq!(frames.len(), 8);
            assert!(frames.iter().all(|f| f.e_pot.is_finite() && f.e_kin.is_finite()));
        }
    }
```

Also update the existing `defaults_are_the_course_contract_values` test to assert the staged default:

```rust
        assert_eq!(c.force_method, ForceMethod::Naive); // staging; flipped in Task 11
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib run_simulation_works_with_both_force_methods`
Expected: compile FAIL — `error[E0609]: no field 'force_method' on type 'SimConfig'`.

- [ ] **Step 3: Implement**

In `simulate.rs`:

```rust
use crate::fluid::{ForceMethod, fluid_accelerations, fluid_potential_energy};
```

Add the field and the staged default:

```rust
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
    pub force_method: ForceMethod,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
            force_method: ForceMethod::Naive, // staging; flipped to Cells in Task 11
        }
    }
}
```

Dispatch in `run_simulation` — replace the acceleration closure and the saved `e_pot`:

```rust
    let accelerations = |s: &State| config.force_method.accelerations(s, &bx);
```

and

```rust
                e_pot: config.force_method.potential_energy(&state, &bx),
```

Update every `SimConfig { .. }` literal to add `force_method: ForceMethod::Naive`:
- `week2/md/src/simulate.rs` `small_config()` (around line 113).
- `week2/md/src/cli.rs` `run_command` (around line 117) — `force_method: ForceMethod::Naive` (replaced by `args.force` in Task 10).
- `week2/md/src/video.rs` `encode_video_of_tiny_run_produces_small_mp4_when_ffmpeg_present` test literal (around line 119).
- `week2/md/tests/checker.rs` literals at lines ~51, ~73, ~91, ~189.

(Each of those files needs `use crate::fluid::ForceMethod;` or `use md::fluid::ForceMethod;` added as appropriate.)

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib run_simulation_works_with_both_force_methods`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --lib --test lj --test dimer --test pair --test fluid --test checker --test cli`
Expected: all PASS (existing `SimConfig` literals now carry the staged `Naive`, so behavior is unchanged; run.json has no `force_method` — `io::tests::run_json_keys_are_exactly_the_course_contract` stays green).

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/simulate.rs week2/md/src/cli.rs week2/md/src/video.rs week2/md/tests/checker.rs
git commit -m "feat: propagate ForceMethod through SimConfig and run_simulation"
```

---

### Task 10: CLI `--force naive|cells`

**Files:**
- Modify: `week2/md/src/cli.rs` (RunArgs + run_command), `week2/md/tests/cli.rs` (new tests).

**Interfaces:**
- Consumes: `ForceMethod` (`clap::ValueEnum` + `Display`, Task 3).
- Produces: `RunArgs { ..., pub force: ForceMethod }`; `md run --force naive|cells`. run.json key set unchanged (no `force_method`).

- [ ] **Step 1: Write the failing tests** (append to `week2/md/tests/cli.rs`):

```rust
#[test]
fn run_accepts_both_force_methods_and_keeps_run_json_schema() {
    for m in ["naive", "cells"] {
        let out = temp_dir(&format!("force-{m}"));
        let output = run_md(&["run", "--n", "16", "--eq-steps", "0", "--steps", "100",
                              "--sample-every", "50", "--force", m, "--out", out.to_str().unwrap()]);
        assert!(output.status.success(), "{m} run failed: {}", String::from_utf8_lossy(&output.stderr));
        let traj = std::fs::read_to_string(out.join("traj.jsonl")).unwrap();
        assert_eq!(traj.lines().filter(|l| !l.trim().is_empty()).count(), 2);
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
        // run.json schema is unchanged: no force_method key.
        assert!(run.get("force_method").is_none(), "run.json must not contain force_method");
        let _ = std::fs::remove_dir_all(&out);
    }
}

#[test]
fn run_rejects_unknown_force_method() {
    let out = temp_dir("force-bad");
    let output = run_md(&["run", "--n", "16", "--force", "bogus", "--out", out.to_str().unwrap()]);
    assert!(!output.status.success());
    let _ = std::fs::remove_dir_all(&out);
}
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --test cli run_accepts_both_force_methods`
Expected: FAIL — clap reports `unexpected argument '--force' found` and the subprocess exits non-zero.

- [ ] **Step 3: Implement**

In `cli.rs`, add to `RunArgs`:

```rust
    /// Force evaluation method: naive O(N^2) reference or the cell list.
    #[arg(long, value_enum, default_value_t = ForceMethod::Naive)]
    pub force: ForceMethod,
```

and set the config in `run_command`:

```rust
        force_method: args.force,
```

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --test cli run_accepts_both_force_methods`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cargo test --manifest-path md/Cargo.toml --test cli`
Expected: all cli tests PASS (now 6).

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/cli.rs week2/md/tests/cli.rs
git commit -m "feat: add --force naive|cells CLI switch"
```

---

### Task 11: Final default flip to `Cells`

**Files:**
- Modify: `week2/md/src/simulate.rs` (Default), `week2/md/src/cli.rs` (RunArgs default), `week2/md/src/simulate.rs` test assertion.

**Interfaces:**
- Consumes: nothing new.
- Produces: `SimConfig::default()` and `--force` default both `ForceMethod::Cells` — the final contract.

- [ ] **Step 1: Write the failing test change**

Update `defaults_are_the_course_contract_values` in `simulate.rs` tests:

```rust
        assert_eq!(c.force_method, ForceMethod::Cells); // final contract default
```

- [ ] **Step 2: Run to verify RED**

Run: `cargo test --manifest-path md/Cargo.toml --lib defaults_are_the_course_contract_values`
Expected: FAIL — `left: Naive, right: Cells`.

- [ ] **Step 3: Implement the flip**

In `simulate.rs` Default: `force_method: ForceMethod::Cells`.
In `cli.rs` RunArgs: `default_value_t = ForceMethod::Cells`.

- [ ] **Step 4: Run to verify GREEN**

Run: `cargo test --manifest-path md/Cargo.toml --lib defaults_are_the_course_contract_values`
Expected: PASS.

- [ ] **Step 5: Full regression + contract**

Run: `cargo test --manifest-path md/Cargo.toml`
Expected: all PASS, including `tests/contract.rs` (the default run now uses cells; the three physics bounds are method-agnostic to rounding).

Run: `cargo test --manifest-path md/Cargo.toml --release --test contract`
Expected: PASS (release contract stays GREEN).

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/simulate.rs week2/md/src/cli.rs
git commit -m "feat: make cells the default force method"
```

---

### Task 12: Post-implementation correctness VERIFY

**Files:**
- Modify: none (fix only if a verification step fails).

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Full release suite**

Run: `cargo test --manifest-path md/Cargo.toml --release`
Expected: all PASS (lib unit tests, `tests/cells.rs` equality/coverage suite, `tests/fluid.rs`, `tests/checker.rs`, `tests/cli.rs`, `tests/contract.rs`, Part 2/3).

- [ ] **Step 2: Cells-produced artifacts pass `md check` (checker is method-independent)**

```bash
cargo run --manifest-path md/Cargo.toml --release -- run --force cells --n 100 --out /tmp/ctr-cells
cargo run --manifest-path md/Cargo.toml --release -- check /tmp/ctr-cells
```
Expected: exit 0; all three bounds PASS; energy cross-check ~1e-15 (cells-stored vs naive-recomputed).

- [ ] **Step 3: Naive-produced artifacts pass `md check` too**

```bash
cargo run --manifest-path md/Cargo.toml --release -- run --force naive --n 100 --out /tmp/ctr-naive
cargo run --manifest-path md/Cargo.toml --release -- check /tmp/ctr-naive
```
Expected: exit 0; all bounds PASS.

- [ ] **Step 4: Cleanup + commit only if a fix was needed**

Remove `/tmp/ctr-cells`, `/tmp/ctr-naive`. If any step failed, fix the cause and re-run; otherwise make no commit here.

---

### Task 13: Profile `cells` (N=400) + `profile-cells.png`

**Files:**
- Modify: `week2/README.md` (fill the Cell-list row of the Profile table).
- Create: `week2/profile-cells.png`.

**Interfaces:**
- Consumes: the release binary. Naive evidence for comparison: force share **97.0%**, elapsed **0.635 s** (recorded in `week2/README.md` / `profile-naive.png`).

- [ ] **Step 1: Build release and check samply**

```bash
cargo build --manifest-path md/Cargo.toml --release
samply --version   # if not found, try `cargo install samply` (offline => note it)
```

- [ ] **Step 2: Profile the cells run (samply available)**

```bash
samply record md/target/release/md run --force cells --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof-cells
```

Open the samply URL, capture the flamegraph as `week2/profile-cells.png`; record the **inclusive force share %** (`md::fluid::fluid_accelerations` and its cells internals) and the **elapsed time (s)** from the profile run.

If samply is unavailable and cannot be installed: record elapsed wall-clock only via

```bash
/usr/bin/time -f "%e s" md/target/release/md run --force cells --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof-cells
```

and record the force share as **not captured (samply unavailable)** — do NOT invent a number.

- [ ] **Step 3: Compare against naive evidence and update README**

Update `week2/README.md` Profile table:

```markdown
| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 97.0 | 0.635 |
| Cell list | <measured force share> | <measured elapsed> |
```

Add a line under the profile command documenting the cells command:

```markdown
The cells profile was recorded with:

    samply record md run --force cells --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof-cells
```

and reference `profile-cells.png` next to `profile-naive.png`. Report measured values only; do not assert an invented speedup.

- [ ] **Step 4: Commit**

```bash
git add week2/README.md week2/profile-cells.png
git commit -m "feat: record Part 5 cells profile"
```

---

### Task 14: Scaling benchmarks N = 100, 400, 1600

**Files:**
- Modify: `week2/README.md` (add a scaling table).

**Interfaces:**
- Consumes: the release binary.

- [ ] **Step 1: Run the measured wall-clock benchmarks (both methods, 3 repeats, median)**

```bash
cargo build --manifest-path md/Cargo.toml --release
for m in naive cells; do
  for N in 100 400 1600; do
    echo "== $m N=$N =="
    for r in 1 2 3; do
      /usr/bin/time -f "%e" md/target/release/md run --force "$m" --n "$N" \
        --eq-steps 200 --steps 1000 --sample-every 100 --out "/tmp/bench-$N-$m" 2>&1 | tail -1
    done
  done
done
```

Record the median wall-clock per (N, method). The workload (`eq 200, steps 1000, sample 100`) is a [Suggestion] measurement choice; it is identical for both methods so the comparison is fair. Do not invent any expected timing or speedup.

- [ ] **Step 2: Record results in `week2/README.md`**

Add a scaling section with the measured medians:

```markdown
## Scaling (release, measured wall-clock median, eq 200 / steps 1000)

| N | Naive median (s) | Cells median (s) |
| --- | ---: | ---: |
| 100 | <measured> | <measured> |
| 400 | <measured> | <measured> |
| 1600 | <measured> | <measured> |
```

- [ ] **Step 3: Commit**

```bash
git add week2/README.md
git commit -m "feat: record Part 5 scaling benchmarks"
```

---

## Self-Review

**1. Spec coverage** — each approved design requirement maps to a task:
- final cells default → Task 11; naive reference preserved → Tasks 2/8; no run.json change → Tasks 9/10 + regression; ForceMethod no serde → Task 3; small-box guard [Suggestion] → Task 4 (comment); tolerance fixed before impl → Task 1; scale-aware criterion → Task 1 constants; `r = rc − ε` [Suggestion] → Task 1; two-cell dedup → Tasks 1 & 5; exact cutoff → Task 1; checker method-independence → Task 12; rebuild each eval + per-particle j>i + single-source pair physics → Tasks 2, 6, 7; cells force + energy → Task 7; no Verlet/incremental/parallelism/SIMD/heating → Global Constraints; VERIFY before perf claims → Task 12 before Tasks 13–14; profile + scaling measured-only → Tasks 13–14.

**2. Placeholder scan** — no TBD/TODO; every code step has concrete content; the only "…"/<measured> strings are the README *recording templates* in Tasks 13/14 whose values come from the executor's measurements (not invented).

**3. Type consistency** — `ForceMethod::{Naive,Cells}`, `.accelerations(state,bx)`, `.potential_energy(state,bx)`, `SimConfig.force_method`, `RunArgs.force` are defined once (Task 3) and used consistently in Tasks 8–11 and the tests/cells.rs from Task 1. `cell_geometry`→`(nx,ny,wx,wy)` is consistent across Tasks 4–7.
