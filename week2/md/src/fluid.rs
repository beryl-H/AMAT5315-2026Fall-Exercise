//! Periodic-boundary fluid forces and energies with the shifted cutoff.
//!
//! O(N^2) pair loop; no cell lists (Part 5).

use crate::pair::{shifted_energy, shifted_force};
use crate::state::State;
use crate::system::{Box2, minimum_image};
use crate::Vec2;

/// Force evaluation method: the naive O(N^2) reference or the cell list.
/// A pure runtime/performance selection; never serialized to run.json.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceMethod {
    Naive,
    Cells,
}

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

/// Accelerations (mass 1) with minimum-image displacements and the
/// shifted-cutoff force. Pair contributions are antisymmetric.
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

/// Sum of shifted pair energies over minimum-image pairs.
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

/// Single source of pair physics: minimum-image displacement + shifted force.
/// Shared by the naive and (later) cell-list enumeration paths.
fn pair_force(dx_raw: f64, dy_raw: f64, bx: &Box2) -> (f64, f64) {
    let dx = minimum_image(dx_raw, bx.lx);
    let dy = minimum_image(dy_raw, bx.ly);
    let r = (dx * dx + dy * dy).sqrt();
    let f_over_r = shifted_force(r) / r;
    (f_over_r * dx, f_over_r * dy)
}

/// Single source of pair physics: minimum-image displacement + shifted energy.
/// Shared by the naive and (later) cell-list enumeration paths.
fn pair_energy(dx_raw: f64, dy_raw: f64, bx: &Box2) -> f64 {
    let dx = minimum_image(dx_raw, bx.lx);
    let dy = minimum_image(dy_raw, bx.ly);
    shifted_energy((dx * dx + dy * dy).sqrt())
}

/// Rectangular cell grid: nx = floor(Lx/rc), wx = Lx/nx, and similarly for y.
///
/// [Course Requirement] For L >= rc: n = floor(L / rc), so n <= L / rc and
/// therefore w = L / n >= rc. This is the invariant that makes the 3x3
/// neighbor search complete.
///
/// [Suggestion] For L < rc the max(1, ...) guard gives n = 1 and w = L,
/// which may be < rc. Completeness in that defensive one-cell case does NOT
/// follow from w >= rc; it holds because the single cell spans the whole
/// axis, so all particles on that axis belong to that same cell.
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

/// Wrapped + deduplicated 3x3 neighbor cell indices of `c` [Course Req]:
/// offsets dx, dy in {-1, 0, 1} wrap periodically via rem_euclid, and when
/// several offsets land on the same wrapped cell (e.g. a two-cell-wide box)
/// the index is returned only once. At most 9 candidates, so a linear
/// contains() is fine.
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

/// Particle-index lists per cell, rebuilt on every evaluation [design]: each
/// particle is pushed into the cell containing its CURRENT wrapped position,
/// so every index appears in exactly one bin; empty cells stay empty.
fn build_cells(positions: &[Vec2], nx: usize, ny: usize, wx: f64, wy: f64) -> Vec<Vec<usize>> {
    let mut cells = vec![Vec::new(); nx * ny];
    for (i, p) in positions.iter().enumerate() {
        cells[cell_index(p[0], p[1], wx, wy, nx, ny)].push(i);
    }
    cells
}

/// Visit each unordered pair exactly once via the cell list [Course Req]:
/// per-particle 3x3 (wrapped + deduplicated) neighborhood with a j > i guard.
/// The callback receives (i, j) and the RAW coordinate differences; the
/// minimum-image / shifted-LJ physics lives in pair_force / pair_energy.
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
                    f(
                        i,
                        j,
                        positions[i][0] - positions[j][0],
                        positions[i][1] - positions[j][1],
                    );
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

    #[test]
    fn force_method_has_both_variants_and_minimal_traits() {
        // Minimal runtime-selection enum: Clone/Copy/Debug/PartialEq/Eq only.
        assert_eq!(ForceMethod::Naive, ForceMethod::Naive);
        assert_eq!(ForceMethod::Cells, ForceMethod::Cells);
        assert_ne!(ForceMethod::Naive, ForceMethod::Cells);
        let copied = ForceMethod::Cells; // Copy
        assert_eq!(copied, ForceMethod::Cells);
        let dbg = format!("{:?}", ForceMethod::Naive); // Debug
        assert_eq!(dbg, "Naive");
    }

    #[test]
    fn cell_geometry_matches_course_rule() {
        // Default contract: Lx ~ 12.014, Ly ~ 10.404 (not multiples of rc).
        let bx = Box2::new(100, 0.8);
        let (nx, ny, wx, wy) = cell_geometry(&bx);
        assert_eq!((nx, ny), (4, 4));
        assert!(wx >= crate::pair::RC && wy >= crate::pair::RC);
        // Profile-sized box (N = 400): Lx ~ 24.028, Ly ~ 20.809, non-divisible.
        let bx400 = Box2::new(400, 0.8);
        let (nx400, ny400, wx400, wy400) = cell_geometry(&bx400);
        assert_eq!((nx400, ny400), (9, 8));
        assert!(wx400 > crate::pair::RC && wy400 > crate::pair::RC);
        // Two-cell-wide box (nx = ny = 2, w = rc = 2.5 exactly).
        let small = Box2 { lx: 5.0, ly: 5.0 };
        let (n2x, n2y, w2x, w2y) = cell_geometry(&small);
        assert_eq!((n2x, n2y), (2, 2));
        assert!((w2x - 2.5).abs() < 1e-12 && (w2y - 2.5).abs() < 1e-12);
        // [Suggestion] defensive sub-rc box (L < rc): the max(1, ...) guard
        // yields n = 1 and w = L = 1.0 < rc. w >= rc is intentionally NOT
        // asserted here; completeness comes from the single cell spanning the
        // whole axis.
        let tiny = Box2 { lx: 1.0, ly: 1.0 };
        let (tnx, tny, twx, twy) = cell_geometry(&tiny);
        assert_eq!((tnx, tny), (1, 1));
        assert!((twx - 1.0).abs() < 1e-12 && (twy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cell_index_maps_wrapped_positions() {
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let (nx, ny, wx, wy) = cell_geometry(&bx); // 4x4, w = 2.5
        // lower boundary
        assert_eq!(cell_index(0.0, 0.0, wx, wy, nx, ny), 0);
        assert_eq!(cell_index(2.4, 2.4, wx, wy, nx, ny), 0);
        // just across an internal cell boundary
        assert_eq!(cell_index(2.5, 0.0, wx, wy, nx, ny), 1);
        assert_eq!(cell_index(0.0, 2.5, wx, wy, nx, ny), nx);
        // upper wrapped boundary (positions are in [0, Lx) x [0, Ly))
        assert_eq!(cell_index(9.99, 9.99, wx, wy, nx, ny), nx * ny - 1);
        assert_eq!(cell_index(10.0 - 1e-12, 0.0, wx, wy, nx, ny), nx - 1);
        assert_eq!(cell_index(0.0, 10.0 - 1e-12, wx, wy, nx, ny), (ny - 1) * nx);
    }

    #[test]
    fn neighbor_cells_of_interior_cell_are_nine_unique() {
        // 4x4 grid, interior cell 5 = (cx=1, cy=1): offsets -1..1 stay in
        // range, so exactly 9 unique neighbors with no wrapping needed.
        let n = neighbor_cells(5, 4, 4);
        assert_eq!(n.len(), 9);
        let mut expected: Vec<usize> = vec![0, 1, 2, 4, 5, 6, 8, 9, 10];
        let mut got = n.clone();
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(got, expected);
        // no duplicates
        let mut s = n.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), n.len());
    }

    #[test]
    fn neighbor_cells_wrap_at_opposite_edges() {
        // 4x4 grid, corner cell 0 = (cx=0, cy=0): the 3x3 window wraps onto
        // the opposite edges, giving 9 unique cells: {0,1,3,4,5,7,12,13,15}.
        let n = neighbor_cells(0, 4, 4);
        assert_eq!(n.len(), 9);
        for d in [0usize, 3, 12, 15] {
            assert!(n.contains(&d), "cell {d} must be a wrapped neighbor of 0");
        }
        let mut expected: Vec<usize> = vec![0, 1, 3, 4, 5, 7, 12, 13, 15];
        let mut got = n.clone();
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(got, expected);
    }

    #[test]
    fn neighbor_cells_deduplicate_in_two_cell_box() {
        // nx = ny = 2: each offset set {-1,0,1} wraps to {0,1}, so the 9
        // candidates collapse to exactly the 4 distinct cells, none twice.
        for c in 0..4 {
            let n = neighbor_cells(c, 2, 2);
            assert_eq!(n.len(), 4, "cell {c} must yield exactly 4 distinct cells");
            for d in 0..4 {
                assert!(n.contains(&d));
            }
        }
    }

    #[test]
    fn neighbor_cells_collapse_in_one_cell_box() {
        // [Suggestion] nx = ny = 1: all 9 offsets wrap onto the only cell.
        let n = neighbor_cells(0, 1, 1);
        assert_eq!(n.len(), 1);
        assert_eq!(n, vec![0]);
    }

    #[test]
    fn build_cells_places_known_positions_into_known_cells() {
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let (nx, ny, wx, wy) = cell_geometry(&bx); // 4x4, w = 2.5
        let pos: Vec<Vec2> = vec![
            [0.1, 0.1], // cell 0
            [0.2, 0.3], // cell 0, same cell as the particle above
            [2.6, 0.2], // cell 1
            [7.7, 7.8], // cell 15
            [9.9, 9.9], // cell 15, same cell as the particle above
            [5.0, 5.0], // cell 10
        ];
        let cells = build_cells(&pos, nx, ny, wx, wy);
        assert_eq!(cells.len(), nx * ny, "one bin per cell");
        assert_eq!(cells.iter().map(|c| c.len()).sum::<usize>(), pos.len());
        for (i, p) in pos.iter().enumerate() {
            let c = cell_index(p[0], p[1], wx, wy, nx, ny);
            assert!(cells[c].contains(&i), "particle {i} must be in its cell {c}");
        }
        // multiple particles share a cell
        assert!(cells[0].contains(&0) && cells[0].contains(&1));
        assert!(cells[15].contains(&3) && cells[15].contains(&4));
        // with 6 particles over 16 cells at least one cell is empty
        assert!(cells.iter().any(|c| c.is_empty()));
    }

    #[test]
    fn build_cells_places_particles_near_periodic_boundaries() {
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let (nx, ny, wx, wy) = cell_geometry(&bx);
        let eps = 1e-9;
        let pos: Vec<Vec2> = vec![
            [eps, eps],                  // lower corner -> cell 0
            [10.0 - eps, 5.0],           // near x = Lx -> cx = nx - 1
            [5.0, 10.0 - eps],           // near y = Ly -> cy = ny - 1
            [10.0 - eps, 10.0 - eps],    // upper corner -> last cell
        ];
        let cells = build_cells(&pos, nx, ny, wx, wy);
        for (i, p) in pos.iter().enumerate() {
            let c = cell_index(p[0], p[1], wx, wy, nx, ny);
            assert!(cells[c].contains(&i), "boundary particle {i} in cell {c}");
        }
        assert_eq!(cell_index(eps, eps, wx, wy, nx, ny), 0);
        assert_eq!(cell_index(10.0 - eps, 10.0 - eps, wx, wy, nx, ny), nx * ny - 1);
    }

    #[test]
    fn build_cells_invariant_collects_each_index_once() {
        // Deterministic wrapped perturbation of the lattice: every index
        // 0..N appears exactly once across all bins, in no particular order.
        let bx = Box2::new(100, 0.8);
        let (nx, ny, wx, wy) = cell_geometry(&bx);
        let state = crate::system::lattice_state(100, 0.8);
        let mut pos = state.positions.clone();
        for (i, p) in pos.iter_mut().enumerate() {
            if i % 3 == 0 {
                p[0] = crate::system::wrap(p[0] + 0.17, bx.lx);
                p[1] = crate::system::wrap(p[1] + 0.09, bx.ly);
            }
        }
        let cells = build_cells(&pos, nx, ny, wx, wy);
        assert_eq!(cells.len(), nx * ny);
        let mut all: Vec<usize> = cells.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..pos.len()).collect::<Vec<usize>>());
    }

    #[test]
    fn build_cells_two_cell_wide_box() {
        let bx = Box2 { lx: 5.0, ly: 5.0 }; // nx = ny = 2
        let (nx, ny, wx, wy) = cell_geometry(&bx);
        let pos: Vec<Vec2> = (0..8)
            .map(|i| [0.7 + (i as f64 % 4.0), 0.7 + (i as f64 / 4.0)])
            .collect();
        let cells = build_cells(&pos, nx, ny, wx, wy);
        assert_eq!(cells.len(), 4);
        let mut all: Vec<usize> = cells.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..8).collect::<Vec<usize>>());
    }

    /// Collect the unordered pairs visited by the cell-list enumerator.
    fn collect_pairs(positions: &[Vec2], bx: &Box2) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for_each_pair_cells(positions, bx, |i, j, _dx, _dy| pairs.push((i, j)));
        pairs
    }

    #[test]
    fn pair_enumeration_normal_multicell_configuration() {
        // 10x10 box (4x4 cells). Particles: p0 cell 0, p1 cell 15, p2 cell 1,
        // p3 cell 15. Cell 0's wrapped 3x3 includes cell 15; cell 1 and 15 are
        // NOT mutual neighbors, so (1,2) and (2,3) are not candidates.
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let pos: Vec<Vec2> = vec![[0.1, 0.1], [9.9, 9.9], [2.6, 0.2], [7.7, 7.8]];
        let pairs = collect_pairs(&pos, &bx);
        assert!(pairs.iter().all(|&(i, j)| i < j));
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), pairs.len(), "no pair may appear twice");
        let mut expected: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3), (1, 3)];
        expected.sort_unstable();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn pair_enumeration_two_cell_box_counts_each_pair_once() {
        // 5x5 box (nx = ny = 2): wrapped offsets duplicate cells, but the
        // deduplicated neighbor list + j > i must still yield each of the
        // N(N-1)/2 unordered pairs exactly once.
        let bx = Box2 { lx: 5.0, ly: 5.0 };
        let pos: Vec<Vec2> = (0..8)
            .map(|i| [0.7 + (i as f64 % 4.0), 0.7 + (i as f64 / 4.0)])
            .collect();
        let pairs = collect_pairs(&pos, &bx);
        assert!(pairs.iter().all(|&(i, j)| i < j));
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), pairs.len(), "wrapped duplicates must not duplicate pairs");
        assert_eq!(pairs.len(), pos.len() * (pos.len() - 1) / 2);
    }

    #[test]
    fn pair_enumeration_one_cell_box_all_pairs_once() {
        // [Suggestion] nx = ny = 1: all particles share the single cell, so
        // every unordered pair is visited exactly once.
        let bx = Box2 { lx: 1.0, ly: 1.0 };
        let pos: Vec<Vec2> = vec![[0.1, 0.1], [0.3, 0.7], [0.5, 0.2], [0.8, 0.9], [0.2, 0.5]];
        let pairs = collect_pairs(&pos, &bx);
        assert!(pairs.iter().all(|&(i, j)| i < j));
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), pairs.len());
        assert_eq!(pairs.len(), pos.len() * (pos.len() - 1) / 2);
    }

    #[test]
    fn pair_enumeration_includes_periodic_edge_neighbors() {
        // 10x10 box: particle 0 in cell (0,2), particle 1 in cell (3,2).
        // Cell (3,2) is a wrapped 3x3 neighbor of cell (0,2), so the pair
        // across the periodic boundary must be enumerated.
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let pos: Vec<Vec2> = vec![[0.1, 5.0], [9.9, 5.0]];
        let pairs = collect_pairs(&pos, &bx);
        assert!(pairs.contains(&(0, 1)), "periodic edge pair must be enumerated");
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn cells_forces_sum_to_zero() {
        // Newton's third law: antisymmetric +f/-f per pair.
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
        // Deterministic wrapped perturbation of the 100-lattice (4x4 cells):
        // the private cell implementation must agree with the naive reference.
        let bx = Box2::new(100, 0.8);
        let mut state = crate::system::lattice_state(100, 0.8);
        for (i, p) in state.positions.iter_mut().enumerate() {
            if i % 3 == 0 {
                p[0] = crate::system::wrap(p[0] + 0.13, bx.lx);
                p[1] = crate::system::wrap(p[1] + 0.07, bx.ly);
            }
        }
        let na = fluid_accelerations(&state, &bx);
        let cl = accelerations_cells(&state, &bx);
        for k in 0..na.len() {
            assert!(
                (na[k][0] - cl[k][0]).abs() < 1e-9 && (na[k][1] - cl[k][1]).abs() < 1e-9,
                "atom {k} force mismatch"
            );
        }
        assert!(
            (fluid_potential_energy(&state, &bx) - potential_energy_cells(&state, &bx)).abs() < 1e-9
        );
    }

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
        assert!(
            (ForceMethod::Naive.potential_energy(&state, &bx)
                - fluid_potential_energy(&state, &bx))
                .abs()
                < 1e-12
        );
    }
}
