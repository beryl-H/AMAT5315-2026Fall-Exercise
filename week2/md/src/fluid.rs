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
}
