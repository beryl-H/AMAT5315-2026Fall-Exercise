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
}
