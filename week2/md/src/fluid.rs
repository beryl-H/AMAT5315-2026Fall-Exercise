//! Periodic-boundary fluid forces and energies with the shifted cutoff.
//!
//! O(N^2) pair loop; no cell lists (Part 5).

use crate::pair::{shifted_energy, shifted_force};
use crate::state::State;
use crate::system::{Box2, minimum_image};
use crate::Vec2;

/// Accelerations (mass 1) with minimum-image displacements and the
/// shifted-cutoff force. Pair contributions are antisymmetric.
pub fn fluid_accelerations(state: &State, bx: &Box2) -> Vec<Vec2> {
    let n = state.positions.len();
    let mut acc = vec![[0.0, 0.0]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = minimum_image(state.positions[i][0] - state.positions[j][0], bx.lx);
            let dy = minimum_image(state.positions[i][1] - state.positions[j][1], bx.ly);
            let r = (dx * dx + dy * dy).sqrt();
            let f_over_r = shifted_force(r) / r;
            let fx = f_over_r * dx;
            let fy = f_over_r * dy;
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
            let dx = minimum_image(state.positions[i][0] - state.positions[j][0], bx.lx);
            let dy = minimum_image(state.positions[i][1] - state.positions[j][1], bx.ly);
            energy += shifted_energy((dx * dx + dy * dy).sqrt());
        }
    }
    energy
}
