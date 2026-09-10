//! Lennard-Jones accelerations, energy measurement, and (in Task 5) the
//! shared experiment driver.

use crate::state::{State, Vec2};
use crate::{lj_energy, lj_force};

/// Accelerations from the plain Lennard-Jones potential.
///
/// Mass is 1, so acceleration equals force. For each pair `i < j`:
///
/// ```text
/// d = r_i - r_j
/// acceleration_i += lj_force(|d|) * d / |d|
/// acceleration_j -= lj_force(|d|) * d / |d|
/// ```
pub fn lj_accelerations(state: &State) -> Vec<Vec2> {
    let n = state.positions.len();
    let mut accelerations = vec![[0.0, 0.0]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = state.positions[i][0] - state.positions[j][0];
            let dy = state.positions[i][1] - state.positions[j][1];
            let r = (dx * dx + dy * dy).sqrt();
            let f_over_r = lj_force(r) / r;

            accelerations[i][0] += f_over_r * dx;
            accelerations[i][1] += f_over_r * dy;
            accelerations[j][0] -= f_over_r * dx;
            accelerations[j][1] -= f_over_r * dy;
        }
    }

    accelerations
}

/// Kinetic energy with mass 1: `0.5 * sum_i |v_i|^2`.
pub fn kinetic_energy(state: &State) -> f64 {
    state
        .velocities
        .iter()
        .map(|velocity| 0.5 * (velocity[0] * velocity[0] + velocity[1] * velocity[1]))
        .sum()
}

/// Potential energy: `sum_{i<j} U(r_ij)` with the existing `lj_energy`.
pub fn potential_energy(state: &State) -> f64 {
    let mut energy = 0.0;
    for i in 0..state.positions.len() {
        for j in (i + 1)..state.positions.len() {
            let dx = state.positions[i][0] - state.positions[j][0];
            let dy = state.positions[i][1] - state.positions[j][1];
            energy += lj_energy((dx * dx + dy * dy).sqrt());
        }
    }
    energy
}

/// Total mechanical energy measured with the Lennard-Jones potential.
pub fn total_energy(state: &State) -> f64 {
    kinetic_energy(state) + potential_energy(state)
}

/// Relative energy error against a fixed reference energy `e0`.
pub fn relative_energy_error(state: &State, e0: f64) -> f64 {
    (total_energy(state) - e0) / e0.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::two_atom_initial_state;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn initial_total_energy_is_lj_energy_at_spec_separation() {
        let state = two_atom_initial_state();
        let expected = 4.0 * (1.2f64.powi(-12) - 1.2f64.powi(-6));
        assert!((total_energy(&state) - expected).abs() < EPSILON);
        assert_eq!(relative_energy_error(&state, total_energy(&state)), 0.0);
    }

    #[test]
    fn pair_acceleration_follows_spec_sign_convention() {
        let state = State {
            positions: vec![[0.0, 0.0], [1.2, 0.0]],
            velocities: vec![[0.0, 0.0], [0.0, 0.0]],
        };
        let accelerations = lj_accelerations(&state);
        let scalar_force = crate::lj_force(1.2);
        // d / r for atom 0 relative to atom 1 is (-1, 0).
        let expected_first_x = scalar_force * (-1.0);

        assert!((accelerations[0][0] - expected_first_x).abs() < EPSILON);
        assert_eq!(accelerations[0][1], 0.0);
        assert!((accelerations[1][0] + accelerations[0][0]).abs() < EPSILON);
        assert_eq!(accelerations[1][1], 0.0);
    }
}
