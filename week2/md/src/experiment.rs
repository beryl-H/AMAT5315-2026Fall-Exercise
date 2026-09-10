//! Lennard-Jones accelerations, energy measurement, and (in Task 5) the
//! shared experiment driver.

use crate::integrator::Integrator;
use crate::state::{two_atom_initial_state, State, Vec2};
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

/// Time series produced by the shared two-atom experiment driver.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentResult {
    pub steps: Vec<usize>,
    pub times: Vec<f64>,
    pub relative_energy_errors: Vec<f64>,
}

/// Run the fixed Week 2 dimer with any `Integrator`.
///
/// `steps` counts integrated steps. The returned series includes step 0 with
/// error 0.0, so its length is `steps + 1`.
pub fn run_experiment<I: Integrator>(
    integrator: &mut I,
    steps: usize,
    dt: f64,
) -> ExperimentResult {
    let mut state = two_atom_initial_state();
    let e0 = total_energy(&state);
    let accelerations = |s: &State| lj_accelerations(s);

    integrator.initialize(&state, &accelerations);

    let mut step_indices = Vec::with_capacity(steps + 1);
    let mut times = Vec::with_capacity(steps + 1);
    let mut errors = Vec::with_capacity(steps + 1);

    for step in 0..=steps {
        if step > 0 {
            integrator.step(&mut state, dt, &accelerations);
        }
        step_indices.push(step);
        times.push(step as f64 * dt);
        errors.push(relative_energy_error(&state, e0));
    }

    ExperimentResult {
        steps: step_indices,
        times,
        relative_energy_errors: errors,
    }
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

    #[test]
    fn experiment_records_step_zero_and_every_integrated_step() {
        let result = run_experiment(&mut crate::Euler::default(), 2, 0.5);
        assert_eq!(result.steps, vec![0, 1, 2]);
        assert_eq!(result.times, vec![0.0, 0.5, 1.0]);
        assert_eq!(result.relative_energy_errors[0], 0.0);
    }
}
