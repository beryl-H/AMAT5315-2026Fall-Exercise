//! Integration tests for the Week 2 two-atom molecular-dynamics experiment.
//!
//! All assertions are derived from the course specification, not from the
//! implementation.

use std::cell::Cell;

use md::{
    Euler, Integrator, State, VelocityVerlet, lj_accelerations, run_experiment,
    two_atom_initial_state,
};

const DT: f64 = 0.01;
const STEPS_500: usize = 500;
const STEPS_5000: usize = 5000;

fn max_abs(errors: &[f64]) -> f64 {
    errors.iter().fold(0.0, |max, error| max.max(error.abs()))
}

#[test]
fn forward_euler_final_relative_energy_error_exceeds_half() {
    let result = run_experiment(&mut Euler::default(), STEPS_500, DT);
    let final_error = *result.relative_energy_errors.last().unwrap();
    assert!(
        final_error > 0.5,
        "Euler final relative energy error was {final_error}, expected > 0.5"
    );
}

#[test]
fn velocity_verlet_max_absolute_relative_energy_error_is_below_one_thousandth() {
    let result = run_experiment(&mut VelocityVerlet::default(), STEPS_500, DT);
    let max_error = max_abs(&result.relative_energy_errors);
    assert!(
        max_error < 1e-3,
        "velocity-Verlet max |relative energy error| was {max_error}, expected < 1e-3"
    );
}

#[test]
fn velocity_verlet_5000_step_run_is_available_and_finite() {
    let result = run_experiment(&mut VelocityVerlet::default(), STEPS_5000, DT);
    assert_eq!(result.relative_energy_errors.len(), STEPS_5000 + 1);
    assert_eq!(*result.steps.last().unwrap(), STEPS_5000);
    for error in &result.relative_energy_errors {
        assert!(error.is_finite(), "non-finite error in long velocity-Verlet run");
    }
}

#[test]
fn shared_driver_handles_both_integrators() {
    let euler = run_experiment(&mut Euler::default(), STEPS_500, DT);
    let verlet = run_experiment(&mut VelocityVerlet::default(), STEPS_500, DT);
    assert_eq!(euler.steps, verlet.steps);
    assert_eq!(euler.times, verlet.times);
    assert_eq!(euler.relative_energy_errors[0], 0.0);
    assert_eq!(verlet.relative_energy_errors[0], 0.0);
}

#[test]
fn velocity_verlet_evaluates_acceleration_once_per_step_after_initialization() {
    let mut state = two_atom_initial_state();
    let mut verlet = VelocityVerlet::default();
    let calls = Cell::new(0usize);
    let counting_acceleration = |s: &State| {
        calls.set(calls.get() + 1);
        lj_accelerations(s)
    };

    verlet.initialize(&state, &counting_acceleration);
    assert_eq!(calls.get(), 1, "initial acceleration must be evaluated once");

    verlet.step(&mut state, DT, &counting_acceleration);
    assert_eq!(calls.get(), 2, "each step must evaluate acceleration once");

    verlet.step(&mut state, DT, &counting_acceleration);
    assert_eq!(calls.get(), 3, "each step must evaluate acceleration once");
}
