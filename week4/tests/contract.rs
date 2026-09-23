//! Contract tests for the Week 4 integrators and rate functions.

use advdiff::{Euler, Integrator};
use num_complex::Complex;

/// dy/dt = -y, exact one-step polynomials are known in closed form.
fn decay(y: &[f64]) -> Vec<f64> {
    y.iter().map(|v| -v).collect()
}

#[test]
fn euler_one_step_matches_exact_polynomial() {
    let dt = 0.1;
    let y = Euler.step(decay, dt, &[1.0]);
    let expected = 1.0 - dt;
    assert!((y[0] - expected).abs() < 1e-12, "got {}, want {}", y[0], expected);
}

use advdiff::Midpoint;

#[test]
fn midpoint_one_step_matches_exact_polynomial() {
    let dt = 0.1;
    let y = Midpoint.step(decay, dt, &[1.0]);
    let expected = 1.0 - dt + dt * dt / 2.0;
    assert!((y[0] - expected).abs() < 1e-12, "got {}, want {}", y[0], expected);
}

use advdiff::RK4;

#[test]
fn rk4_one_step_matches_exact_polynomial() {
    let dt = 0.1_f64;
    let y = RK4.step(decay, dt, &[1.0]);
    let expected = 1.0 - dt + dt * dt / 2.0 - dt.powi(3) / 6.0 + dt.powi(4) / 24.0;
    assert!((y[0] - expected).abs() < 1e-12, "got {}, want {}", y[0], expected);
}

#[test]
fn euler_works_for_complex_state() {
    let y = Euler.step(
        |s: &[Complex<f64>]| s.iter().map(|v| -*v).collect(),
        0.1,
        &[Complex::new(1.0, 2.0)],
    );
    assert!((y[0] - Complex::new(0.9, 1.8)).norm() < 1e-12);
}
