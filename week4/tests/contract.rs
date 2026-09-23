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

use advdiff::{dft, FourierAdvDiff, idft};

fn fourier_scheme() -> FourierAdvDiff {
    FourierAdvDiff { c: 1.0, nu: 0.1, n: 32 }
}

#[test]
fn fourier_derivative_of_constant_field_is_zero() {
    let scheme = fourier_scheme();
    // Constant field: only the k = 0 mode is excited; i * 0 = 0.
    let mut u_hat = vec![Complex::new(0.0, 0.0); scheme.n];
    u_hat[0] = Complex::new(3.7, 0.0);
    assert!(scheme.derivative(&u_hat).iter().all(|&d| d == Complex::new(0.0, 0.0)));
}

#[test]
fn fourier_derivative_of_single_mode_is_ik() {
    let scheme = fourier_scheme();
    let mut u_hat = vec![Complex::new(0.0, 0.0); scheme.n];
    u_hat[2] = Complex::new(1.0, 0.0);
    let d = scheme.derivative(&u_hat);
    // d/dx e^{i k x} = i k e^{i k x}, k = 2
    assert!((d[2] - Complex::new(0.0, 2.0)).norm() < 1e-14);
    assert!(d.iter().enumerate().all(|(j, &v)| j == 2 || v.norm() < 1e-14));
}

#[test]
fn fourier_rate_of_single_mode_is_neg_ick_neg_nu_k2() {
    let scheme = fourier_scheme();
    let mut u_hat = vec![Complex::new(0.0, 0.0); scheme.n];
    u_hat[2] = Complex::new(1.0, 0.0);
    let r = scheme.rate(&u_hat);
    // (-i c k - nu k^2) u = (-2i - 0.4) at k = 2
    assert!((r[2] - Complex::new(-0.4, -2.0)).norm() < 1e-14);
}

#[test]
fn nyquist_mode_does_not_travel_only_diffuses() {
    let scheme = fourier_scheme(); // n = 32, Nyquist index 16
    let nyq = scheme.n / 2;
    let mut u_hat = vec![Complex::new(0.0, 0.0); scheme.n];
    u_hat[nyq] = Complex::new(1.5, -0.5);
    let r = scheme.rate(&u_hat);
    // Exactly -nu (n/2)^2 * u_hat: a real damping coefficient, no advective
    // (i c k) term. A defect that lets the mode travel changes the complex
    // product and fails this comparison.
    let expected = Complex::new(-scheme.nu * (nyq as f64) * (nyq as f64), 0.0) * u_hat[nyq];
    assert_eq!(r[nyq], expected, "Nyquist mode must not travel");
}

#[test]
fn dft_idft_round_trip_recovers_sine() {
    let n = 32usize;
    let k0 = 2.0_f64;
    let u: Vec<f64> = (0..n)
        .map(|j| {
            let x = std::f64::consts::TAU * j as f64 / n as f64;
            (k0 * x).sin()
        })
        .collect();
    let u_hat = dft(&u);
    // sin(k0 x) = (e^{i k0 x} - e^{-i k0 x}) / (2i): u_hat[k0] = -i/2, u_hat[n - k0] = i/2
    assert!((u_hat[2] - Complex::new(0.0, -0.5)).norm() < 1e-12);
    assert!((u_hat[30] - Complex::new(0.0, 0.5)).norm() < 1e-12);
    let back = idft(&u_hat);
    for (a, b) in back.iter().zip(&u) {
        assert!((a - b).abs() < 1e-12, "round trip mismatch {} vs {}", a, b);
    }
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
