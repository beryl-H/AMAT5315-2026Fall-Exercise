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

use advdiff::CentredFD;
use std::f64::consts::TAU;

fn fd_scheme() -> CentredFD {
    CentredFD { c: 1.0, nu: 0.1, n: 32 }
}

/// Grid values of cos(k0 x), k0 = 2.
fn cos_wave(n: usize, k0: f64) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            (k0 * x).cos()
        })
        .collect()
}

#[test]
fn fd_derivative_of_constant_field_is_zero() {
    let scheme = fd_scheme();
    let u = vec![3.7; scheme.n];
    assert!(scheme.derivative(&u).iter().all(|&d| d == 0.0));
    assert!(scheme.rate(&u).iter().all(|&r| r.abs() < 1e-15));
}

#[test]
fn fd_derivative_of_single_mode_matches_modified_wavenumber() {
    let scheme = fd_scheme();
    let h = TAU / scheme.n as f64;
    let k0 = 2.0_f64;
    let u = cos_wave(scheme.n, k0);
    let d = scheme.derivative(&u);
    // D cos(k0 x_j) = -(sin(k0 h)/h) sin(k0 x_j) exactly for centred differences.
    for (j, &dj) in d.iter().enumerate() {
        let x = TAU * j as f64 / scheme.n as f64;
        let expected = -(k0 * h).sin() / h * (k0 * x).sin();
        assert!((dj - expected).abs() < 1e-12, "j = {}: {} vs {}", j, dj, expected);
    }
}

#[test]
fn fd_rate_matches_modified_wavenumber_eigenvalues() {
    let scheme = fd_scheme();
    let h = TAU / scheme.n as f64;
    let k0 = 2.0_f64;
    let amp = 1.3;
    let mut u = cos_wave(scheme.n, k0);
    for v in u.iter_mut() {
        *v *= amp;
    }
    let r = scheme.rate(&u);
    // Advective eigenvalue -i c sin(k0 h)/h, diffusive -4 nu sin^2(k0 h/2)/h^2.
    let gamma = -4.0 * scheme.nu * (0.5 * k0 * h).sin().powi(2) / (h * h);
    let omega = scheme.c * (k0 * h).sin() / h;
    for (j, &rj) in r.iter().enumerate() {
        let x = TAU * j as f64 / scheme.n as f64;
        let expected = amp * (gamma * (k0 * x).cos() + omega * (k0 * x).sin());
        assert!((rj - expected).abs() < 1e-12, "j = {}: {} vs {}", j, rj, expected);
    }
}

/// Max abs difference between two vectors.
fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Max abs difference between two complex vectors.
fn max_err_c(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
}

/// Initial spectrum of sin(k0 x): u_hat[k0] = -i/2, u_hat[n - k0] = i/2.
fn sine_spectrum(n: usize, k0: usize) -> Vec<Complex<f64>> {
    let mut u_hat = vec![Complex::new(0.0, 0.0); n];
    u_hat[k0] = Complex::new(0.0, -0.5);
    u_hat[n - k0] = Complex::new(0.0, 0.5);
    u_hat
}

/// Exact one-step reference for the Fourier scheme: each mode grows by
/// exp((-i c k - nu k^2) dt) (single wave, spatially exact).
fn fourier_reference(scheme: &FourierAdvDiff, u_hat: &[Complex<f64>], dt: f64) -> Vec<Complex<f64>> {
    u_hat.iter()
        .enumerate()
        .map(|(j, &u)| {
            let k = scheme.wavenumber(j);
            let lambda = Complex::new(-scheme.nu * k * k, -scheme.c * k);
            u * (lambda * dt).exp()
        })
        .collect()
}

/// Grid values of sin(k0 x).
fn sine_wave(n: usize, k0: f64) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            (k0 * x).sin()
        })
        .collect()
}

/// Exact one-step reference for the centred-difference scheme: the
/// semi-discrete single wave travels with modified speed c sin(k0 h)/h and
/// decays at rate 4 nu sin^2(k0 h/2)/h^2.
fn fd_reference(scheme: &CentredFD, dt: f64, k0: f64) -> Vec<f64> {
    let h = TAU / scheme.n as f64;
    let damp = (-4.0 * scheme.nu * (0.5 * k0 * h).sin().powi(2) * dt / (h * h)).exp();
    let speed = scheme.c * (k0 * h).sin() / h;
    (0..scheme.n)
        .map(|j| {
            let x = TAU * j as f64 / scheme.n as f64;
            damp * (k0 * x - speed * dt).sin()
        })
        .collect()
}

fn step_fourier<I: Integrator>(int: &I, dt: f64) -> f64 {
    let scheme = fourier_scheme();
    let u_hat = sine_spectrum(scheme.n, 2);
    let y1 = int.step(|s| scheme.rate(s), dt, &u_hat);
    max_err_c(&y1, &fourier_reference(&scheme, &u_hat, dt))
}

fn step_fd<I: Integrator>(int: &I, dt: f64) -> f64 {
    let scheme = fd_scheme();
    let u = sine_wave(scheme.n, 2.0);
    let y1 = int.step(|s| scheme.rate(s), dt, &u);
    max_err(&y1, &fd_reference(&scheme, dt, 2.0))
}

#[test]
fn one_step_fourier_euler_within_bound() {
    assert!(step_fourier(&Euler, 0.05) < 1e-2);
}

#[test]
fn one_step_fourier_midpoint_within_bound() {
    assert!(step_fourier(&Midpoint, 0.05) < 1e-3);
}

#[test]
fn one_step_fourier_rk4_within_bound() {
    assert!(step_fourier(&RK4, 0.05) < 1e-6);
}

#[test]
fn one_step_fd_euler_within_bound() {
    assert!(step_fd(&Euler, 0.05) < 1e-2);
}

#[test]
fn one_step_fd_midpoint_within_bound() {
    assert!(step_fd(&Midpoint, 0.05) < 1e-3);
}

#[test]
fn one_step_fd_rk4_within_bound() {
    assert!(step_fd(&RK4, 0.05) < 1e-6);
}

/// One-step (local) error must shrink by 2^(p+1) when dt halves:
/// ~4 for Euler (p=1), ~8 for Midpoint (p=2), ~32 for RK4 (p=4).
#[test]
fn convergence_order_on_fourier_scheme() {
    let dt = 0.05_f64;
    let e_euler = step_fourier(&Euler, dt) / step_fourier(&Euler, dt / 2.0);
    let e_mid = step_fourier(&Midpoint, dt) / step_fourier(&Midpoint, dt / 2.0);
    let e_rk4 = step_fourier(&RK4, dt) / step_fourier(&RK4, dt / 2.0);
    assert!(e_euler > 3.5 && e_euler < 4.5, "Euler ratio {}", e_euler);
    assert!(e_mid > 7.0 && e_mid < 9.0, "Midpoint ratio {}", e_mid);
    assert!(e_rk4 > 28.0 && e_rk4 < 36.0, "RK4 ratio {}", e_rk4);
}

// --- Step driver ------------------------------------------------------------
use advdiff::advance;

#[test]
fn advance_lands_on_t_final_with_full_plus_partial_steps() {
    let history = advance(&Euler, decay, 0.3_f64, 1.0, &[1.0]);
    let times: Vec<f64> = history.iter().map(|(t, _)| *t).collect();
    let expected = [0.3, 0.6, 0.9, 1.0];
    assert_eq!(times.len(), expected.len());
    for (i, &t) in times.iter().enumerate() {
        assert!((t - expected[i]).abs() < 1e-12, "times[{}] = {}", i, t);
    }
    assert_eq!(*times.last().expect("non-empty"), 1.0);
}

#[test]
fn advance_with_dt_beyond_t_final_takes_one_short_step() {
    let history = advance(&Euler, decay, 2.0, 0.5, &[1.0]);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].0, 0.5);
    assert!((history[0].1[0] - (1.0 - 0.5)).abs() < 1e-12);
}

#[test]
fn advance_works_for_complex_state() {
    let history = advance(
        &Euler,
        |s: &[Complex<f64>]| s.iter().map(|v| -*v).collect(),
        0.1,
        0.2,
        &[Complex::new(1.0, 0.0)],
    );
    assert_eq!(history.len(), 2);
    assert!((history[0].1[0] - Complex::new(0.9, 0.0)).norm() < 1e-12);
    assert!((history[1].1[0] - Complex::new(0.81, 0.0)).norm() < 1e-12);
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
