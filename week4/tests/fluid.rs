//! Contract tests for the vorticity-equation solver (fluid.design.toml).

use advdiff::field::{hermitian_mode, idft2, taylor_green_spectral};
use advdiff::fluid::{energy, enstrophy, nonlinear_term, rate, stream_function, velocity};
use advdiff::{advance, RK4};
use num_complex::Complex;
use std::f64::consts::TAU;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

#[test]
fn stream_function_has_the_confirmed_negative_sign_and_zero_mean() {
    let n = 32;
    let mut omega = vec![c(0.0, 0.0); n * n];
    omega[2 * n + 1] = c(1.0, 0.0); // ky = 2, kx = 1, kappa^2 = 5
    let psi = stream_function(n, &omega);
    // pairing A: psi_hat = +omega_hat / kappa^2 (omega = -laplacian(psi))
    assert!((psi[2 * n + 1] - c(1.0 / 5.0, 0.0)).norm() < 1e-14);
    assert_eq!(psi[0], c(0.0, 0.0));
}

#[test]
fn taylor_green_velocity_matches_the_textbook_field() {
    let n = 16;
    let omega = taylor_green_spectral(n);
    let psi = stream_function(n, &omega);
    let (ux, uy) = velocity(n, &psi);
    let ux_p = idft2(n, &ux);
    let uy_p = idft2(n, &uy);
    for j in 0..n {
        for i in 0..n {
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            assert!((ux_p[j * n + i] - x.sin() * y.cos()).abs() < 1e-12);
            assert!((uy_p[j * n + i] + x.cos() * y.sin()).abs() < 1e-12);
        }
    }
}

#[test]
fn single_hermitian_mode_has_zero_nonlinear_term() {
    let n = 32;
    let omega = hermitian_mode(n, 2, 3, c(1.0, 0.5));
    let nl = nonlinear_term(n, &omega);
    for (q, &v) in nl.iter().enumerate() {
        assert!(v.norm() < 1e-12, "mode {} = {}", q, v);
    }
}

#[test]
fn two_mode_interaction_matches_the_paper_spectrum() {
    let n = 32;
    // modes k1 = (1, 2) amplitude 1 and k2 = (3, 1) amplitude 2i, Hermitian
    let mut omega = vec![c(0.0, 0.0); n * n];
    let at = |kx: usize, ky: usize| ky * n + kx;
    omega[at(1, 2)] = c(1.0, 0.0);
    omega[at(31, 30)] = c(1.0, 0.0);
    omega[at(3, 1)] = c(0.0, 2.0);
    omega[at(29, 31)] = c(0.0, -2.0);
    let nl = nonlinear_term(n, &omega);
    // reference spectrum (fluid.design.toml, numpy cross-implementation):
    // +i on the sum and difference modes, -i on their conjugates
    let expected = [
        (at(4, 3), c(0.0, 1.0)),    // k1 + k2
        (at(n - 4, n - 3), c(0.0, -1.0)),
        (at(n - 2, 1), c(0.0, 1.0)),  // k1 - k2
        (at(2, n - 1), c(0.0, -1.0)),
    ];
    for (q, &v) in nl.iter().enumerate() {
        if let Some(&(_, e)) = expected.iter().find(|(qe, _)| *qe == q) {
            assert!((v - e).norm() < 1e-10, "mode {}: got {}, want {}", q, v, e);
        } else {
            assert!(v.norm() < 1e-10, "mode {} = {}, want 0", q, v);
        }
    }
}

#[test]
fn taylor_green_rate_is_pure_viscous_decay() {
    let n = 16;
    let nu = 0.05;
    let omega = taylor_green_spectral(n);
    let r = rate(n, nu, &omega);
    // nonlinear term zero, laplacian = -kappa^2 = -2: rate = -2 nu omega
    for (q, &v) in r.iter().enumerate() {
        assert!((v - omega[q] * (-2.0 * nu)).norm() < 1e-12, "mode {}", q);
    }
}

#[test]
fn taylor_green_decays_as_exp_minus_two_nu_t() {
    let n = 16;
    let nu = 0.05;
    let omega0 = taylor_green_spectral(n);
    let final_state = advance(&RK4, |s| rate(n, nu, s), 0.01, 1.0, &omega0)
        .last()
        .expect("non-empty")
        .1
        .clone();
    let factor = (-2.0 * nu).exp();
    for (q, &v) in final_state.iter().enumerate() {
        assert!((v - omega0[q] * factor).norm() < 1e-10, "mode {}", q);
    }
}

#[test]
fn taylor_green_energy_and_enstrophy_match_the_analytic_values() {
    let n = 16;
    let omega = taylor_green_spectral(n);
    assert!((energy(n, &omega) - 0.25).abs() < 1e-12);
    assert!((enstrophy(n, &omega) - 0.5).abs() < 1e-12);
}

/// Measured inviscid energy drift (fluid.design.toml: the bound comes from
/// measurement, not assumption). The verification run --nu 0 --n 32
/// --dt 0.01 --t-final 0.5 (runs/inviscid-drift) showed a relative drift of
/// exactly 0.0; the bound below sits at float-noise scale with margin.
#[test]
fn inviscid_energy_drift_stays_at_measured_bound() {
    let n = 32;
    let omega0 = taylor_green_spectral(n);
    let e0 = energy(n, &omega0);
    let final_state = advance(&RK4, |s| rate(n, 0.0, s), 0.01, 0.5, &omega0)
        .last()
        .expect("non-empty")
        .1
        .clone();
    let drift = (energy(n, &final_state) - e0) / e0;
    assert!(drift.abs() < 1e-14, "relative energy drift {}", drift);
}
