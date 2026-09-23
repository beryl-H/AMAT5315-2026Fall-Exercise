//! Contract tests for the two-dimensional field machinery (field.design.toml).

use advdiff::field::{
    advdiff_exponential, advdiff_rate, dealias, dealias_band, dft2, deriv_x, deriv_y,
    hermitian_mode, idft2, is_nyquist, laplacian, taylor_green_spectral, wavenumber,
};
use advdiff::{advance, RK4};
use num_complex::Complex;
use std::f64::consts::TAU;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

#[test]
fn wavenumber_and_nyquist_follow_the_one_dimensional_rule() {
    let n = 32;
    assert_eq!(wavenumber(0, n), 0.0);
    assert_eq!(wavenumber(3, n), 3.0);
    assert_eq!(wavenumber(16, n), 16.0);
    assert_eq!(wavenumber(17, n), -15.0);
    assert_eq!(wavenumber(31, n), -1.0);
    assert!(is_nyquist(16, n));
    assert!(!is_nyquist(15, n));
    assert!(!is_nyquist(17, n));
    assert!(!is_nyquist(3, n));
}

#[test]
fn dft2_idft2_round_trip_recovers_a_real_field() {
    let n = 16;
    let u: Vec<f64> = (0..n * n)
        .map(|q| {
            let i = q % n;
            let j = q / n;
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            (3.0 * x).sin() * (2.0 * y).cos() + 0.7
        })
        .collect();
    let back = idft2(n, &dft2(n, &u));
    for (a, b) in back.iter().zip(&u) {
        assert!((a - b).abs() < 1e-12, "{} vs {}", a, b);
    }
}

#[test]
fn plane_wave_lands_on_a_hermitian_mode_pair() {
    let n = 16;
    let u: Vec<f64> = (0..n * n)
        .map(|q| {
            let i = q % n;
            let x = TAU * i as f64 / n as f64;
            (2.0 * x).cos()
        })
        .collect();
    let u_hat = dft2(n, &u);
    // spectral index [ky*n + kx]; the plane wave sits at ky = 0, kx = 2 and 14
    assert!((u_hat[0 * n + 2] - c(0.5, 0.0)).norm() < 1e-12);
    assert!((u_hat[0 * n + 14] - c(0.5, 0.0)).norm() < 1e-12);
    for (q, &v) in u_hat.iter().enumerate() {
        if q != 0 * n + 2 && q != 0 * n + 14 {
            assert!(v.norm() < 1e-12, "mode {} = {}", q, v);
        }
    }
}

#[test]
fn derivatives_of_an_in_band_single_mode() {
    let n = 32;
    let mut u_hat = vec![c(0.0, 0.0); n * n];
    let amp = c(1.0, 1.0);
    u_hat[2 * n + 1] = amp; // ky = 2, kx = 1
    let dx = deriv_x(n, &u_hat);
    assert!((dx[2 * n + 1] - c(0.0, 1.0) * amp).norm() < 1e-14); // i kx, kx = 1
    let dy = deriv_y(n, &u_hat);
    assert!((dy[2 * n + 1] - c(0.0, 2.0) * amp).norm() < 1e-14); // i ky, ky = 2
    let lap = laplacian(n, &u_hat);
    assert!((lap[2 * n + 1] - c(-5.0, 0.0) * amp).norm() < 1e-14); // -(1 + 4)
}

#[test]
fn nyquist_no_travel_rule_is_independent_per_axis() {
    let n = 32;
    let amp = c(1.0, 0.0);
    // x-Nyquist index 16 with y-index 3: d/dx zero, d/dy = i * 3.
    let mut u_hat = vec![c(0.0, 0.0); n * n];
    u_hat[3 * n + 16] = amp;
    assert!(is_nyquist(16, n));
    assert!(deriv_x(n, &u_hat)[3 * n + 16].norm() < 1e-14);
    assert!((deriv_y(n, &u_hat)[3 * n + 16] - c(0.0, 3.0) * amp).norm() < 1e-14);
    // y-Nyquist index 16 with x-index 3: d/dy zero, d/dx = i * 3.
    let mut v_hat = vec![c(0.0, 0.0); n * n];
    v_hat[16 * n + 3] = amp;
    assert!(deriv_y(n, &v_hat)[16 * n + 3].norm() < 1e-14);
    assert!((deriv_x(n, &v_hat)[16 * n + 3] - c(0.0, 3.0) * amp).norm() < 1e-14);
    // the laplacian is not a travel operator: full kappa^2 = 9 + 256
    assert!(
        (laplacian(n, &v_hat)[16 * n + 3] - c(-265.0, 0.0) * amp).norm() < 1e-14
    );
}

#[test]
fn dealias_enforces_the_two_thirds_band() {
    let n = 32;
    assert_eq!(dealias_band(n), 10);
    let mut u_hat = vec![c(0.0, 0.0); n * n];
    u_hat[0 * n + 10] = c(1.0, 0.0); // |kx| = 10, boundary, kept
    u_hat[0 * n + 11] = c(1.0, 0.0); // |kx| = 11, zeroed
    u_hat[11 * n + 0] = c(1.0, 0.0); // |ky| = 11, zeroed
    u_hat[3 * n + 21] = c(1.0, 0.0); // kx index 21 -> wavenumber -11, zeroed
    u_hat[5 * n + 5] = c(1.0, 0.0); // kept
    u_hat[0 * n + 0] = c(2.0, 0.0); // mean, kept
    let d = dealias(n, &u_hat);
    assert!((d[0 * n + 10] - c(1.0, 0.0)).norm() < 1e-14);
    assert_eq!(d[0 * n + 11], c(0.0, 0.0));
    assert_eq!(d[11 * n + 0], c(0.0, 0.0));
    assert_eq!(d[3 * n + 21], c(0.0, 0.0));
    assert!((d[5 * n + 5] - c(1.0, 0.0)).norm() < 1e-14);
    assert!((d[0 * n + 0] - c(2.0, 0.0)).norm() < 1e-14);
}

#[test]
fn hermitian_mode_generator_is_dealiased_and_matches_its_formula() {
    let n = 32;
    let u_hat = hermitian_mode(n, 3, 5, c(1.0, 0.0));
    for (q, &v) in u_hat.iter().enumerate() {
        let in_pair = q == 5 * n + 3 || q == (n - 5) * n + (n - 3);
        if !in_pair {
            assert_eq!(v, c(0.0, 0.0), "mode {} not dealiased", q);
        }
    }
    assert_eq!(u_hat[5 * n + 3], c(1.0, 0.0));
    assert_eq!(u_hat[(n - 5) * n + (n - 3)], c(1.0, 0.0));
    let u = idft2(n, &u_hat);
    // physical field: 2 cos(3 x + 5 y)
    for j in 0..n {
        for i in 0..n {
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            let expected = 2.0 * (3.0 * x + 5.0 * y).cos();
            assert!((u[j * n + i] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn linear_rate_advances_toward_the_exact_exponential() {
    let n = 32;
    let (kx, ky) = (2usize, 3usize);
    let (cx, cy, nu) = (1.0, 0.5, 0.05);
    let u_hat0 = hermitian_mode(n, kx, ky, c(1.0, 0.0));
    let final_state = advance(
        &RK4,
        |s| advdiff_rate(n, cx, cy, nu, s),
        0.01,
        1.0,
        &u_hat0,
    )
    .last()
    .expect("non-empty")
    .1
    .clone();
    let factor = advdiff_exponential(n, kx, ky, cx, cy, nu, 1.0);
    assert!(
        (final_state[ky * n + kx] - u_hat0[ky * n + kx] * factor).norm() < 1e-10,
        "got {}",
        final_state[ky * n + kx]
    );
    // Hermitian partner: conjugate amplitude, conjugate factor
    assert!(
        (final_state[(n - ky) * n + (n - kx)] - u_hat0[(n - ky) * n + (n - kx)] * factor.conj())
            .norm()
            < 1e-10
    );
}

#[test]
fn taylor_green_generator_matches_its_formula() {
    let n = 16;
    let u = idft2(n, &taylor_green_spectral(n));
    for j in 0..n {
        for i in 0..n {
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            let expected = 2.0 * x.sin() * y.sin();
            assert!((u[j * n + i] - expected).abs() < 1e-12);
        }
    }
}
