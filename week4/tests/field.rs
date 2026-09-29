//! Contract tests for the two-dimensional field machinery (field.design.toml).

use advdiff::field::{
    dealias, dealias_band, dft2, deriv_x, deriv_y, idft2, is_nyquist, laplacian,
    random_vorticity, taylor_green_spectral, taylor_green_velocity, wavenumber,
};
use advdiff::fluid::{energy, vorticity_from_velocity};
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
fn taylor_green_vorticity_is_minus_two_cos_x_cos_y() {
    let n = 16;
    let u = idft2(n, &taylor_green_spectral(n));
    for j in 0..n {
        for i in 0..n {
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            let expected = -2.0 * x.cos() * y.cos();
            assert!((u[j * n + i] - expected).abs() < 1e-12);
        }
    }
    // the four modes (+-1, +-1) all carry -1/2; nothing else is present
    let u_hat = taylor_green_spectral(n);
    for (q, &v) in u_hat.iter().enumerate() {
        let (ky, kx) = (q / n, q % n);
        let is_corner = (kx == 1 || kx == n - 1) && (ky == 1 || ky == n - 1);
        if is_corner {
            assert!((v - c(-0.5, 0.0)).norm() < 1e-12, "mode {}", q);
        } else {
            assert!(v.norm() < 1e-12, "mode {} = {}", q, v);
        }
    }
}

#[test]
fn taylor_green_velocity_matches_the_course_formula() {
    let n = 16;
    let (u, v) = taylor_green_velocity(n, 0.0, 0.0);
    for j in 0..n {
        for i in 0..n {
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            assert!((u[j * n + i] - x.cos() * y.sin()).abs() < 1e-12);
            assert!((v[j * n + i] + x.sin() * y.cos()).abs() < 1e-12);
        }
    }
    // decay: at time t both components carry exp(-2 nu t)
    let (u1, v1) = taylor_green_velocity(n, 1.0, 0.1);
    let decay = (-0.2_f64).exp();
    assert!((u1[5 * n + 3] - u[5 * n + 3] * decay).abs() < 1e-12);
    assert!((v1[5 * n + 3] - v[5 * n + 3] * decay).abs() < 1e-12);
}

#[test]
fn random_vorticity_has_energy_exactly_one_half() {
    let n = 64;
    let omega = random_vorticity(n, 2026, 2.0, 6.0);
    assert!((energy(n, &omega) - 0.5).abs() < 1e-12);
}

#[test]
fn random_vorticity_is_deterministic_per_seed() {
    let a = random_vorticity(64, 2026, 2.0, 6.0);
    let b = random_vorticity(64, 2026, 2.0, 6.0);
    assert_eq!(a, b);
    let c = random_vorticity(64, 2027, 2.0, 6.0);
    assert_ne!(a, c);
}

#[test]
fn random_phases_are_independent_of_n() {
    // the phase drawn for a given mode (kx, ky) must not depend on n
    let small = random_vorticity(64, 2026, 2.0, 6.0);
    let large = random_vorticity(96, 2026, 2.0, 6.0);
    for ky in -6i64..=6 {
        for kx in -6i64..=6 {
            let r2 = (kx * kx + ky * ky) as f64;
            if r2 < 4.0 || r2 > 36.0 || (kx == 0 && ky == 0) {
                continue;
            }
            let at = |n: usize, kx: i64, ky: i64| {
                (ky.rem_euclid(n as i64) as usize) * n + (kx.rem_euclid(n as i64) as usize)
            };
            assert_eq!(
                small[at(64, kx, ky)],
                large[at(96, kx, ky)],
                "mode ({}, {})",
                kx,
                ky
            );
        }
    }
}

#[test]
fn random_vorticity_is_hermitian_and_band_limited() {
    let n = 64;
    let omega = random_vorticity(n, 2026, 2.0, 6.0);
    let at = |kx: i64, ky: i64| {
        (ky.rem_euclid(n as i64) as usize) * n + (kx.rem_euclid(n as i64) as usize)
    };
    for ky in -8i64..=8 {
        for kx in -8i64..=8 {
            if kx == 0 && ky == 0 {
                continue;
            }
            let v = omega[at(kx, ky)];
            assert_eq!(v, omega[at(-kx, -ky)].conj(), "Hermitian ({}, {})", kx, ky);
            let r2 = ((kx * kx + ky * ky) as f64).sqrt();
            if r2 < 2.0 || r2 > 6.0 {
                assert_eq!(v, c(0.0, 0.0), "outside annulus ({}, {})", kx, ky);
            } else {
                assert!((v.norm() - v.norm()).abs() < 1e-14);
            }
        }
    }
    // equal amplitudes inside the annulus
    let amps: Vec<f64> = [(1i64, 3i64), (3, 1), (0, 6), (6, 0), (4, 4)]
        .iter()
        .map(|&(kx, ky)| omega[at(kx, ky)].norm())
        .collect();
    for a in &amps {
        assert!((a - amps[0]).abs() < 1e-12, "unequal amplitudes");
    }
}

#[test]
fn vorticity_from_velocity_recovers_the_taylor_green_spectrum() {
    let n = 16;
    let (u, v) = taylor_green_velocity(n, 0.0, 0.0);
    let omega = vorticity_from_velocity(n, &u, &v);
    // the round trip is exact up to floating-point noise
    for (a, b) in omega.iter().zip(taylor_green_spectral(n)) {
        assert!((a - b).norm() < 1e-12, "got {}, want {}", a, b);
    }
}
