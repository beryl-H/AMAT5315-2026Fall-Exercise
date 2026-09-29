//! Two-dimensional spectral field machinery on the periodic [0, 2pi)^2 grid
//! (field.design.toml).
//!
//! Spectral state: `Vec<Complex<f64>>` of length n*n, row-major, index
//! `[ky * n + kx]`; physical state: `Vec<f64>`, index `[j * n + i]`,
//! x_i = 2 pi i / n (the i axis), y_j = 2 pi j / n (the j axis).
//!
//! Wavenumbers per axis: index for index <= n/2, index - n beyond. The
//! no-travel rule applies independently per axis to the FIRST derivatives
//! (d/dx zero at the x-Nyquist index, d/dy at the y-Nyquist index); the
//! laplacian keeps the full kappa^2. The two-thirds rule keeps only modes
//! with |kx|, |ky| <= floor(n/3) — applied to the field itself and to every
//! product. Scaling convention (SPEC.md): the integrators scale states by
//! real factors only; complex multiplication happens here, in the rate
//! functions.

use num_complex::Complex;
use std::f64::consts::TAU;

/// Wavenumber of spectral index along one axis (same rule as one dimension).
pub fn wavenumber(index: usize, n: usize) -> f64 {
    if index <= n / 2 {
        index as f64
    } else {
        index as f64 - n as f64
    }
}

/// True when `index` is the Nyquist index of an even-length axis.
pub fn is_nyquist(index: usize, n: usize) -> bool {
    n % 2 == 0 && index == n / 2
}

/// Largest |kx| or |ky| kept by the two-thirds rule: floor(n/3).
pub fn dealias_band(n: usize) -> usize {
    n / 3
}

/// Row-column DFT of a real n*n field, 1/n per axis:
/// u_hat[kx, ky] = (1/n^2) sum_{i,j} u[i, j] e^{-i (kx x_i + ky y_j)}.
pub fn dft2(n: usize, u: &[f64]) -> Vec<Complex<f64>> {
    // over x (i axis) for each row j, then over y (j axis) for each column kx
    let mut rows = vec![Complex::new(0.0, 0.0); n * n];
    for j in 0..n {
        for kx in 0..n {
            let mut acc = Complex::new(0.0, 0.0);
            for i in 0..n {
                let x = TAU * i as f64 / n as f64;
                acc += u[j * n + i] * Complex::new(0.0, -(kx as f64) * x).exp();
            }
            rows[j * n + kx] = acc / n as f64;
        }
    }
    let mut out = vec![Complex::new(0.0, 0.0); n * n];
    for ky in 0..n {
        for kx in 0..n {
            let mut acc = Complex::new(0.0, 0.0);
            for j in 0..n {
                let y = TAU * j as f64 / n as f64;
                acc += rows[j * n + kx] * Complex::new(0.0, -(ky as f64) * y).exp();
            }
            out[ky * n + kx] = acc / n as f64;
        }
    }
    out
}

/// Inverse row-column DFT back to physical space (real part; exact for the
/// Hermitian spectra produced by `dft2` of a real field).
pub fn idft2(n: usize, u_hat: &[Complex<f64>]) -> Vec<f64> {
    let mut rows = vec![Complex::new(0.0, 0.0); n * n];
    for j in 0..n {
        for kx in 0..n {
            let mut acc = Complex::new(0.0, 0.0);
            for ky in 0..n {
                let y = TAU * j as f64 / n as f64;
                acc += u_hat[ky * n + kx] * Complex::new(0.0, (ky as f64) * y).exp();
            }
            rows[j * n + kx] = acc;
        }
    }
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let x = TAU * i as f64 / n as f64;
            let mut acc = Complex::new(0.0, 0.0);
            for kx in 0..n {
                acc += rows[j * n + kx] * Complex::new(0.0, (kx as f64) * x).exp();
            }
            out[j * n + i] = acc.re;
        }
    }
    out
}

/// d/dx: multiplies by i kx; zero at the x-Nyquist index (no travel).
pub fn deriv_x(n: usize, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
    u_hat
        .iter()
        .enumerate()
        .map(|(q, &u)| {
            let kx = q % n;
            let coeff = if is_nyquist(kx, n) {
                Complex::new(0.0, 0.0)
            } else {
                Complex::new(0.0, wavenumber(kx, n))
            };
            coeff * u
        })
        .collect()
}

/// d/dy: multiplies by i ky; zero at the y-Nyquist index (no travel).
pub fn deriv_y(n: usize, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
    u_hat
        .iter()
        .enumerate()
        .map(|(q, &u)| {
            let ky = q / n;
            let coeff = if is_nyquist(ky, n) {
                Complex::new(0.0, 0.0)
            } else {
                Complex::new(0.0, wavenumber(ky, n))
            };
            coeff * u
        })
        .collect()
}

/// Laplacian: multiplies by -(kx^2 + ky^2); not a travel operator, so the
/// full kappa^2 applies (moot after dealias: Nyquist modes carry nothing).
pub fn laplacian(n: usize, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
    u_hat
        .iter()
        .enumerate()
        .map(|(q, &u)| {
            let (ky, kx) = (q / n, q % n);
            let kappa2 = wavenumber(kx, n).powi(2) + wavenumber(ky, n).powi(2);
            Complex::new(-kappa2, 0.0) * u
        })
        .collect()
}

/// Two-thirds rule: zero every mode with |kx| > floor(n/3) or |ky| > floor(n/3).
pub fn dealias(n: usize, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let band = dealias_band(n);
    let in_band = |index: usize| -> bool {
        let k = if index <= n / 2 { index } else { n - index };
        k <= band
    };
    u_hat
        .iter()
        .enumerate()
        .map(|(q, &u)| {
            let (ky, kx) = (q / n, q % n);
            if in_band(kx) && in_band(ky) {
                u
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect()
}

/// Course Taylor-Green vorticity omega = -2 cos(x) cos(y), dealiased
/// spectral state (the modes (+-1, +-1) all carry -1/2).
pub fn taylor_green_spectral(n: usize) -> Vec<Complex<f64>> {
    let u: Vec<f64> = (0..n * n)
        .map(|q| {
            let (i, j) = (q % n, q / n);
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            -2.0 * x.cos() * y.cos()
        })
        .collect();
    dealias(n, &dft2(n, &u))
}

/// Course Taylor-Green velocity at time t:
/// u = cos(x) sin(y) exp(-2 nu t), v = -sin(x) cos(y) exp(-2 nu t).
pub fn taylor_green_velocity(n: usize, t: f64, nu: f64) -> (Vec<f64>, Vec<f64>) {
    let decay = (-2.0 * nu * t).exp();
    let u: Vec<f64> = (0..n * n)
        .map(|q| {
            let (i, j) = (q % n, q / n);
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            x.cos() * y.sin() * decay
        })
        .collect();
    let v: Vec<f64> = (0..n * n)
        .map(|q| {
            let (i, j) = (q % n, q / n);
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            -x.sin() * y.cos() * decay
        })
        .collect();
    (u, v)
}

/// Deterministic splitmix64 stream backing the random initial condition.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Random vorticity initial condition (field.design.toml, `random`):
/// vorticity modes of equal amplitude with
/// k_min <= sqrt(kx^2 + ky^2) <= k_max, phases uniform in [0, 2pi) drawn
/// from `seed` in an order independent of n (representatives kx > 0 or
/// kx == 0, ky > 0, enumerated by ascending (kx, ky), each completed
/// Hermitian). The amplitude is set so the delivered field has E(0) = 1/2.
/// Self-conjugate modes (kx, ky in {0, n/2}) lie outside the two-thirds
/// band for kept modes and are excluded.
pub fn random_vorticity(
    n: usize,
    seed: u64,
    k_min: f64,
    k_max: f64,
) -> Vec<Complex<f64>> {
    assert!(0.0 <= k_min && k_min <= k_max, "k range must satisfy 0 <= k-min <= k-max");
    let radius = k_max as i64;
    let mut rng = SplitMix64(seed);
    let mut reps: Vec<(i64, i64)> = Vec::new();
    for kx in -radius..=radius {
        for ky in -radius..=radius {
            if kx == 0 && ky == 0 {
                continue;
            }
            if !(kx > 0 || (kx == 0 && ky > 0)) {
                continue;
            }
            let r2 = (kx * kx + ky * ky) as f64;
            if k_min * k_min <= r2 && r2 <= k_max * k_max {
                reps.push((kx, ky));
            }
        }
    }
    assert!(!reps.is_empty(), "the annulus contains no modes");
    let inv_kappa2_sum: f64 = reps
        .iter()
        .map(|(kx, ky)| {
            let kappa2 = (kx * kx + ky * ky) as f64;
            1.0 / kappa2
        })
        .sum();
    let amp = (0.5 / inv_kappa2_sum).sqrt();
    let mut omega = vec![Complex::new(0.0, 0.0); n * n];
    for &(kx, ky) in &reps {
        let theta = TAU * rng.next_unit();
        let mode = amp * Complex::new(0.0, theta).exp();
        let ix = kx.rem_euclid(n as i64) as usize;
        let iy = ky.rem_euclid(n as i64) as usize;
        omega[iy * n + ix] = mode;
        let (mx, my) = ((n - ix) % n, (n - iy) % n);
        if (mx, my) != (ix, iy) {
            omega[my * n + mx] = mode.conj();
        }
    }
    dealias(n, &omega)
}
