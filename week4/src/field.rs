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

/// Hermitian single mode: amplitude `amp` at (kx, ky) and conj(amp) at the
/// mirrored index, so the physical field 2 Re(amp e^{i(kx x + ky y)}) is real.
/// Panics when (kx, ky) is outside the two-thirds band.
pub fn hermitian_mode(n: usize, kx: usize, ky: usize, amp: Complex<f64>) -> Vec<Complex<f64>> {
    let band = dealias_band(n);
    assert!(
        kx <= band && ky <= band,
        "kx, ky must satisfy the two-thirds rule (<= {band})"
    );
    let mut u_hat = vec![Complex::new(0.0, 0.0); n * n];
    u_hat[ky * n + kx] = amp;
    let (mx, my) = ((n - kx) % n, (n - ky) % n);
    if (mx, my) != (kx, ky) {
        u_hat[my * n + mx] = amp.conj();
    }
    u_hat
}

/// Taylor-Green vortex omega = 2 sin(x) sin(y), dealiased spectral state.
pub fn taylor_green_spectral(n: usize) -> Vec<Complex<f64>> {
    let u: Vec<f64> = (0..n * n)
        .map(|q| {
            let (i, j) = (q % n, q / n);
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            2.0 * x.sin() * y.sin()
        })
        .collect();
    dealias(n, &dft2(n, &u))
}

/// Linear 2D advection-diffusion rate:
/// u_t = -cx u_x - cy u_y + nu laplacian(u), with the per-axis no-travel rule
/// on the two advection parts and the full kappa^2 on the diffusion.
pub fn advdiff_rate(
    n: usize,
    cx: f64,
    cy: f64,
    nu: f64,
    u_hat: &[Complex<f64>],
) -> Vec<Complex<f64>> {
    u_hat
        .iter()
        .enumerate()
        .map(|(q, &u)| {
            let (ky, kx) = (q / n, q % n);
            let mut coeff = Complex::new(-nu * (wavenumber(kx, n).powi(2) + wavenumber(ky, n).powi(2)), 0.0);
            if !is_nyquist(kx, n) {
                coeff -= Complex::new(0.0, cx * wavenumber(kx, n));
            }
            if !is_nyquist(ky, n) {
                coeff -= Complex::new(0.0, cy * wavenumber(ky, n));
            }
            coeff * u
        })
        .collect()
}

/// Exact per-mode exponential factor of the linear advection-diffusion
/// solution after time t: exp((-i (cx kx + cy ky) - nu (kx^2 + ky^2)) t).
pub fn advdiff_exponential(
    n: usize,
    kx: usize,
    ky: usize,
    cx: f64,
    cy: f64,
    nu: f64,
    t: f64,
) -> Complex<f64> {
    let lambda = Complex::new(
        -nu * (wavenumber(kx, n).powi(2) + wavenumber(ky, n).powi(2)),
        -(cx * wavenumber(kx, n) + cy * wavenumber(ky, n)),
    );
    (lambda * t).exp()
}
