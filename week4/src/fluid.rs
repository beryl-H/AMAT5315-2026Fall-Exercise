//! Pseudospectral solver of the two-dimensional incompressible vorticity
//! equation omega_t + u . grad(omega) = nu laplacian(omega) on the periodic
//! [0, 2pi)^2 grid (fluid.design.toml).
//!
//! Stream function (pairing A, confirmed against the lecture):
//! u = (d psi/dy, -d psi/dx) with omega = -laplacian(psi), i.e.
//! omega_hat = +kappa^2 psi_hat, so psi_hat = +omega_hat / kappa^2 at every
//! nonzero mode; psi_hat[0, 0] = 0 and the mean vorticity is carried
//! untouched (it produces no velocity). The nonlinear term is formed in
//! physical space and dealiased by the two-thirds rule, as is the vorticity
//! itself.

use crate::field::{
    dealias, deriv_x, deriv_y, dft2, idft2, is_nyquist, laplacian, wavenumber,
};
use crate::scalar::Scalar;
use num_complex::Complex;

/// Spectral vorticity from a physical velocity field:
/// omega = d/dx v - d/dy u, so omega_hat = i kx v_hat - i ky u_hat, with the
/// per-axis no-travel rule on the two first derivatives.
pub fn vorticity_from_velocity(
    n: usize,
    u: &[f64],
    v: &[f64],
) -> Vec<Complex<f64>> {
    let u_hat = dft2(n, u);
    let v_hat = dft2(n, v);
    (0..n * n)
        .map(|q| {
            let (ky, kx) = (q / n, q % n);
            let mut w = Complex::new(0.0, 0.0);
            if !is_nyquist(kx, n) {
                w += Complex::new(0.0, wavenumber(kx, n)) * v_hat[q];
            }
            if !is_nyquist(ky, n) {
                w -= Complex::new(0.0, wavenumber(ky, n)) * u_hat[q];
            }
            w
        })
        .collect()
}

/// Stream function from vorticity: psi_hat = -omega_hat / kappa^2 at every
/// nonzero mode, zero at the mean mode.
pub fn stream_function(n: usize, omega_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
    omega_hat
        .iter()
        .enumerate()
        .map(|(q, &w)| {
            if q == 0 {
                return Complex::new(0.0, 0.0);
            }
            let (ky, kx) = (q / n, q % n);
            let kappa2 = wavenumber(kx, n).powi(2) + wavenumber(ky, n).powi(2);
            w / kappa2
        })
        .collect()
}

/// Spectral velocity from the stream function:
/// u_x = i ky psi (y-Nyquist index: zero), u_y = -i kx psi (x-Nyquist: zero).
pub fn velocity(
    n: usize,
    psi_hat: &[Complex<f64>],
) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let mut u_x = vec![Complex::new(0.0, 0.0); n * n];
    let mut u_y = vec![Complex::new(0.0, 0.0); n * n];
    for q in 0..n * n {
        let (ky, kx) = (q / n, q % n);
        let psi = psi_hat[q];
        // d/dy of psi: i ky psi, zero at the y-Nyquist index
        u_x[q] = if crate::field::is_nyquist(ky, n) {
            Complex::new(0.0, 0.0)
        } else {
            Complex::new(0.0, wavenumber(ky, n)) * psi
        };
        // -d/dx of psi: -i kx psi, zero at the x-Nyquist index
        u_y[q] = if crate::field::is_nyquist(kx, n) {
            Complex::new(0.0, 0.0)
        } else {
            Complex::new(0.0, -wavenumber(kx, n)) * psi
        };
    }
    (u_x, u_y)
}

/// Nonlinear term N(omega) = -u . grad(omega): gradient and velocity in
/// spectral space, transformed to physical space, products formed there,
/// transformed back, dealiased, negated.
pub fn nonlinear_term(n: usize, omega_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let psi = stream_function(n, omega_hat);
    let (u_x, u_y) = velocity(n, &psi);
    let omega_x = deriv_x(n, omega_hat);
    let omega_y = deriv_y(n, omega_hat);
    let u_x_p = idft2(n, &u_x);
    let u_y_p = idft2(n, &u_y);
    let w_x_p = idft2(n, &omega_x);
    let w_y_p = idft2(n, &omega_y);
    let product: Vec<f64> = (0..n * n)
        .map(|q| u_x_p[q] * w_x_p[q] + u_y_p[q] * w_y_p[q])
        .collect();
    let spec = dft2(n, &product);
    dealias(n, &spec)
        .iter()
        .map(|&v| -v)
        .collect()
}

/// Full rate of the vorticity equation: nonlinear term + nu laplacian(omega).
pub fn rate(n: usize, nu: f64, omega_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let nl = nonlinear_term(n, omega_hat);
    let diff = laplacian(n, omega_hat);
    nl.iter().zip(diff).map(|(a, b)| a + b * nu).collect()
}

/// Domain-mean kinetic energy 1/2 <u.u> = 1/2 sum_{k != 0} |omega_hat|^2 /
/// kappa^2 (Parseval for the stated DFT convention).
pub fn energy(n: usize, omega_hat: &[Complex<f64>]) -> f64 {
    omega_hat
        .iter()
        .enumerate()
        .filter(|(q, _)| *q != 0)
        .map(|(q, &w)| {
            let (ky, kx) = (q / n, q % n);
            let kappa2 = wavenumber(kx, n).powi(2) + wavenumber(ky, n).powi(2);
            w.norm_sqr() / kappa2
        })
        .sum::<f64>()
        * 0.5
}

/// Domain-mean enstrophy 1/2 <omega^2> = 1/2 sum over ALL modes of
/// |omega_hat|^2 (the mean mode included).
pub fn enstrophy(_n: usize, omega_hat: &[Complex<f64>]) -> f64 {
    omega_hat.iter().map(|w| w.norm_sqr()).sum::<f64>() * 0.5
}

// keep Scalar in scope for the blanket bounds used by callers of `advance`
#[allow(unused_imports)]
use Scalar as _;
