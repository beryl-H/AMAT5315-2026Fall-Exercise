//! Fourier pseudo-spectral rate function for u_t + c u_x = nu u_xx on the
//! periodic grid x_j = 2 pi j / n, j = 0..n-1.
//!
//! Spectral state has length n, indices 0..n-1. Wavenumbers: k_j = j for
//! 0 <= j < n/2; for even n the Nyquist index j = n/2 has wavenumber
//! magnitude n/2 with first-derivative coefficient zero (it does not
//! travel); k_j = j - n for n/2 < j <= n-1. For odd n there is no Nyquist
//! index and the split is at (n-1)/2.

use num_complex::Complex;
use std::f64::consts::TAU;

pub struct FourierAdvDiff {
    pub c: f64,
    pub nu: f64,
    pub n: usize,
}

impl FourierAdvDiff {
    /// Wavenumber of spectral index `j`. For even `n` the Nyquist index
    /// `j = n/2` returns magnitude `n/2` (used only for the diffusion term).
    pub fn wavenumber(&self, j: usize) -> f64 {
        if j <= self.n / 2 {
            j as f64
        } else {
            j as f64 - self.n as f64
        }
    }

    fn is_nyquist(&self, j: usize) -> bool {
        self.n % 2 == 0 && j == self.n / 2
    }

    /// First spectral derivative d/dx. The Nyquist index contributes zero.
    pub fn derivative(&self, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
        assert_eq!(u_hat.len(), self.n, "state length must equal n");
        u_hat.iter()
            .enumerate()
            .map(|(j, uj)| {
                let coeff = if self.is_nyquist(j) {
                    Complex::new(0.0, 0.0)
                } else {
                    Complex::new(0.0, 1.0) * self.wavenumber(j)
                };
                coeff * uj
            })
            .collect()
    }

    /// Rate of change du_hat/dt = (-i c k - nu k^2) u_hat, with the Nyquist
    /// index advecting nothing: rate = -nu (n/2)^2 u_hat there.
    pub fn rate(&self, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
        assert_eq!(u_hat.len(), self.n, "state length must equal n");
        u_hat.iter()
            .enumerate()
            .map(|(j, uj)| {
                let k = self.wavenumber(j);
                let adv = if self.is_nyquist(j) {
                    Complex::new(0.0, 0.0)
                } else {
                    Complex::new(0.0, -self.c * k)
                };
                let diff = Complex::new(-self.nu * k * k, 0.0);
                (adv + diff) * uj
            })
            .collect()
    }
}

/// Discrete Fourier transform, convention u(x) = sum_k u_hat[k] e^{ikx}:
/// u_hat[k] = (1/n) sum_j u_j e^{-i k x_j}.
pub fn dft(u: &[f64]) -> Vec<Complex<f64>> {
    let n = u.len();
    (0..n)
        .map(|k| {
            let mut acc = Complex::new(0.0, 0.0);
            for (j, &uj) in u.iter().enumerate() {
                let x = TAU * j as f64 / n as f64;
                acc += uj * Complex::new(0.0, -(k as f64) * x).exp();
            }
            acc / n as f64
        })
        .collect()
}

/// Inverse discrete Fourier transform back to physical space (real part,
/// exact for input spectra coming from `dft` of a real field).
pub fn idft(u_hat: &[Complex<f64>]) -> Vec<f64> {
    let n = u_hat.len();
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            let mut acc = Complex::new(0.0, 0.0);
            for (k, &uk) in u_hat.iter().enumerate() {
                acc += uk * Complex::new(0.0, (k as f64) * x).exp();
            }
            acc.re
        })
        .collect()
}
