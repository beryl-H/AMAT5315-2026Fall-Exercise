//! Accuracy comparison on one lap of a periodic Gaussian pulse.
//!
//! Initial condition: u(x,0) = exp(-(x - pi/2)^2 / (2 sigma^2)), sigma = 0.25,
//! periodic on [0, 2pi), n = 64, c = 1, nu = 0.002, integrated to t = 2pi
//! (exactly one lap) with:
//!   1. RK4, dt = 0.02,  Fourier derivatives
//!   2. RK4, dt = 0.02,  centred finite differences
//!   3. forward Euler, dt = 0.005, Fourier derivatives
//! Reference: the exact solution -- the Gaussian advected one full lap back
//! to x = pi/2 and diffused to sigma_t = sqrt(sigma^2 + 2 nu t), periodized.
//!
//! Usage: cargo run --bin run_accuracy -- evidence/line_accuracy.json
//! (run from `week4/`). Prints the maximum error of each run.

use advdiff::FourierAdvDiff;

#[cfg(test)]
mod tests {
    use super::{exact, initial, max_err, N};

    #[test]
    fn exact_at_t_zero_is_the_initial_pulse() {
        // sigma_t = sigma and the pulse is effectively compact: the
        // periodized sum reduces to the single pulse.
        let u0 = initial(N);
        let e0 = exact(N, 0.0);
        for (a, b) in u0.iter().zip(&e0) {
            assert!((a - b).abs() < 1e-9, "{} vs {}", a, b);
        }
    }

    #[test]
    fn max_err_of_identical_profiles_is_zero() {
        let v = vec![0.3_f64; N];
        assert_eq!(max_err(&v, &v), 0.0);
    }
}
