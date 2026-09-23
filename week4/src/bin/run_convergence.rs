//! Convergence sweep on one lap-free interval: Gaussian pulse
//! (sigma = 0.35, centred at x = pi/2, n = 64, c = 1, nu = 0.05) evolved to
//! t = 1 with Fourier derivatives by forward Euler, explicit midpoint,
//! classical RK4, and equal-weight RK4, each at dt = 0.02, 0.01, 0.005,
//! 0.0025. Reference: the exact solution of the semi-discrete system,
//! u_hat_k(T) = u_hat_k(0) exp(lambda_k T) with the scheme's own Nyquist rule
//! (no travel at k = n/2), so the measured error is purely temporal.
//!
//! Usage: cargo run --bin run_convergence -- evidence/line_convergence.json
//! (run from `week4/`). Prints the max error of every method/dt pair.

use advdiff::FourierAdvDiff;

#[cfg(test)]
mod tests {
    use super::exact_semi;
    use advdiff::{advance, dft, FourierAdvDiff, RK4};
    use num_complex::Complex;

    #[test]
    fn exact_semi_matches_fine_rk4_evolution_of_a_single_mode() {
        let scheme = FourierAdvDiff { c: 1.0, nu: 0.05, n: 16 };
        let mut u_hat0 = vec![Complex::new(0.0, 0.0); 16];
        u_hat0[2] = Complex::new(1.0, -0.5);
        let reference = exact_semi(&scheme, &u_hat0, 1.0);
        // RK4 with a very fine step must reproduce the exponential reference.
        let fine = advance(&RK4, |s| scheme.rate(s), 0.001, 1.0, &u_hat0);
        let final_state = fine.last().expect("non-empty").1;
        assert!(
            (final_state[2] - reference[2]).norm() < 1e-10,
            "got {}, want {}",
            final_state[2],
            reference[2]
        );
    }

    #[test]
    fn exact_semi_damps_nyquist_without_travel() {
        let scheme = FourierAdvDiff { c: 1.0, nu: 0.05, n: 16 };
        let mut u_hat0 = vec![Complex::new(0.0, 0.0); 16];
        u_hat0[8] = Complex::new(1.5, 2.0); // Nyquist index of n = 16
        let reference = exact_semi(&scheme, &u_hat0, 2.0);
        let factor = reference[8] / u_hat0[8];
        assert_eq!(factor.im, 0.0, "Nyquist mode must not travel");
        let expected = (-scheme.nu * 64.0 * 2.0).exp();
        assert!((factor.re - expected).abs() < 1e-12);
    }
}
