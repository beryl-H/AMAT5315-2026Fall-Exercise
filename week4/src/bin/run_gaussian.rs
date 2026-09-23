//! Integrates the Gaussian-pulse initial condition of the advection-diffusion
//! line to t = 6 with the library RK4 and records space-time snapshots.
//!
//! Usage: cargo run --bin run_gaussian -- evidence/line_evolution.json
//! (run from `week4/`).

use advdiff::FourierAdvDiff;
use num_complex::Complex;

#[cfg(test)]
mod tests {
    use super::advance_spectral;
    use advdiff::FourierAdvDiff;
    use num_complex::Complex;

    fn scheme() -> FourierAdvDiff {
        FourierAdvDiff { c: 1.0, nu: 0.05, n: 16 }
    }

    fn single_mode(n: usize, k: usize) -> Vec<Complex<f64>> {
        let mut u_hat = vec![Complex::new(0.0, 0.0); n];
        u_hat[k] = Complex::new(1.0, 0.0);
        u_hat
    }

    #[test]
    fn runner_lands_exactly_on_t_final_with_full_plus_partial_steps() {
        // dt = 0.3, T = 1: three full steps and one partial step of 0.1.
        let s = scheme();
        let history = advance_spectral(&s, &single_mode(16, 2), 0.3, 1.0);
        let times: Vec<f64> = history.iter().map(|(t, _)| *t).collect();
        assert_eq!(times, vec![0.3, 0.6, 0.9, 1.0]);
    }

    #[test]
    fn runner_matches_exact_single_mode_evolution_to_t_final() {
        // Exact: u_hat_k(T) = u_hat_k(0) * exp((-i c k - nu k^2) T).
        let s = scheme();
        let k = 2.0_f64;
        let history = advance_spectral(&s, &single_mode(16, 2), 0.3, 1.0);
        let (t, final_state) = history.last().expect("non-empty history");
        assert_eq!(*t, 1.0);
        let lambda = Complex::new(-s.nu * k * k, -s.c * k);
        let expected = (lambda * t).exp();
        assert!(
            (final_state[2] - expected).norm() < 1e-3,
            "got {}, want {}",
            final_state[2],
            expected
        );
    }
}
