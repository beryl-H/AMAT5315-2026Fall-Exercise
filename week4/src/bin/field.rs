//! Field verification binary: linear 2D advection-diffusion of a single
//! Hermitian Fourier mode (field.design.toml).
//!
//! Usage: field --n 32 --kx 1 --ky 2 --cx 1 --cy 1 --method rk4 --nu 0.05
//!            --t-final 1.0 --dt 0.01 --out runs/field-single-mode
//! (run from `week4/`).

#[cfg(test)]
mod tests {
    use super::exact_solution;
    use advdiff::field::{advdiff_exponential, hermitian_mode};
    use num_complex::Complex;

    #[test]
    fn exact_solution_is_the_initial_mode_times_the_exponential() {
        let n = 32;
        let u0 = exact_solution(n, 1, 2, 1.0, 1.0, 0.05, 0.0);
        // at t = 0 the exact solution is the initial Hermitian mode
        assert_eq!(u0, hermitian_mode(n, 1, 2, Complex::new(1.0, 0.0)));
        // at t = 1 every mode carries the exponential factor
        let u1 = exact_solution(n, 1, 2, 1.0, 1.0, 0.05, 1.0);
        let factor = advdiff_exponential(n, 1, 2, 1.0, 1.0, 0.05, 1.0);
        let q = 2 * n + 1; // [ky*n + kx]
        assert!((u1[q] - u0[q] * factor).norm() < 1e-14);
    }
}
