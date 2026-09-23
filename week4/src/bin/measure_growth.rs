//! Measures one-step growth factors |y1| of the three library integrators
//! on a grid of z = lambda*h (here h = 1) by actually advancing y' = lambda*y
//! one step from y0 = 1 with `advdiff`'s `Integrator::step`.
//!
//! Usage: cargo run --bin measure_growth -- evidence/line_growth.json
//! (run from `week4/`).

use advdiff::{Euler, Integrator, Midpoint, RK4};
use num_complex::Complex;

#[cfg(test)]
mod tests {
    use super::measured_growth;
    use advdiff::{Euler, Midpoint, RK4};
    use num_complex::Complex;

    #[test]
    fn measured_one_step_growth_matches_stability_polynomial_on_real_axis() {
        // z = -1: Euler |1+z| = 0, midpoint |1+z+z^2/2| = 1/2,
        // RK4 |1+z+z^2/2+z^3/6+z^4/24| = |3/8|.
        let z = Complex::new(-1.0, 0.0);
        assert!(measured_growth(&Euler, z) < 1e-14);
        assert!((measured_growth(&Midpoint, z) - 0.5).abs() < 1e-14);
        let rk4_expected = (1.0 - 1.0 + 0.5 - 1.0 / 6.0 + 1.0 / 24.0).abs();
        assert!((measured_growth(&RK4, z) - rk4_expected).abs() < 1e-14);
    }

    #[test]
    fn zero_eigenvalue_gives_unit_growth_for_all_integrators() {
        let z = Complex::new(0.0, 0.0);
        assert!((measured_growth(&Euler, z) - 1.0).abs() < 1e-14);
        assert!((measured_growth(&Midpoint, z) - 1.0).abs() < 1e-14);
        assert!((measured_growth(&RK4, z) - 1.0).abs() < 1e-14);
    }
}
