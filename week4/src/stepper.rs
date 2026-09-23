//! Driver that calls an `Integrator` repeatedly from t = 0 to `t_final`,
//! taking full steps of `dt` plus one shorter final step so the final time
//! is hit exactly. Returns `(t, state)` after every step.

use crate::integrators::Integrator;
use crate::scalar::Scalar;

pub fn advance<I, F, S>(
    integrator: &I,
    f: F,
    dt: f64,
    t_final: f64,
    y0: &[S],
) -> Vec<(f64, Vec<S>)>
where
    I: Integrator,
    F: Fn(&[S]) -> Vec<S>,
    S: Scalar,
{
    let mut y = y0.to_vec();
    let mut t = 0.0;
    let mut history = Vec::new();
    while t < t_final - 1e-12 {
        let step = dt.min(t_final - t);
        y = integrator.step(&f, step, &y);
        t += step;
        history.push((t, y.clone()));
    }
    history
}

#[cfg(test)]
mod tests {
    use super::advance;
    use crate::{Euler, Midpoint};
    use num_complex::Complex;

    fn decay(y: &[f64]) -> Vec<f64> {
        y.iter().map(|v| -v).collect()
    }

    #[test]
    fn midpoint_driver_accumulates_two_steps() {
        let dt = 0.1_f64;
        let history = advance(&Midpoint, decay, dt, 0.2, &[1.0]);
        assert_eq!(history.len(), 2);
        // For dy/dt = -y the midpoint one-step factor is (1 - dt + dt^2/2),
        // applied twice by the driver.
        let factor = 1.0 - dt + dt * dt / 2.0;
        assert!((history[1].1[0] - factor * factor).abs() < 1e-12);
    }

    #[test]
    fn complex_state_accumulates() {
        let history = advance(
            &Euler,
            |s: &[Complex<f64>]| s.iter().map(|v| -*v).collect(),
            0.05_f64,
            0.1,
            &[Complex::new(2.0, -1.0)],
        );
        assert!((history[1].1[0] - Complex::new(1.9, -0.95)).norm() < 1e-12);
    }
}
