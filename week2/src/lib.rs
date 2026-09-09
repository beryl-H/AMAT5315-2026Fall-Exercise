//! Library for the `md` molecular-dynamics crate.

mod system;
pub use system::System;

/// The greeting returned by this crate.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// The Lennard-Jones pair energy in reduced units (sigma = 1, epsilon = 1):
/// U(r) = 4 (r^-12 - r^-6), zero at r = 1, minimum -1 at r0 = 2^(1/6).
pub fn energy(r: f64) -> f64 {
    let r2 = r * r; // r^-6 = (r^-2)^3
    let r6 = 1.0 / (r2 * r2 * r2);
    4.0 * r6 * (r6 - 1.0)
}

/// The pair force magnitude F(r) = -dU/dr = 24/r (2 r^-12 - r^-6):
/// positive repels (r < r0), zero at r0 = 2^(1/6), negative attracts (r > r0).
pub fn force(r: f64) -> f64 {
    let r2 = r * r;
    let r6 = 1.0 / (r2 * r2 * r2);
    let r12 = r6 * r6;
    24.0 / r * (2.0 * r12 - r6)
}

#[cfg(test)]
mod tests {
    use super::{dimer, energy, force, greeting, run, Euler, VelocityVerlet};

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn energy_well_depth_is_minus_one() {
        // U is minimal at r0 = 2^(1/6), where U(r0) = -1.
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        assert!((energy(r0) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn force_matches_the_energy_slope() {
        // F(r) = -dU/dr, checked against a central difference with h = 1e-5.
        // The separations straddle r0 ≈ 1.122: 1.0 repels, 1.5 attracts.
        let h = 1e-5;
        for r in [1.0, 1.5] {
            let df = -(energy(r + h) - energy(r - h)) / (2.0 * h);
            let tol = 1e-6 * force(r).abs().max(1.0);
            assert!((force(r) - df).abs() < tol, "r = {r}");
        }
    }

    #[test]
    fn dimer_conservation() {
        // Same initial state and dt for both methods; only the integrator differs.
        let dt = 0.01;

        // Velocity-Verlet, 500 steps: error bounded below 1e-3.
        let mut system = dimer();
        let verlet = run(&VelocityVerlet, &mut system, dt, 500);
        let max_err = verlet.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
        assert!(max_err < 1e-3, "Verlet 500-step max error {max_err} >= 1e-3");

        // Euler, 500 steps: final error exceeds 0.5.
        let mut system = dimer();
        let euler = run(&Euler, &mut system, dt, 500);
        assert!(
            *euler.last().unwrap() > 0.5,
            "Euler final error {} <= 0.5",
            euler.last().unwrap()
        );

        // Velocity-Verlet alone, 5000 steps: same bound holds.
        let mut system = dimer();
        let verlet = run(&VelocityVerlet, &mut system, dt, 5000);
        let max_err = verlet.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
        assert!(max_err < 1e-3, "Verlet 5000-step max error {max_err} >= 1e-3");
    }
}
