//! Library for the `md` molecular-dynamics crate.

pub mod rng;

mod integrator;
mod system;
pub use integrator::{Euler, Integrator, VelocityVerlet};
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

/// Shifted-force pair energy with cutoff rc:
/// V(r) = U(r) - U(rc) - U'(rc) (r - rc) for r < rc, else 0.
/// Continuous at rc with V(rc) = 0.
pub fn energy_cutoff(r: f64, rc: f64) -> f64 {
    if r >= rc {
        0.0
    } else {
        energy(r) - energy(rc) + force(rc) * (r - rc)
    }
}

/// Shifted-force pair force with cutoff rc:
/// F(r) - F(rc) for r < rc, else 0. Continuous at rc.
pub fn force_cutoff(r: f64, rc: f64) -> f64 {
    if r >= rc {
        0.0
    } else {
        force(r) - force(rc)
    }
}

/// Run `steps` steps of `dt` through `integrator` from the given state and
/// return the relative total-energy error (E(t) - E0) / |E0| after each step,
/// where E0 is the energy at t = 0.
pub fn run<I: Integrator>(integrator: &I, system: &mut System, dt: f64, steps: usize) -> Vec<f64> {
    let e0 = system.total_energy();
    let mut errors = Vec::with_capacity(steps);
    for _ in 0..steps {
        integrator.step(system, dt);
        errors.push((system.total_energy() - e0) / e0.abs());
    }
    errors
}

/// The dimer experiment's initial state: two atoms at rest, separation 1.2.
pub fn dimer() -> System {
    System::new(
        vec![[0.0, 0.0], [1.2, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]],
    )
}

#[cfg(test)]
mod tests {
    use super::{dimer, energy, energy_cutoff, force, force_cutoff, greeting, run, Euler, VelocityVerlet};

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
    fn cutoff_is_continuous_and_vanishes() {
        // Exactly zero at and beyond the cutoff.
        let rc = 2.5;
        assert_eq!(energy_cutoff(rc, rc), 0.0);
        assert_eq!(force_cutoff(rc, rc), 0.0);
        assert_eq!(energy_cutoff(rc + 0.1, rc), 0.0);
        assert_eq!(force_cutoff(rc + 0.1, rc), 0.0);
        // Continuous from below (shifted potential: V(rc-) -> 0).
        assert!(energy_cutoff(rc - 1e-12, rc).abs() < 1e-9);
        assert!(force_cutoff(rc - 1e-12, rc).abs() < 1e-9);
    }

    #[test]
    fn cutoff_force_matches_energy_slope() {
        // The shifted force is still minus the slope of the shifted energy.
        let rc = 2.5;
        let h = 1e-6;
        for r in [1.0, 1.5, 2.2] {
            let df = -(energy_cutoff(r + h, rc) - energy_cutoff(r - h, rc)) / (2.0 * h);
            let tol = 1e-5 * force_cutoff(r, rc).abs().max(1.0);
            assert!((force_cutoff(r, rc) - df).abs() < tol, "r = {r}");
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
