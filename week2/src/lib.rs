//! Library for the `md` molecular-dynamics crate.

/// The greeting returned by this crate.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// The Lennard-Jones pair energy in reduced units (sigma = 1, epsilon = 1):
/// U(r) = 4 (r^-12 - r^-6), zero at r = 1, minimum -1 at r0 = 2^(1/6).
pub fn energy(r: f64) -> f64 {
    unimplemented!("pair energy U(r) = 4 (r^-12 - r^-6)")
}

/// The pair force magnitude F(r) = -dU/dr = 24/r (2 r^-12 - r^-6):
/// positive repels (r < r0), zero at r0 = 2^(1/6), negative attracts (r > r0).
pub fn force(r: f64) -> f64 {
    unimplemented!("pair force F(r) = 24/r (2 r^-12 - r^-6)")
}

#[cfg(test)]
mod tests {
    use super::{energy, force, greeting};

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
}
