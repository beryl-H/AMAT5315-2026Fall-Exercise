//! Unit tests for the Lennard-Jones pair potential in reduced units.
//!
//! Reduced units set the energy scale to epsilon = 1 and the length scale to
//! sigma = 1, so the pair energy is
//!
//!     lj_energy(r) = 4 * (r^-12 - r^-6)
//!
//! and the associated (repulsive-positive) scalar force is
//!
//!     lj_force(r) = -d lj_energy / dr
//!                 = 24 / r * (2 * r^-12 - r^-6).

const EPSILON: f64 = 1e-12;

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < EPSILON
}

const FD_H: f64 = 1e-5;

fn force_matches_finite_difference(r: f64) -> bool {
    // Independent reference: negative central finite-difference derivative.
    let f_fd = -(md::lj_energy(r + FD_H) - md::lj_energy(r - FD_H)) / (2.0 * FD_H);
    let tolerance = 1e-6 * f64::max(1.0, md::lj_force(r).abs());
    (md::lj_force(r) - f_fd).abs() < tolerance
}

#[test]
fn lj_pair_energy_is_minus_one_at_the_potential_minimum() {
    // The minimum sits at r_min = 2^(1/6), where V(r_min) = -epsilon = -1.
    let r_min = 2f64.powf(1.0 / 6.0);
    assert!(close(md::lj_energy(r_min), -1.0));
}

#[test]
fn lj_force_matches_finite_difference_of_energy_across_minimum() {
    // Check separations on both sides of r_min, covering repulsive
    // (r < r_min) and attractive (r > r_min) regions.
    let r_min = 2f64.powf(1.0 / 6.0);
    let separations = [
        0.75,
        r_min - 0.25,
        r_min,
        r_min + 0.25,
        r_min + 0.5,
    ];
    for r in separations {
        assert!(
            force_matches_finite_difference(r),
            "lj_force({r}) disagrees with the finite-difference derivative"
        );
    }
}
