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

#[test]
fn lj_pair_energy_is_minus_one_at_the_potential_minimum() {
    // The minimum sits at r_min = 2^(1/6), where V(r_min) = -epsilon = -1.
    let r_min = 2f64.powf(1.0 / 6.0);
    assert!(close(md::lj_energy(r_min), -1.0));
}

#[test]
fn lj_force_vanishes_at_the_potential_minimum() {
    // dV/dr = 0 at r_min, so the force is zero there.
    let r_min = 2f64.powf(1.0 / 6.0);
    assert!(close(md::lj_force(r_min), 0.0));
}
