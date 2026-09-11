//! Acceptance tests for the shifted-cutoff Lennard-Jones pair interaction.
//!
//! Course Requirement: U_cut(r) = U(r) - U(rc) for r < rc, 0 for r >= rc;
//! the potential is continuous and approaches zero at rc; the force is the
//! original Part 2 LJ force below rc and zero at/above rc (it has a small
//! jump at rc).

use md::pair::{RC, shifted_energy, shifted_force};

#[test]
fn shifted_potential_is_continuous_approaching_rc_from_below() {
    // Probing just inside rc, the value approaches 0.
    for eps in [1e-1, 1e-2, 1e-3, 1e-6, 1e-9] {
        let u = shifted_energy(RC - eps);
        assert!(u.abs() < 1e-6, "U_cut({}) = {u}, expected ~0", RC - eps);
    }
    assert_eq!(shifted_energy(RC), 0.0);
    assert_eq!(shifted_energy(RC + 1.0), 0.0);
}

#[test]
fn shifted_potential_minus_one_at_minimum() {
    let r_min = 2f64.powf(1.0 / 6.0);
    assert!((shifted_energy(r_min) - (md::lj_energy(r_min) - md::lj_energy(RC))).abs() < 1e-12);
}

#[test]
fn force_is_plain_lj_below_rc_and_zero_at_or_above() {
    let r_min = 2f64.powf(1.0 / 6.0);
    for r in [0.9, 1.0, r_min, 2.0, RC - 1e-9] {
        assert!((shifted_force(r) - md::lj_force(r)).abs() < 1e-12);
    }
    assert_eq!(shifted_force(RC), 0.0);
    assert_eq!(shifted_force(RC + 0.5), 0.0);
}

#[test]
fn force_matches_finite_difference_below_rc_with_non_straddling_stencil() {
    // Course Requirement: force is -dU_cut/dr below rc. Stencils must stay
    // strictly inside r < rc because the force jumps at rc.
    let h = 1e-6;
    for r in [1.0, 1.5, 2.0, 2.4] {
        let fd = -(shifted_energy(r + h) - shifted_energy(r - h)) / (2.0 * h);
        let tol = 1e-5 * f64::max(1.0, shifted_force(r).abs());
        assert!(
            (shifted_force(r) - fd).abs() < tol,
            "force({r}) disagrees with finite difference"
        );
    }
}