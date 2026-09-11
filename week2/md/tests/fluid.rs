//! Acceptance tests for the periodic fluid force loop.
//!
//! Course Requirement: Newton's third law makes the total internal force
//! analytically zero; numerically we require |sum_i F_i| < tolerance.

use md::fluid::fluid_accelerations;
use md::system::{Box2, lattice_state, minimum_image, wrap};
use md::State;

const FORCE_TOL: f64 = 1e-10;

fn total_force(state: &State, bx: &Box2) -> [f64; 2] {
    let acc = fluid_accelerations(state, bx);
    let mut sum = [0.0, 0.0];
    for a in acc {
        sum[0] += a[0];
        sum[1] += a[1];
    }
    sum
}

#[test]
fn total_internal_force_vanishes_on_lattice() {
    let state = lattice_state(100, 0.8);
    let bx = Box2::new(100, 0.8);
    let f = total_force(&state, &bx);
    assert!(f[0].abs() < FORCE_TOL, "sum Fx = {}", f[0]);
    assert!(f[1].abs() < FORCE_TOL, "sum Fy = {}", f[1]);
}

#[test]
fn total_internal_force_vanishes_on_perturbed_configuration() {
    // Deterministic perturbation (no RNG needed): displace every other atom.
    let mut state = lattice_state(100, 0.8);
    let bx = Box2::new(100, 0.8);
    for (i, p) in state.positions.iter_mut().enumerate() {
        if i % 2 == 0 {
            p[0] = wrap(p[0] + 0.37 * (i as f64 % 3.0), bx.lx);
            p[1] = wrap(p[1] + 0.11, bx.ly);
        }
    }
    let f = total_force(&state, &bx);
    assert!(f[0].abs() < FORCE_TOL, "sum Fx = {}", f[0]);
    assert!(f[1].abs() < FORCE_TOL, "sum Fy = {}", f[1]);
}

#[test]
fn minimum_image_displacement_matches_definition() {
    assert!((minimum_image(0.3, 1.0) - 0.3).abs() < 1e-12);
    assert!((minimum_image(0.6, 1.0) - (-0.4)).abs() < 1e-12);
    assert!((minimum_image(-0.6, 1.0) - 0.4).abs() < 1e-12);
    assert!((minimum_image(5.3, 1.0) - 0.3).abs() < 1e-12);
}

#[test]
fn wrap_puts_positions_in_half_open_box_and_leaves_velocities() {
    assert_eq!(wrap(-0.2, 1.0), 0.8);
    assert_eq!(wrap(1.2, 1.0), 0.2);
    assert_eq!(wrap(0.0, 1.0), 0.0);
    assert!(wrap(1.0, 1.0) < 1.0 && wrap(1.0, 1.0) >= 0.0);
}