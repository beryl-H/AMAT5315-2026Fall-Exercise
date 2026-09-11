//! Shifted-cutoff Lennard-Jones pair functions (rc = 2.5).
//!
//! U_cut(r) = lj_energy(r) - lj_energy(rc) for r < rc, else 0: continuous,
//! zero at rc. The force is the plain Part 2 LJ force below rc and zero
//! at/above rc — it has a small jump at rc [Course Requirement].

use crate::{lj_energy, lj_force};

/// Cutoff distance in reduced units [Course Requirement].
pub const RC: f64 = 2.5;

/// Shifted pair energy.
pub fn shifted_energy(r: f64) -> f64 {
    if r < RC {
        lj_energy(r) - lj_energy(RC)
    } else {
        0.0
    }
}

/// Cut pair force: plain LJ below rc, zero at and above rc.
pub fn shifted_force(r: f64) -> f64 {
    if r < RC {
        lj_force(r)
    } else {
        0.0
    }
}
