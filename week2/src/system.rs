//! Owned particle state and pair-force physics for the md crate.

use crate::{energy, force};

/// Owns the particle arrays; dropping it releases their storage.
pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    /// Acceleration cache: integrators refresh it at the positions they need.
    pub(crate) accelerations: Vec<[f64; 2]>,
}

impl System {
    /// New state from positions and velocities; computes the initial
    /// accelerations from the pair forces.
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have equal length"
        );
        let mut system = Self {
            positions,
            velocities,
            accelerations: Vec::new(),
        };
        system.refresh_accelerations();
        system
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    /// Refresh the acceleration cache from the current positions.
    /// With d = x_i - x_j and r = |d|, the force on atom i is force(r) * d / r
    /// (m = 1, so acceleration = force); atom j feels exactly the opposite.
    pub(crate) fn refresh_accelerations(&mut self) {
        let n = self.n_atoms();
        let mut a = vec![[0.0, 0.0]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    self.positions[i][0] - self.positions[j][0],
                    self.positions[i][1] - self.positions[j][1],
                ];
                let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
                let f = force(r) / r;
                for k in 0..2 {
                    a[i][k] += f * d[k];
                    a[j][k] -= f * d[k];
                }
            }
        }
        self.accelerations = a;
    }

    /// Potential energy from all pairs: sum_{i<j} energy(r_ij).
    pub fn pair_energy(&self) -> f64 {
        let mut u = 0.0;
        for i in 0..self.n_atoms() {
            for j in (i + 1)..self.n_atoms() {
                let d = [
                    self.positions[i][0] - self.positions[j][0],
                    self.positions[i][1] - self.positions[j][1],
                ];
                let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
                u += energy(r);
            }
        }
        u
    }

    /// Kinetic energy: 0.5 * sum_i |v_i|^2.
    pub fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
            .sum()
    }

    /// Total energy: kinetic plus potential.
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.pair_energy()
    }
}
