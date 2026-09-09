//! Owned particle state and pair-force physics for the md crate.

use crate::{energy, energy_cutoff, force, force_cutoff};

/// Owns the particle arrays; dropping it releases their storage.
pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    /// Acceleration cache: integrators refresh it at the positions they need.
    pub(crate) accelerations: Vec<[f64; 2]>,
    /// Periodic box side length; None for open boundaries (the dimer).
    pub(crate) box_size: Option<f64>,
    /// Pair cutoff with shifted-force potential; None for full range.
    pub(crate) rc: Option<f64>,
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
            box_size: None,
            rc: None,
        };
        system.refresh_accelerations();
        system
    }

    /// Periodic fluid state: minimum-image boundaries and a shifted-force cutoff.
    pub fn periodic(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_l: f64,
        rc: f64,
    ) -> Self {
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have equal length"
        );
        let mut system = Self {
            positions,
            velocities,
            accelerations: Vec::new(),
            box_size: Some(box_l),
            rc: Some(rc),
        };
        system.refresh_accelerations();
        system
    }

    /// Box side, if periodic.
    pub fn box_l(&self) -> Option<f64> {
        self.box_size
    }

    /// Minimum-image displacement d = x_i - x_j.
    fn displacement(&self, i: usize, j: usize) -> [f64; 2] {
        let mut d = [
            self.positions[i][0] - self.positions[j][0],
            self.positions[i][1] - self.positions[j][1],
        ];
        if let Some(l) = self.box_size {
            for k in 0..2 {
                if d[k] > 0.5 * l {
                    d[k] -= l;
                } else if d[k] < -0.5 * l {
                    d[k] += l;
                }
            }
        }
        d
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
                let d = self.displacement(i, j);
                let r2 = d[0] * d[0] + d[1] * d[1];
                if r2 == 0.0 {
                    continue;
                }
                let r = r2.sqrt();
                let f = match self.rc {
                    Some(rc) => force_cutoff(r, rc),
                    None => force(r),
                } / r;
                for k in 0..2 {
                    a[i][k] += f * d[k];
                    a[j][k] -= f * d[k];
                }
            }
        }
        self.accelerations = a;
    }

    /// Potential energy from all pairs: sum_{i<j} energy(r_ij), respecting the
    /// box (minimum image) and cutoff when set.
    pub fn pair_energy(&self) -> f64 {
        let mut u = 0.0;
        for i in 0..self.n_atoms() {
            for j in (i + 1)..self.n_atoms() {
                let d = self.displacement(i, j);
                let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
                u += match self.rc {
                    Some(rc) => energy_cutoff(r, rc),
                    None => energy(r),
                };
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

#[cfg(test)]
mod tests {
    use super::System;

    #[test]
    fn minimum_image_force_across_boundary() {
        // Two atoms a hair inside opposite walls: the minimum image distance
        // is small, so they repel strongly through the boundary.
        let l = 10.0;
        let mut s = System::periodic(
            vec![[0.1, 5.0], [9.9, 5.0]],
            vec![[0.0, 0.0]; 2],
            l,
            2.5,
        );
        s.refresh_accelerations();
        // Minimum image distance is 0.2: atom 0 sees atom 1's image at
        // x = -0.1 (left), atom 1 sees atom 0's image at x = 10.1 (right).
        // Repulsion pushes each back toward the box interior: atom 0 +x,
        // atom 1 -x.
        assert!(s.accelerations[0][0] > 0.0);
        assert!(s.accelerations[1][0] < 0.0);
    }

    #[test]
    fn periodic_energy_uses_shifted_cutoff() {
        // A pair beyond rc contributes exactly zero energy.
        let l = 10.0;
        let s = System::periodic(
            vec![[0.0, 0.0], [3.0, 0.0]], // 3.0 > rc = 2.5
            vec![[0.0, 0.0]; 2],
            l,
            2.5,
        );
        assert_eq!(s.pair_energy(), 0.0);
    }

    #[test]
    fn open_boundaries_unchanged() {
        // System::new keeps full-range potential and no wrapping.
        let s = System::new(vec![[0.0, 0.0], [3.0, 0.0]], vec![[0.0, 0.0]; 2]);
        assert_eq!(s.box_l(), None);
        assert!(s.pair_energy() != 0.0); // full-range U(3) < 0
    }
}
