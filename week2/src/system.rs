//! Owned particle state and pair-force physics for the md crate.

use crate::{energy, energy_cutoff, force, force_cutoff};

/// Which pair-scan strategy the force and energy loops use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ForceStrategy {
    /// All pairs i < j (open boundaries, small systems, legacy behavior).
    Naive,
    /// Neighbor search through a cell list over the periodic box.
    Cells,
}

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
    /// Pair-scan strategy; the default is the naive loop.
    force: ForceStrategy,
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
            force: ForceStrategy::Naive,
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
            force: ForceStrategy::Naive,
        };
        system.refresh_accelerations();
        system
    }

    /// Box side, if periodic.
    pub fn box_l(&self) -> Option<f64> {
        self.box_size
    }

    /// Choose the pair-scan strategy used by forces and energies.
    pub fn set_force(&mut self, strategy: ForceStrategy) {
        self.force = strategy;
    }

    /// Visit every pair (i, j, d, r) inside the cutoff once, with the
    /// minimum-image displacement d and distance r.
    fn scan_pairs(&self, visit: impl FnMut(usize, usize, [f64; 2], f64)) {
        let use_cells = matches!(self.force, ForceStrategy::Cells)
            && self.box_size.is_some()
            && self.rc.is_some()
            && self.box_size.unwrap() / self.rc.unwrap() >= 2.0;
        if use_cells {
            self.scan_cells(visit);
        } else {
            self.scan_naive(visit);
        }
    }

    fn scan_naive(&self, mut visit: impl FnMut(usize, usize, [f64; 2], f64)) {
        let n = self.n_atoms();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.displacement(i, j);
                let r2 = d[0] * d[0] + d[1] * d[1];
                if r2 == 0.0 {
                    continue;
                }
                let r = r2.sqrt();
                if let Some(rc) = self.rc {
                    if r >= rc {
                        continue;
                    }
                }
                visit(i, j, d, r);
            }
        }
    }

    /// Cell-list scan: pairs within the cutoff never straddle more than one
    /// cell boundary, so each atom needs only its own cell and its 8
    /// periodic neighbours. Every unordered pair is visited once.
    fn scan_cells(&self, mut visit: impl FnMut(usize, usize, [f64; 2], f64)) {
        let l = self.box_size.unwrap();
        let rc = self.rc.unwrap();
        let n_side = (l / rc) as usize; // cell side >= rc
        let cell_side = l / n_side as f64;
        let n_cells = n_side * n_side;
        let mut grid: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
        for (idx, p) in self.positions.iter().enumerate() {
            let cx = (p[0].div_euclid(cell_side) as usize).rem_euclid(n_side);
            let cy = (p[1].div_euclid(cell_side) as usize).rem_euclid(n_side);
            grid[cy * n_side + cx].push(idx);
        }
        let mut pair = |i: usize, j: usize| {
            let d = self.displacement(i, j);
            let r2 = d[0] * d[0] + d[1] * d[1];
            if r2 == 0.0 {
                return;
            }
            let r = r2.sqrt();
            if r < rc {
                visit(i, j, d, r);
            }
        };
        for cy in 0..n_side {
            for cx in 0..n_side {
                let c = cy * n_side + cx;
                // Same cell: each unordered pair once.
                for a in 0..grid[c].len() {
                    for b in (a + 1)..grid[c].len() {
                        pair(grid[c][a], grid[c][b]);
                    }
                }
                // Neighbour cells with a higher cell id (periodic wrap).
                for oy in -1i64..=1 {
                    for ox in -1i64..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let mcy = (cy as i64 + oy).rem_euclid(n_side as i64) as usize;
                        let mcx = (cx as i64 + ox).rem_euclid(n_side as i64) as usize;
                        let m = mcy * n_side + mcx;
                        if m <= c {
                            continue;
                        }
                        for &i in &grid[c] {
                            for &j in &grid[m] {
                                pair(i, j);
                            }
                        }
                    }
                }
            }
        }
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
        self.scan_pairs(|i, j, d, r| {
            let f = match self.rc {
                Some(rc) => force_cutoff(r, rc),
                None => force(r),
            } / r;
            for k in 0..2 {
                a[i][k] += f * d[k];
                a[j][k] -= f * d[k];
            }
        });
        self.accelerations = a;
    }

    /// Potential energy from all pairs: sum_{i<j} energy(r_ij), respecting the
    /// box (minimum image) and cutoff when set.
    pub fn pair_energy(&self) -> f64 {
        let mut u = 0.0;
        self.scan_pairs(|_i, _j, _d, r| {
            u += match self.rc {
                Some(rc) => energy_cutoff(r, rc),
                None => energy(r),
            };
        });
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
    fn cells_match_naive_on_random_configs() {
        use crate::rng::Rng;
        use crate::ForceStrategy;
        // Same random periodic configuration under both strategies.
        let (l, rc, n) = (10.0, 2.5, 37);
        let mut rng = Rng::new(2026);
        let positions: Vec<[f64; 2]> = (0..n)
            .map(|_| [rng.uniform() * l, rng.uniform() * l])
            .collect();
        let velocities = vec![[0.0, 0.0]; n];
        let mut naive = System::periodic(positions.clone(), velocities.clone(), l, rc);
        naive.set_force(ForceStrategy::Naive);
        naive.refresh_accelerations();
        let e_naive = naive.pair_energy();
        let a_naive = naive.accelerations.clone();

        let mut cells = System::periodic(positions, velocities, l, rc);
        cells.set_force(ForceStrategy::Cells);
        cells.refresh_accelerations();
        let e_cells = cells.pair_energy();
        let a_cells = cells.accelerations.clone();

        let scale_a = a_naive.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
        for k in 0..n {
            for c in 0..2 {
                assert!(
                    (a_naive[k][c] - a_cells[k][c]).abs() < 1e-12 * scale_a,
                    "accel [{k}][{c}] {} vs {}",
                    a_naive[k][c],
                    a_cells[k][c]
                );
            }
        }
        let scale_e = e_naive.abs().max(1.0);
        assert!((e_naive - e_cells).abs() < 1e-12 * scale_e, "energy {e_naive} vs {e_cells}");
    }

    #[test]
    fn cells_match_naive_across_boundary() {
        use crate::ForceStrategy;
        // A pair straddling the periodic boundary, both strategies equal.
        let (l, rc) = (10.0, 2.5);
        let positions = vec![[0.05, 5.0], [9.95, 5.0]];
        let velocities = vec![[0.0, 0.0]; 2];
        let mut naive = System::periodic(positions.clone(), velocities.clone(), l, rc);
        naive.set_force(ForceStrategy::Naive);
        let mut cells = System::periodic(positions, velocities, l, rc);
        cells.set_force(ForceStrategy::Cells);
        naive.refresh_accelerations();
        cells.refresh_accelerations();
        for k in 0..2 {
            for c in 0..2 {
                assert_eq!(naive.accelerations[k][c], cells.accelerations[k][c]);
            }
        }
        assert!(naive.accelerations[0][0] > 0.0, "repulsion through the boundary");
    }

    #[test]
    fn cells_and_naive_vanish_beyond_cutoff() {
        use crate::ForceStrategy;
        // Every minimum-image distance above rc: zero force and energy.
        let (l, rc) = (10.5, 2.5);
        let positions: Vec<[f64; 2]> = (0..9)
            .map(|i| [3.5 * (i % 3) as f64, 3.5 * (i / 3) as f64])
            .collect();
        let velocities = vec![[0.0, 0.0]; 9];
        for strategy in [ForceStrategy::Naive, ForceStrategy::Cells] {
            let mut s = System::periodic(positions.clone(), velocities.clone(), l, rc);
            s.set_force(strategy);
            s.refresh_accelerations();
            assert_eq!(s.pair_energy(), 0.0);
            assert!(s.accelerations.iter().flatten().all(|v| *v == 0.0));
        }
    }

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
