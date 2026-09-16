//! Two-dimensional Ising lattice: L x L spins of +1/-1 on a periodic
//! square lattice, stored row-major (index = y * L + x, row 0 at the top).

/// An L x L Ising lattice. `spins` has length l*l and holds +1 or -1 at
/// index y * l + x.
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    pub l: usize,
    pub spins: Vec<i8>,
}

impl Lattice {
    /// An all-up (+1) lattice of side `l`.
    pub fn all_up(l: usize) -> Self {
        Lattice { l, spins: vec![1; l * l] }
    }

    /// Spin at site index `i`.
    pub fn get(&self, i: usize) -> i8 {
        self.spins[i]
    }

    /// Flip the spin at site index `i`.
    pub fn flip(&mut self, i: usize) {
        self.spins[i] = -self.spins[i];
    }

    /// Sum of the four periodic neighbours of site `i`: (x+1,y), (x-1,y),
    /// (x,y+1), (x,y-1), wrapping at both boundaries.
    pub fn neighbour_sum(&self, i: usize) -> i8 {
        let l = self.l as isize;
        let x = (i % self.l) as isize;
        let y = (i / self.l) as isize;
        let at = |dx: isize, dy: isize| {
            self.spins[((y + dy).rem_euclid(l)) as usize * self.l + ((x + dx).rem_euclid(l)) as usize]
        };
        at(1, 0) + at(-1, 0) + at(0, 1) + at(0, -1)
    }

    /// The four periodic neighbour indices of site `i`:
    /// (x+1,y), (x-1,y), (x,y+1), (x,y-1), wrapping at both boundaries.
    pub fn neighbours(&self, i: usize) -> [usize; 4] {
        let l = self.l;
        let x = i % l;
        let y = i / l;
        let xm = (x + 1) % l;
        let xp = (x + l - 1) % l;
        let ym = ((y + 1) % l) * l;
        let yp = ((y + l - 1) % l) * l;
        [ym + xm, ym + xp, yp + x, y * l + x]
    }

    /// Total energy E = -sum over nearest-neighbour pairs s_i s_j
    /// (each unordered pair counted once).
    pub fn energy(&self) -> f64 {
        let l = self.l;
        let mut ordered_pairs = 0.0;
        for i in 0..l * l {
            ordered_pairs += (self.spins[i] as f64) * (self.neighbour_sum(i) as f64);
        }
        -0.5 * ordered_pairs
    }

    /// Energy per site.
    pub fn energy_per_site(&self) -> f64 {
        self.energy() / (self.l * self.l) as f64
    }

    /// Magnetisation per spin M = sum(s_i) / L^2 in [-1, 1].
    pub fn magnetization(&self) -> f64 {
        self.spins.iter().map(|&s| s as f64).sum::<f64>() / (self.l * self.l) as f64
    }

    /// Energy change of flipping spin i: 2 * s_i * (s1+s2+s3+s4).
    pub fn delta_e(&self, i: usize) -> f64 {
        2.0 * (self.spins[i] as f64) * (self.neighbour_sum(i) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    /// L x L checkerboard built from an all-up lattice.
    fn checkerboard(l: usize) -> Lattice {
        let mut lat = Lattice::all_up(l);
        for y in 0..l {
            for x in 0..l {
                if (x + y) % 2 == 1 {
                    lat.flip(y * l + x);
                }
            }
        }
        lat
    }

    #[test]
    fn all_up_energy_is_minus_two_per_site() {
        let lat = Lattice::all_up(4);
        assert_eq!(lat.energy(), -32.0);
        assert_eq!(lat.energy_per_site(), -2.0);
    }

    #[test]
    fn checkerboard_energy_is_plus_two_per_site() {
        let lat = checkerboard(4);
        assert_eq!(lat.energy(), 32.0);
    }

    #[test]
    fn magnetization_is_per_spin_signed() {
        let lat = Lattice::all_up(4);
        assert_eq!(lat.magnetization(), 1.0);
        let mut mixed = Lattice::all_up(4);
        mixed.flip(5); // one down spin: (16 - 2) / 16
        assert_eq!(mixed.magnetization(), 0.875);
    }

    #[test]
    fn delta_e_matches_direct_energy_change() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut lat = checkerboard(4);
        for _ in 0..20 {
            let i = rng.gen_range(0..16);
            let de = lat.delta_e(i);
            let e_before = lat.energy();
            lat.flip(i);
            let e_after = lat.energy();
            assert!(
                (e_after - e_before - de).abs() < 1e-12,
                "site {i}: de={de}, e_before={e_before}, e_after={e_after}"
            );
            lat.flip(i); // restore
        }
    }

    #[test]
    fn delta_e_all_up_flip_costs_plus_8() {
        let lat = Lattice::all_up(4);
        assert_eq!(lat.delta_e(0), 8.0);
    }

    #[test]
    fn periodic_boundary_neighbours_wrap() {
        let mut lat = Lattice::all_up(3);
        lat.flip(6); // (x=0, y=2): the wrapped "up" neighbour of corner (0,0)
        assert_eq!(lat.neighbour_sum(0), 2);
        assert_eq!(lat.delta_e(0), 4.0); // 2 * s_i * (1+1+1-1)
    }
}
