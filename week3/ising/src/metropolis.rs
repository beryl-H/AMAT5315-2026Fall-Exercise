//! Random-site Metropolis updates on the Ising lattice.
//!
//! One sweep is exactly l*l attempted flips of uniformly random sites
//! (chosen with replacement), counting rejected proposals.

use rand::Rng;

use crate::lattice::Lattice;

/// Outcome of one Metropolis sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepStats {
    /// Number of accepted flips out of the l*l proposals in this sweep.
    pub accepted: usize,
}

/// Accept a proposed flip of energy change `delta_e` at temperature `t`,
/// given a uniform random draw in [0, 1): accept with probability
/// min(1, exp(-delta_e / t)).
pub fn should_accept(delta_e: f64, t: f64, draw: f64) -> bool {
    let _ = (delta_e, t, draw);
    false // RED stub
}

/// One Metropolis sweep: exactly l*l random-site proposals.
pub fn sweep<G: Rng>(lattice: &mut Lattice, rng: &mut G, t: f64) -> SweepStats {
    let _ = (rng, t);
    let l2 = lattice.l * lattice.l;
    // RED stub: flips every site and counts every proposal as accepted.
    for i in 0..l2 {
        lattice.flip(i);
    }
    SweepStats { accepted: l2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn non_positive_delta_e_always_accepted() {
        assert!(should_accept(0.0, 1.0, 0.999));
        assert!(should_accept(-4.0, 1.0, 0.999));
    }

    #[test]
    fn positive_delta_e_accepted_with_boltzmann_probability() {
        // exp(-8/1) ~= 3.35e-4
        assert!(!should_accept(8.0, 1.0, 0.99));
        assert!(should_accept(8.0, 1.0, 1e-6));
    }

    #[test]
    fn low_temperature_sweep_from_all_up_rejects_everything() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut lat = Lattice::all_up(4);
        let stats = sweep(&mut lat, &mut rng, 1e-12);
        assert_eq!(stats.accepted, 0);
        assert_eq!(lat, Lattice::all_up(4)); // no flip can lower the energy from all-up
    }

    #[test]
    fn high_temperature_sweep_accepts_exactly_l2_proposals() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        let mut lat = Lattice::all_up(4);
        let stats = sweep(&mut lat, &mut rng, 1e12);
        assert_eq!(stats.accepted, 16); // one sweep is exactly l*l proposals
    }
}
