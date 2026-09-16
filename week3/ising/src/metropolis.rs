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
    if delta_e <= 0.0 {
        return true;
    }
    draw < (-delta_e / t).exp()
}

/// One Metropolis sweep: exactly l*l random-site proposals.
pub fn sweep<G: Rng>(lattice: &mut Lattice, rng: &mut G, t: f64) -> SweepStats {
    let l2 = lattice.l * lattice.l;
    let mut accepted = 0;
    for _ in 0..l2 {
        let i = rng.gen_range(0..l2);
        if should_accept(lattice.delta_e(i), t, rng.gen::<f64>()) {
            lattice.flip(i);
            accepted += 1;
        }
    }
    SweepStats { accepted }
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
