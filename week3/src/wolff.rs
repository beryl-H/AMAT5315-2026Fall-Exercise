//! Single-cluster (Wolff) updates on the Ising lattice.
//!
//! One step is one cluster flip: pick a uniformly random seed site, grow
//! the cluster of same-spin sites by adding each boundary neighbour with
//! probability p_add = 1 - exp(-2/T), then flip the whole cluster. Every
//! proposed addition is accepted or rejected exactly once, so there is no
//! rejection rate: the reported statistic is the cluster size.

use rand::Rng;

use crate::lattice::Lattice;

/// Outcome of one Wolff step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WolffStats {
    /// Number of spins flipped in this move (the cluster size, >= 1).
    pub cluster_size: usize,
}

/// Wolff bond addition probability for J = 1: p_add = 1 - exp(-2/T).
pub fn add_probability(t: f64) -> f64 {
    1.0 - (-2.0 / t).exp()
}

/// One Wolff step: grow a single cluster from a random seed site and flip
/// it. Returns the cluster size.
pub fn cluster_flip<G: Rng>(lattice: &mut Lattice, rng: &mut G, t: f64) -> WolffStats {
    let l2 = lattice.l * lattice.l;
    let p_add = add_probability(t);
    let seed = rng.gen_range(0..l2);
    let spin = lattice.get(seed);

    let mut in_cluster = vec![false; l2];
    in_cluster[seed] = true;
    let mut stack = vec![seed];
    let mut size = 0usize;
    while let Some(s) = stack.pop() {
        size += 1;
        for nb in lattice.neighbours(s) {
            if lattice.get(nb) == spin && !in_cluster[nb] && rng.gen::<f64>() < p_add {
                in_cluster[nb] = true;
                stack.push(nb);
            }
        }
    }
    for i in 0..l2 {
        if in_cluster[i] {
            lattice.flip(i);
        }
    }
    WolffStats { cluster_size: size }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn add_probability_extremes_and_monotonicity() {
        // T -> 0: p_add -> 1 (huge ordered clusters).
        assert_eq!(add_probability(1e-9), 1.0);
        // T -> infinity: p_add -> 0 (single-spin clusters).
        assert!(add_probability(1e12) < 1e-11);
        // T = 1: 1 - e^-2.
        assert!((add_probability(1.0) - (1.0 - (-2.0f64).exp())).abs() < 1e-15);
        // Monotonically decreasing in T.
        let mut prev = 1.0;
        for t in [0.1, 0.5, 1.0, 2.269, 5.0, 100.0] {
            let p = add_probability(t);
            assert!(p <= prev, "p_add must not increase with T");
            prev = p;
        }
    }

    #[test]
    fn low_temperature_cluster_is_whole_lattice_from_all_up() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let mut lat = Lattice::all_up(4);
        let stats = cluster_flip(&mut lat, &mut rng, 1e-9);
        assert_eq!(stats.cluster_size, 16);
        assert_eq!(lat.magnetization(), -1.0);
        assert_eq!(lat.energy_per_site(), -2.0); // energy unchanged
    }

    #[test]
    fn high_temperature_cluster_is_seed_site_alone() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(19);
        let mut lat = Lattice::all_up(4);
        let stats = cluster_flip(&mut lat, &mut rng, 1e12);
        assert_eq!(stats.cluster_size, 1);
        assert!((lat.magnetization() - 0.875).abs() < 1e-12);
    }

    #[test]
    fn flipped_sites_match_cluster_size_and_are_connected_in_practice() {
        // At T = 2.269 on a 4x4 lattice the cluster is larger than 1 and
        // its flip leaves exactly cluster_size sites opposite to before.
        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        let mut lat = Lattice::all_up(4);
        let before = lat.spins.clone();
        let stats = cluster_flip(&mut lat, &mut rng, 2.269);
        let flips: usize = before
            .iter()
            .zip(lat.spins.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(flips, stats.cluster_size);
        assert!(stats.cluster_size > 1);
    }

    #[test]
    fn two_temperature_grid_runs_and_stays_bounded() {
        // Statistical smoke test: wolff ramp on 8x8 over a wide T range.
        let mut rng = rand::rngs::StdRng::seed_from_u64(2026);
        let results = crate::ramp::run_ramp(
            8,
            crate::ramp::Update::Wolff,
            &[1.5, 3.5],
            100,
            200,
            0,
            &mut rng,
        );
        let m = results[0].mean_abs_m;
        assert!(m > 0.8, "T=1.5 wolff |M| should be large: {m}");
        let m = results[1].mean_abs_m;
        assert!(m < 0.6, "T=3.5 wolff |M| should be small: {m}");
        // The reported statistic at wolff temperatures is the mean cluster
        // size, bounded between 1 and the lattice size.
        for r in &results {
            assert!((1.0..=(64.0)).contains(&r.stat), "mean cluster size out of range: {}", r.stat);
        }
    }
}
