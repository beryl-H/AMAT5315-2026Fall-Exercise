//! Seeded Gaussian initialization, COM removal, and the simple thermostat.

use crate::Vec2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Independent Gaussian velocity components, mean 0, variance `temperature`,
/// drawn atom-major (vx, vy per atom), seeded by `seed`.
pub fn gaussian_velocities(n: usize, temperature: f64, seed: u64) -> Vec<Vec2> {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigma = temperature.sqrt();
    (0..n)
        .map(|_| {
            let vx: f64 = rng.sample(rand_distr::Normal::new(0.0, sigma).unwrap());
            let vy: f64 = rng.sample(rand_distr::Normal::new(0.0, sigma).unwrap());
            [vx, vy]
        })
        .collect()
}

/// Subtract the component-wise mean velocity (once, per the schedule).
pub fn remove_com_velocity(velocities: &mut [Vec2]) {
    let n = velocities.len() as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for v in velocities.iter() {
        mx += v[0];
        my += v[1];
    }
    mx /= n;
    my /= n;
    for v in velocities.iter_mut() {
        v[0] -= mx;
        v[1] -= my;
    }
}

/// T_thermo = 2*E_kin/(2N - 2) [Course Requirement].
pub fn thermodynamic_temperature(velocities: &[Vec2]) -> f64 {
    let e_kin: f64 = velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum();
    let n = velocities.len() as f64;
    2.0 * e_kin / (2.0 * n - 2.0)
}

/// Rescale every component by sqrt(target / T_thermo) [Course Requirement].
pub fn rescale_to(velocities: &mut [Vec2], target: f64) {
    let scale = (target / thermodynamic_temperature(velocities)).sqrt();
    for v in velocities.iter_mut() {
        v[0] *= scale;
        v[1] *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn gaussian_velocities_are_seeded_deterministic_with_variance_t() {
        // Course Requirement: seeded, independent, mean 0, variance T.
        let v1 = gaussian_velocities(100, 0.5, 2026);
        let v2 = gaussian_velocities(100, 0.5, 2026);
        assert_eq!(v1, v2, "same seed must reproduce identical velocities");
        let v_other = gaussian_velocities(100, 0.5, 7);
        assert_ne!(v1, v_other, "different seed must differ");
        // Empirical variance over 200 components ~ T (loose bound).
        let mean: f64 = v1.iter().flat_map(|v| v.iter()).sum::<f64>() / 200.0;
        assert!(mean.abs() < 0.2, "mean too far from 0: {mean}");
        let var: f64 = v1
            .iter()
            .flat_map(|v| v.iter())
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / 200.0;
        assert!((var - 0.5).abs() < 0.2, "variance {var} far from 0.5");
    }

    #[test]
    fn com_removal_zeroes_the_mean() {
        let mut v = vec![[1.0, 2.0], [3.0, -1.0], [-2.0, 5.0]];
        remove_com_velocity(&mut v);
        for axis in 0..2 {
            let m = v.iter().map(|p| p[axis]).sum::<f64>() / v.len() as f64;
            assert!(m.abs() < EPS, "axis {axis} mean {m}");
        }
    }

    #[test]
    fn thermodynamic_temperature_uses_two_n_minus_two() {
        // 4 atoms at rest except one: E_kin = 0.5*2 = 1, T = 2*1/(8-2).
        let v = vec![[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        assert!((thermodynamic_temperature(&v) - 2.0 / 6.0).abs() < EPS);
    }

    #[test]
    fn rescale_sets_the_thermodynamic_temperature() {
        // Zero-COM input (alternating +vx/-vx) so that "uniform rescale
        // preserves zero COM" is actually what is asserted.
        let mut v: Vec<Vec2> = (0..10)
            .map(|i| if i % 2 == 0 { [1.0, 0.0] } else { [-1.0, 0.0] })
            .collect();
        rescale_to(&mut v, 0.5);
        assert!((thermodynamic_temperature(&v) - 0.5).abs() < 1e-12);
        // Uniform rescale preserves zero COM.
        let mx = v.iter().map(|p| p[0]).sum::<f64>() / 10.0;
        assert!(mx.abs() < EPS);
    }
}
