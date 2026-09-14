//! Independent physics metrics recomputed from raw trajectory data.

use crate::fluid::fluid_potential_energy;
use crate::simulate::Frame;
use crate::state::State;
use crate::system::{Box2, minimum_image};
use crate::Vec2;

/// Recomputed total energies per frame: shifted E_pot from raw wrapped
/// positions (minimum image) + E_kin from raw velocities. Independent of the
/// stored E_pot/E_kin in each frame.
pub fn frame_total_energies(frames: &[Frame], bx: &Box2) -> Vec<f64> {
    frames
        .iter()
        .map(|f| {
            let state = State {
                positions: f.pos.clone(),
                velocities: f.vel.clone(),
            };
            fluid_potential_energy(&state, bx)
                + f.vel
                    .iter()
                    .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
                    .sum::<f64>()
        })
        .collect()
}

/// |mean(E_last_k) - mean(E_first_k)| / |E0| with k = max(1, floor(F/10))
/// and E0 = totals[0]. [Suggestion] empty input or E0 = 0 is not a Course
/// Requirement: the plan uses totals[0] as E0 and would panic on an empty
/// slice (the caller always supplies F >= 2 real frames).
pub fn secular_drift(totals: &[f64]) -> f64 {
    let f = totals.len();
    let k = std::cmp::max(1, f / 10);
    let first: f64 = totals[..k].iter().sum::<f64>() / k as f64;
    let last: f64 = totals[f - k..].iter().sum::<f64>() / k as f64;
    (last - first).abs() / totals[0].abs()
}

/// All speeds v = |velocity| pooled over frames and atoms.
pub fn pooled_speeds(frames: &[Frame]) -> Vec<f64> {
    frames
        .iter()
        .flat_map(|f| f.vel.iter())
        .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
        .collect()
}

/// T_speed = mean(v^2)/2 [Course Requirement].
pub fn t_speed(speeds: &[f64]) -> f64 {
    let n = speeds.len() as f64;
    speeds.iter().map(|s| s * s).sum::<f64>() / n / 2.0
}

/// 24 equal-probability Rayleigh bin edges at temperature t; b_24 = inf.
/// b_j = sqrt(-2*t*ln(1 - j/24)) for j = 1..23, b_0 = 0.
pub fn bin_edges(t: f64) -> [f64; 25] {
    let mut edges = [0.0; 25];
    for k in 1..24 {
        edges[k] = (-2.0 * t * (1.0 - k as f64 / 24.0).ln()).sqrt();
    }
    edges[24] = f64::INFINITY;
    edges
}

/// Reduced chi-square: (1/22) * sum_b (O_b - E_b)^2 / E_b with E_b = S/24.
/// Bin edges use the passed temperature (callers pass T_speed) [Course Req].
pub fn chi2_22(speeds: &[f64], t: f64) -> f64 {
    let edges = bin_edges(t);
    let s = speeds.len() as f64;
    let expected = s / 24.0;
    let mut stat = 0.0;
    for b in 0..24 {
        let observed = speeds
            .iter()
            .filter(|&&v| v >= edges[b] && v < edges[b + 1])
            .count() as f64;
        stat += (observed - expected).powi(2) / expected;
    }
    stat / 22.0
}

/// g(r) averaged over atoms and frames: unordered minimum-image pairs,
/// g_k = 2*H_k / (N*F*rho*pi*(r_out^2 - r_in^2)) [Course Requirement].
/// Returns (r_center, g) per bin; 50 bins is [Suggestion].
pub fn radial_distribution(pos_frames: &[&[Vec2]], bx: &Box2, bins: usize) -> Vec<(f64, f64)> {
    let r_max = 0.5 * bx.lx.min(bx.ly);
    let dr = r_max / bins as f64;
    let n = pos_frames.first().map(|f| f.len()).unwrap_or(0) as f64;
    let f_count = pos_frames.len() as f64;
    let rho = n / (bx.lx * bx.ly); // N / area
    let mut hist = vec![0u64; bins];
    for frame in pos_frames {
        for i in 0..frame.len() {
            for j in (i + 1)..frame.len() {
                let dx = minimum_image(frame[i][0] - frame[j][0], bx.lx);
                let dy = minimum_image(frame[i][1] - frame[j][1], bx.ly);
                let r = (dx * dx + dy * dy).sqrt();
                if r < r_max {
                    hist[(r / dr) as usize] += 1;
                }
            }
        }
    }
    (0..bins)
        .map(|b| {
            let r_in = b as f64 * dr;
            let r_out = r_in + dr;
            let area = std::f64::consts::PI * (r_out * r_out - r_in * r_in);
            let g = 2.0 * hist[b] as f64 / (n * f_count * rho * area);
            (r_in + 0.5 * dr, g)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec2;

    #[test]
    fn secular_drift_uses_k_is_floor_of_tenth() {
        // F = 10 frames -> k = 1: drift = |E_last - E_first| / |E_first|.
        let totals: Vec<f64> = (0..10).map(|i| 10.0 + 0.001 * i as f64).collect();
        let expected = (10.009 - 10.0) / 10.0;
        assert!((secular_drift(&totals) - expected).abs() < 1e-15);
        // F = 200 -> k = 20: means over first and last 20; constant energy
        // sequence gives zero drift.
        let t = vec![5.0; 200];
        assert!(secular_drift(&t).abs() < 1e-15);
    }

    #[test]
    fn t_speed_is_half_the_mean_squared_speed() {
        let speeds = vec![1.0, 2.0, 3.0]; // mean(v^2) = 14/3
        assert!((t_speed(&speeds) - 14.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn bin_edges_are_equal_probability_rayleigh() {
        let t = 0.5;
        let edges = bin_edges(t);
        assert_eq!(edges[0], 0.0);
        assert_eq!(edges[24], f64::INFINITY);
        // b_k = sqrt(-2 t ln(1 - k/24)).
        let expected = (-2.0 * t * (1.0f64 - 12.0 / 24.0).ln()).sqrt();
        assert!((edges[12] - expected).abs() < 1e-15);
    }

    #[test]
    fn chi2_22_is_small_for_rayleigh_samples_and_reduced() {
        // Synthetic speeds drawn from Rayleigh(T): chi2_22 should be ~1.
        // Deterministic construction via inverse CDF on a regular grid. n is
        // deliberately NOT a multiple of 24: at n = 24000 the grid lands
        // exactly on the equal-probability bin boundaries, giving chi2 == 0
        // and tripping the c > 1e-6 guard below. n = 24001 keeps a small
        // nonzero statistic (~4e-5) while staying well under the course bound.
        let t = 0.5;
        let n = 24001;
        let speeds: Vec<f64> = (0..n)
            .map(|i| (-2.0 * t * (1.0 - (i as f64 + 0.5) / n as f64).ln()).sqrt())
            .collect();
        let ts = t_speed(&speeds);
        let c = chi2_22(&speeds, ts);
        assert!(c < 2.0, "chi2_22 = {c}");
        assert!(c > 1e-6, "chi2_22 suspiciously zero: {c}");
    }

    #[test]
    fn chi2_22_is_large_for_non_maxwell_speeds() {
        // All speeds in one place: the statistic blows up.
        let speeds = vec![0.5; 2400];
        let ts = t_speed(&speeds);
        assert!(chi2_22(&speeds, ts) > 2.0);
    }

    #[test]
    fn radial_distribution_of_a_uniform_box_is_about_one() {
        // Hand-built near-uniform positions on a coarse grid; middle bins of
        // a uniform box average ~1.
        let bx = Box2 { lx: 10.0, ly: 10.0 };
        let mut pos = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                pos.push([(i as f64 + 0.5), (j as f64 + 0.5)]);
            }
        }
        let frames: Vec<&[Vec2]> = vec![&pos];
        let g = radial_distribution(&frames, &bx, 50);
        assert_eq!(g.len(), 50);
        // Mean over bins between r = 1 and r = 4 should be near 1.
        let mid: Vec<f64> = g
            .iter()
            .filter(|(r, _)| *r > 1.5 && *r < 4.0)
            .map(|(_, v)| *v)
            .collect();
        let mean: f64 = mid.iter().sum::<f64>() / mid.len() as f64;
        assert!((mean - 1.0).abs() < 0.2, "mean g = {mean}");
    }
}
