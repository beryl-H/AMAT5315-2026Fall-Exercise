//! Temperature-ramp driver for the Ising model.
//!
//! One run uses a single random stream for the entire ramp. The first
//! temperature starts from an all-up lattice; every later temperature
//! starts from the final lattice of the preceding temperature. At each
//! temperature `discard` sweeps are discarded, then `measure` sweeps are
//! measured.

use rand::Rng;

use crate::lattice::Lattice;
use crate::metropolis;

/// One measured step at a temperature (one series.jsonl row).
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesRow {
    pub t: f64,
    pub sweep: usize,
    pub m: f64,
    pub e: f64,
}

/// One recorded spin frame (one spins.jsonl row).
#[derive(Clone, Debug, PartialEq)]
pub struct SpinFrame {
    pub t: f64,
    /// Cumulative measured-sweep index across the whole ramp: k*measure + j
    /// for temperature index k (0-based) and measured step j (1-based).
    pub global_sweep: usize,
    pub m: f64,
    pub spins: Vec<i8>,
}

/// Everything measured at one temperature.
#[derive(Clone, Debug, PartialEq)]
pub struct TemperatureResult {
    pub t: f64,
    pub mean_abs_m: f64,
    pub acceptance: f64,
    pub series: Vec<SeriesRow>,
    pub frames: Vec<SpinFrame>,
}

/// Run the whole temperature ramp on one RNG stream.
///
/// The first temperature starts from an all-up lattice; each later
/// temperature starts from the final lattice of the preceding one. At each
/// temperature `discard` sweeps are run and discarded, then `measure` sweeps
/// are measured. Frames are recorded when `every > 0` and the measured-step
/// index is divisible by `every`.
pub fn run_ramp<G: Rng>(
    l: usize,
    grid: &[f64],
    discard: usize,
    measure: usize,
    every: usize,
    rng: &mut G,
) -> Vec<TemperatureResult> {
    let mut lattice = Lattice::all_up(l);
    let mut results = Vec::with_capacity(grid.len());
    for (k, &t) in grid.iter().enumerate() {
        for _ in 0..discard {
            metropolis::sweep(&mut lattice, rng, t);
        }
        let mut total_abs_m = 0.0;
        let mut accepted = 0usize;
        let mut series = Vec::with_capacity(measure);
        let mut frames = Vec::new();
        for j in 1..=measure {
            let stats = metropolis::sweep(&mut lattice, rng, t);
            accepted += stats.accepted;
            let m = lattice.magnetization();
            let e = lattice.energy_per_site();
            total_abs_m += m.abs();
            series.push(SeriesRow { t, sweep: j, m, e });
            if every > 0 && j % every == 0 {
                frames.push(SpinFrame {
                    t,
                    global_sweep: k * measure + j,
                    m,
                    spins: lattice.spins.clone(),
                });
            }
        }
        results.push(TemperatureResult {
            t,
            mean_abs_m: total_abs_m / measure as f64,
            acceptance: accepted as f64 / (measure * l * l) as f64,
            series,
            frames,
        });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn two_temperature_grid() -> Vec<f64> {
        vec![1.0, 2.0]
    }

    #[test]
    fn series_sweep_resets_each_temperature() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        let results = run_ramp(2, &two_temperature_grid(), 0, 3, 0, &mut rng);
        let sweeps: Vec<usize> = results
            .iter()
            .flat_map(|r| r.series.iter().map(|s| s.sweep))
            .collect();
        assert_eq!(sweeps, vec![1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn every_zero_records_no_frames() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(29);
        let results = run_ramp(2, &two_temperature_grid(), 0, 3, 0, &mut rng);
        assert!(results.iter().all(|r| r.frames.is_empty()));
    }

    #[test]
    fn frames_at_multiples_of_every_with_cumulative_global_sweep() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(31);
        let results = run_ramp(2, &two_temperature_grid(), 0, 6, 2, &mut rng);
        let frames: Vec<usize> = results
            .iter()
            .flat_map(|r| r.frames.iter().map(|f| f.global_sweep))
            .collect();
        assert_eq!(frames, vec![2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn same_seed_reproduces_identical_results() {
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(2026);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(2026);
        let r1 = run_ramp(4, &two_temperature_grid(), 1, 5, 2, &mut rng1);
        let r2 = run_ramp(4, &two_temperature_grid(), 1, 5, 2, &mut rng2);
        assert_eq!(r1, r2);
    }
}
