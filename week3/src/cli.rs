//! Command-line interface: `ising --update ... --out ...` (no subcommands).

use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

use crate::io::{RunConfig, write_artifacts};
use crate::ramp::{Update, run_ramp};

/// Sample the Ising model along a temperature ramp (metropolis or wolff).
#[derive(Debug, Parser)]
#[command(name = "ising", about = "Week 3 Ising model sampler (metropolis | wolff)")]
pub struct Args {
    /// Update scheme: "metropolis" (l*l-proposal sweeps) or "wolff" (cluster flips).
    #[arg(long)]
    pub update: String,
    /// Lattice side length L (l*l spins).
    #[arg(long)]
    pub l: usize,
    /// Lowest temperature; the ramp ascends from here, from an all-up lattice.
    #[arg(long, allow_negative_numbers = true)]
    pub t_from: f64,
    /// Highest temperature.
    #[arg(long, allow_negative_numbers = true)]
    pub t_to: f64,
    /// Temperature step; each temperature starts from the previous lattice.
    #[arg(long, allow_negative_numbers = true)]
    pub t_step: f64,
    /// Equilibration sweeps discarded at each temperature.
    #[arg(long)]
    pub discard: usize,
    /// Measured sweeps at each temperature.
    #[arg(long)]
    pub measure: usize,
    /// Random seed; one random stream per run, carried through the ramp.
    #[arg(long)]
    pub seed: u64,
    /// Record a spins frame every this many measured sweeps; 0 records none.
    #[arg(long, default_value_t = 0)]
    pub every: usize,
    /// Output folder for run.json / series.jsonl / spins.jsonl.
    #[arg(long)]
    pub out: PathBuf,
}

/// Ascending temperature grid t_from ..= t_to at step t_step. Errors when
/// the span is not an integer number of steps (within a numerical
/// tolerance) instead of silently truncating.
pub fn temperature_grid(t_from: f64, t_to: f64, t_step: f64) -> Result<Vec<f64>, String> {
    if !(t_from.is_finite() && t_to.is_finite() && t_step.is_finite()) {
        return Err("--t-from, --t-to and --t-step must be finite numbers".to_string());
    }
    if t_from <= 0.0 {
        return Err(format!("--t-from must be positive, got {t_from}"));
    }
    if t_to < t_from {
        return Err(format!("--t-to ({t_to}) must be >= --t-from ({t_from})"));
    }
    if t_step <= 0.0 {
        return Err(format!("--t-step must be positive, got {t_step}"));
    }
    let span = t_to - t_from;
    let n = (span / t_step).round() as i64;
    let tol = 1e-9 * span.abs().max(1.0);
    if (span - n as f64 * t_step).abs() > tol {
        return Err(format!(
            "--t-from..--t-to ({t_from}..{t_to}) is not an integer number of --t-step ({t_step}) intervals"
        ));
    }
    if n > 1_000_000 {
        return Err(format!("--t-step produces too many temperatures ({n})"));
    }
    let mut grid: Vec<f64> = (0..=n).map(|i| t_from + i as f64 * t_step).collect();
    *grid.last_mut().unwrap() = t_to; // exact endpoint (within tolerance)
    Ok(grid)
}

/// Validate every argument; Err(msg) on the first problem found.
pub fn validate(args: &Args) -> Result<(), String> {
    match args.update.as_str() {
        "metropolis" | "wolff" => {}
        other => {
            return Err(format!(
                "unknown --update {other:?} (expected \"metropolis\" or \"wolff\")"
            ))
        }
    }
    if args.l == 0 {
        return Err(format!("--l must be at least 1, got {}", args.l));
    }
    if args.measure == 0 {
        return Err(format!("--measure must be at least 1, got {}", args.measure));
    }
    temperature_grid(args.t_from, args.t_to, args.t_step).map(|_| ())
}

/// CLI entry point; returns the process exit code.
pub fn main() -> i32 {
    let args = Args::parse();
    match run_command(&args) {
        Ok(()) => 0,
        Err(msg) => {
            eprintln!("error: {msg}");
            2
        }
    }
}

fn run_command(args: &Args) -> Result<(), String> {
    validate(args)?;
    let grid = temperature_grid(args.t_from, args.t_to, args.t_step)?;
    let update = match args.update.as_str() {
        "metropolis" => Update::Metropolis,
        _ => Update::Wolff,
    };

    let mut rng = StdRng::seed_from_u64(args.seed);
    let results = run_ramp(
        args.l,
        update,
        &grid,
        args.discard,
        args.measure,
        args.every,
        &mut rng,
    );

    let run = RunConfig {
        l: args.l,
        update: args.update.clone(),
        t_grid: grid,
        discard: args.discard,
        measure: args.measure,
        seed: args.seed,
        sample_every: 1,
        time_unit: update.time_unit().to_string(),
    };
    write_artifacts(&args.out, &run, &results).map_err(|e| format!("cannot write artifacts: {e}"))?;

    // The third column is the acceptance rate (metropolis) or the mean
    // cluster size (wolff), per the design contract.
    let header = match update {
        Update::Metropolis => "T\tmean_abs_M\tacceptance",
        Update::Wolff => "T\tmean_abs_M\tmean_cluster_size",
    };
    println!("{header}");
    for r in &results {
        println!("{:.6}\t{:.6}\t{:.6}", r.t, r.mean_abs_m, r.stat);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Args {
        Args {
            update: "metropolis".to_string(),
            l: 4,
            t_from: 1.0,
            t_to: 1.0,
            t_step: 0.5,
            discard: 0,
            measure: 5,
            seed: 2026,
            every: 0,
            out: PathBuf::from("x"),
        }
    }

    #[test]
    fn grid_includes_endpoints_and_steps() {
        let g = temperature_grid(1.5, 3.5, 0.05).unwrap();
        assert_eq!(g.len(), 41);
        assert!((g[0] - 1.5).abs() < 1e-12);
        assert!((g[40] - 3.5).abs() < 1e-12);
        for w in g.windows(2) {
            assert!((w[1] - w[0] - 0.05).abs() < 1e-12);
        }
    }

    #[test]
    fn single_temperature_when_t_from_equals_t_to() {
        assert_eq!(temperature_grid(2.0, 2.0, 0.25).unwrap(), vec![2.0]);
    }

    #[test]
    fn non_integer_span_rejected_with_tolerance() {
        assert!(temperature_grid(1.0, 1.53, 0.05).is_err());
        // Tiny float noise on an exactly-representable span is accepted.
        assert!(temperature_grid(1.5, 3.5, 0.05).is_ok());
    }

    #[test]
    fn descending_or_non_positive_or_non_finite_rejected() {
        assert!(temperature_grid(2.0, 1.0, 0.1).is_err());
        assert!(temperature_grid(0.0, 1.0, 0.1).is_err());
        assert!(temperature_grid(1.0, 1.0, 0.0).is_err());
        assert!(temperature_grid(f64::NAN, 1.0, 0.1).is_err());
    }

    #[test]
    fn validate_accepts_metropolis_and_wolff() {
        let args = base_args();
        assert!(validate(&args).is_ok());

        let mut wolff = base_args();
        wolff.update = "wolff".to_string();
        assert!(validate(&wolff).is_ok());

        let mut unknown = base_args();
        unknown.update = "swendsen".to_string();
        assert!(validate(&unknown).unwrap_err().contains("unknown"));
    }
}
