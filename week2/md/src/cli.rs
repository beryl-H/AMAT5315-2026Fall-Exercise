//! Command-line interface: one `md` binary with run / check / video.

use clap::{Parser, Subcommand};
use crate::fluid::ForceMethod;
use crate::io::RunConfig;
use crate::simulate::SimConfig;
use std::path::PathBuf;

/// Parameters accepted by `md run`. Defaults are the course contract values.
#[derive(Debug, Parser)]
pub struct RunArgs {
    /// Number of atoms; must be a perfect square.
    #[arg(long, default_value_t = 100)]
    pub n: usize,
    /// Number density in reduced units.
    #[arg(long, default_value_t = 0.8)]
    pub rho: f64,
    /// Target temperature in reduced units.
    #[arg(long, default_value_t = 0.5)]
    pub temperature: f64,
    /// Time step.
    #[arg(long, default_value_t = 0.01)]
    pub dt: f64,
    /// Equilibration steps (thermostat on).
    #[arg(long, default_value_t = 2000)]
    pub eq_steps: usize,
    /// Production steps (thermostat off).
    #[arg(long, default_value_t = 10000)]
    pub steps: usize,
    /// Save every this many production steps.
    #[arg(long, default_value_t = 50)]
    pub sample_every: usize,
    /// RNG seed for the initial Gaussian velocities.
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Output directory (relative to the working directory).
    #[arg(long, default_value = "artifacts")]
    pub out: PathBuf,
    /// Force evaluation method: naive O(N^2) reference or the cell list.
    /// Final course default is "cells".
    #[arg(long, default_value = "cells")]
    pub force: String,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Simulate and record an equilibrium Lennard-Jones fluid.
    Run(RunArgs),
    /// Independently recompute the physics of a saved run.
    Check {
        /// Directory containing run.json and traj.jsonl.
        artifacts: PathBuf,
    },
    /// Render the trajectory and g(r) to an MP4 via ffmpeg.
    Video {
        /// Directory containing run.json and traj.jsonl.
        artifacts: PathBuf,
        /// Output MP4 path (required explicitly).
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Debug, Parser)]
#[command(name = "md", about = "Week 2 molecular dynamics CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// CLI entry point. Returns the process exit code.
pub fn main() -> i32 {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => match run_command(&args) {
            Ok(()) => 0,
            Err(msg) => {
                eprintln!("error: {msg}");
                2
            }
        },
        Command::Check { artifacts } => crate::checker::run_check(&artifacts),
        Command::Video { artifacts, out } => match crate::video::encode_video(&artifacts, &out) {
            Ok(()) => 0,
            Err(msg) => {
                eprintln!("error: {msg}");
                3
            }
        },
    }
}

/// Validate arguments and run the production driver, reusing the existing
/// simulation and I/O modules (no logic duplicated here).
fn run_command(args: &RunArgs) -> Result<(), String> {
    // [Suggestion] per design: clear validation errors instead of panics.
    if crate::system::side(args.n).is_none() {
        return Err(format!("--n must be a perfect square, got {}", args.n));
    }
    if args.rho <= 0.0 {
        return Err(format!("--rho must be positive, got {}", args.rho));
    }
    if args.temperature <= 0.0 {
        return Err(format!("--temperature must be positive, got {}", args.temperature));
    }
    if args.dt <= 0.0 {
        return Err(format!("--dt must be positive, got {}", args.dt));
    }
    if args.sample_every == 0 {
        return Err(format!("--sample-every must be positive, got {}", args.sample_every));
    }
    if args.steps == 0 {
        return Err(format!("--steps must be positive, got {}", args.steps));
    }
    if args.steps % args.sample_every != 0 {
        return Err(format!(
            "--steps ({}) must be a multiple of --sample-every ({}) so the final saved frame is exactly steps",
            args.steps, args.sample_every
        ));
    }

    let config = SimConfig {
        n: args.n,
        rho: args.rho,
        temperature: args.temperature,
        dt: args.dt,
        eq_steps: args.eq_steps,
        steps: args.steps,
        sample_every: args.sample_every,
        seed: args.seed,
        force_method: args
            .force
            .parse::<ForceMethod>()
            .map_err(|e| format!("invalid --force: {e}"))?,
    };
    let frames = crate::simulate::run_simulation(&config);
    crate::io::write_artifacts(&args.out, &RunConfig::from(&config), &frames)
        .map_err(|e| format!("cannot write artifacts: {e}"))?;
    println!(
        "wrote {} frames to {}",
        frames.len(),
        args.out.join("traj.jsonl").display()
    );
    Ok(())
}