//! Command-line interface: one `md` binary with run / check / video.

use clap::{Parser, Subcommand};
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
        Command::Run(_) => todo!("md run (Task 8)"),
        Command::Check { .. } => todo!("md check (Task 10)"),
        Command::Video { .. } => todo!("md video (Task 13)"),
    }
}