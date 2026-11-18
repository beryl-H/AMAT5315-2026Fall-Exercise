mod entry;
mod gadgets;
mod magnet;

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use magnet::{Magnet, Order};
use serde_json::{Value, json};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Gadgets,
    Plan,
    Count,
    Sample,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Gadgets => "gadgets",
            Self::Plan => "plan",
            Self::Count => "count",
            Self::Sample => "sample",
        }
    }
}

#[derive(Debug, Parser)]
#[command(version, about = "Week 11 exercises in reductions and tensor networks")]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, conflicts_with = "l")]
    input: Option<PathBuf>,
    #[arg(long)]
    l: Option<usize>,
    #[arg(long, requires = "l")]
    p_af: Option<f64>,
    #[arg(long)]
    beta: Option<f64>,
    #[arg(long, value_enum, default_value = "treesa")]
    order: Order,
    #[arg(long)]
    samples: Option<usize>,
}

fn main() -> Result<()> {
    let start = Instant::now();
    let args = Args::parse();
    let mode = args.mode.name();
    let (mut data, metrics) = match args.mode {
        Mode::Gadgets => {
            ensure!(
                args.input.is_none()
                    && args.l.is_none()
                    && args.beta.is_none()
                    && args.samples.is_none(),
                "gadgets mode only uses --seed and --out"
            );
            let data = gadgets::evidence();
            let metrics = vec![
                (
                    "circuit_states",
                    json!(data["circuit"].as_array().map_or(0, Vec::len)),
                ),
                (
                    "gate_count",
                    json!(data["gates"].as_object().map_or(0, serde_json::Map::len)),
                ),
            ];
            (data, metrics)
        }
        Mode::Plan | Mode::Count | Mode::Sample => {
            let magnet = match (&args.input, args.l) {
                (Some(path), None) => Magnet::bundle(path)?,
                (None, Some(l)) => Magnet::lattice(
                    l,
                    args.p_af.context("--p-af is required with --l")?,
                    args.seed,
                )?,
                _ => anyhow::bail!("provide exactly one of --input and --l"),
            };
            let beta = if matches!(args.mode, Mode::Plan) {
                ensure!(args.beta.is_none(), "plan mode does not use --beta");
                None
            } else {
                let beta = args
                    .beta
                    .context("--beta is required in magnet count/sample mode")?;
                ensure!(
                    beta.is_finite() && beta >= 0.0,
                    "--beta must be finite and nonnegative"
                );
                Some(beta)
            };
            let samples = if matches!(args.mode, Mode::Sample) {
                let count = args
                    .samples
                    .context("--samples is required in sample mode")?;
                ensure!(count > 0, "--samples must be positive");
                count
            } else {
                ensure!(
                    args.samples.is_none(),
                    "--samples is only used in sample mode"
                );
                0
            };
            magnet::analyze(&magnet, args.order, beta, samples, args.seed)?
        }
    };
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;
    data["schema_version"] = json!(1);
    data["mode"] = json!(mode);
    data["seed"] = json!(args.seed);
    data["runtime_ms"] = json!(runtime_ms);
    data["parameters"] = json!({
        "input": args.input, "l": args.l, "p_af": args.p_af,
        "beta": args.beta, "order": args.order.name(), "samples": args.samples,
    });
    save_run(args.out, mode, data, &metrics, runtime_ms)
}

fn save_run(
    out: PathBuf,
    mode: &str,
    data: Value,
    metrics: &[(&str, Value)],
    runtime_ms: f64,
) -> Result<()> {
    fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let path = out.join("run.json");
    let content = serde_json::to_string_pretty(&data)?;
    fs::write(&path, content + "\n").with_context(|| format!("writing {}", path.display()))?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for (metric, value) in metrics {
        println!("{mode}\t{metric}\t{value}\t{runtime_ms:.3}");
    }
    Ok(())
}
