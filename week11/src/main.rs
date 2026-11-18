mod gadgets;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde_json::{Value, json};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Gadgets,
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
}

fn main() -> Result<()> {
    let start = Instant::now();
    let args = Args::parse();
    let (mode, mut data) = match args.mode {
        Mode::Gadgets => ("gadgets", gadgets::evidence()),
    };
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;
    data["schema_version"] = json!(1);
    data["mode"] = json!(mode);
    data["seed"] = json!(args.seed);
    data["runtime_ms"] = json!(runtime_ms);
    let metrics = [
        (
            "circuit_states",
            json!(data["circuit"].as_array().map_or(0, Vec::len)),
        ),
        (
            "gate_count",
            json!(data["gates"].as_object().map_or(0, serde_json::Map::len)),
        ),
    ];
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
