mod gap;
mod problem;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Gap,
}

#[derive(Parser)]
#[command(about = "Solve spin-glass integer programs and compare branching rules")]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    spins: usize,
    #[arg(long)]
    edges: usize,
    #[arg(long)]
    count: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut output = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let graphs = problem::instances(args.spins, args.edges, args.count, args.seed)?;
    let started = Instant::now();
    let (mode, mut run) = match args.mode {
        Mode::Gap => ("gap", gap::run(&graphs)?),
    };
    let runtime_ms = started.elapsed().as_secs_f64() * 1000.0;
    run["schema"] = json!("amat5315.week12.spin-glass.run.v1");
    run["mode"] = json!(mode);
    run["seed"] = json!(args.seed);
    run["parameters"] = json!({"spins": args.spins, "edges": args.edges, "count": args.count,
                               "presolve": false, "threads": 1, "prng": "SplitMix64"});
    run["runtime_ms"] = json!(runtime_ms);
    fs::create_dir_all(&args.out)?;
    write_json(&args.out.join("run.json"), &run)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    if let Some(summary) = run["summary"].as_object() {
        for (metric, value) in summary {
            println!("{mode}\t{metric}\t{value}\t{runtime_ms:.6}");
        }
    }
    Ok(())
}
