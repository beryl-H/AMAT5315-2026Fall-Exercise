mod bnb;
#[cfg(test)]
mod bnb_tests;
mod features;
mod gap;
mod neural;
mod problem;
mod study;

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
    Bnb,
    Eval,
}

#[derive(Clone, Copy, ValueEnum)]
enum Branching {
    MostFrac,
    Strong,
    Neural,
    RootNeural,
}

impl Branching {
    fn named(self, model: Option<&neural::Model>) -> Result<(&'static str, bnb::Policy<'_>)> {
        Ok(match self {
            Self::MostFrac => ("most-frac", bnb::Policy::MostFrac),
            Self::Strong => ("strong", bnb::Policy::Strong),
            Self::Neural => (
                "neural",
                bnb::Policy::Neural(model.context("neural requires --model")?),
            ),
            Self::RootNeural => (
                "root-neural",
                bnb::Policy::RootNeural(model.context("root-neural requires --model")?),
            ),
        })
    }
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
    #[arg(long, value_enum)]
    policy: Option<Branching>,
    #[arg(long)]
    export_labels: bool,
    #[arg(long)]
    model: Option<PathBuf>,
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
    let (mode, mut run, dataset) = match args.mode {
        Mode::Gap => {
            anyhow::ensure!(
                args.policy.is_none() && !args.export_labels && args.model.is_none(),
                "--mode gap accepts no branching options"
            );
            ("gap", gap::run(&graphs)?, None)
        }
        Mode::Bnb => {
            let policy = args.policy.context("--mode bnb requires --policy")?;
            anyhow::ensure!(
                !args.export_labels || (matches!(policy, Branching::Strong) && args.count == 48),
                "--export-labels requires --policy strong --count 48"
            );
            anyhow::ensure!(
                args.model.is_none() || matches!(policy, Branching::Neural | Branching::RootNeural),
                "--model is only used by neural policies"
            );
            let model = args.model.as_deref().map(neural::load).transpose()?;
            let (run, dataset) = study::compare(
                &graphs,
                &[policy.named(model.as_ref())?],
                1e-9,
                args.export_labels,
            )?;
            ("bnb", run, dataset)
        }
        Mode::Eval => {
            anyhow::ensure!(
                args.policy.is_none() && !args.export_labels,
                "--mode eval compares all policies and exports no labels"
            );
            let model = neural::load(
                args.model
                    .as_deref()
                    .context("--mode eval requires --model")?,
            )?;
            let parity = model.parity()?;
            let policies = [
                Branching::MostFrac,
                Branching::Strong,
                Branching::Neural,
                Branching::RootNeural,
            ]
            .map(|p| p.named(Some(&model)).expect("model available"));
            let (mut run, _) = study::compare(&graphs, &policies, 1e-6, false)?;
            run["summary"]["probe_max_abs_logit_error"] = parity["max_abs_logit_error"].clone();
            run["summary"]["probe_max_abs_probability_error"] =
                parity["max_abs_probability_error"].clone();
            run["probe_parity"] = parity;
            run["largest_neural_saving"] =
                study::best_saving(&run).context("missing neural result")?;
            ("eval", run, None)
        }
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
    if let Some(dataset) = dataset {
        write_json(&args.out.join("dataset.json"), &dataset)?;
    }
    println!("mode\tmetric\tvalue\truntime_ms");
    if let Some(summary) = run["summary"].as_object() {
        for (metric, value) in summary {
            println!("{mode}\t{metric}\t{value}\t{runtime_ms:.6}");
        }
    }
    if let Some(trace) = run["tetrahedron"]["trace"].as_array() {
        for (i, node) in trace.iter().enumerate() {
            println!(
                "{mode}\ttetrahedron_node_{}\t{node}\t{runtime_ms:.6}",
                i + 1
            );
        }
    }
    if mode == "eval" {
        println!(
            "{mode}\tlargest_neural_saving\t{}\t{runtime_ms:.6}",
            run["largest_neural_saving"]
        );
    }
    Ok(())
}
