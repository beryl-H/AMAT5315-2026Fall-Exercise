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
use std::io::BufWriter;
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
#[command(about = "Compare LP, MILP, and branching results")]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    vars: usize,
    #[arg(long)]
    rows: usize,
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
    use std::io::Write;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let problems = problem::instances(args.vars, args.rows, args.count, args.seed)?;
    let started = Instant::now();
    let (mode, mut run, dataset) = match args.mode {
        Mode::Gap => {
            anyhow::ensure!(
                args.policy.is_none() && !args.export_labels && args.model.is_none(),
                "--mode gap accepts no branching options"
            );
            ("gap", gap::run(&problems)?, None)
        }
        Mode::Bnb => {
            let branching = args.policy.context("--mode bnb requires --policy")?;
            anyhow::ensure!(
                !args.export_labels || matches!(branching, Branching::Strong),
                "--export-labels requires --policy strong"
            );
            if matches!(branching, Branching::MostFrac | Branching::Strong) {
                anyhow::ensure!(
                    args.model.is_none(),
                    "--model requires a neural branching policy"
                );
            }
            let model = args.model.as_deref().map(neural::load).transpose()?;
            let (run, dataset) = study::compare(
                &problems,
                &[branching.named(model.as_ref())?],
                args.export_labels,
                1e-9,
            )?;
            ("bnb", run, dataset)
        }
        Mode::Eval => {
            anyhow::ensure!(
                args.policy.is_none() && !args.export_labels,
                "--mode eval compares all policies and exports no training labels"
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
            .into_iter()
            .map(|policy| policy.named(Some(&model)))
            .collect::<Result<Vec<_>>>()?;
            let (mut run, _) = study::compare(&problems, &policies, false, 1e-4)?;
            run["probe_parity"] = parity;
            run["largest_neural_saving"] =
                study::best_saving(&run).context("missing paired neural results")?;
            ("eval", run, None)
        }
    };
    let runtime_ms = started.elapsed().as_secs_f64() * 1000.0;
    run["schema"] = json!("amat5315.week12.branchopt.v1");
    run["mode"] = json!(mode);
    run["seed"] = json!(args.seed);
    run["parameters"] = json!({"vars": args.vars, "rows": args.rows, "count": args.count,
                                "presolve": false, "threads": 1, "prng": "ChaCha8"});
    run["runtime_ms"] = json!(runtime_ms);
    fs::create_dir_all(&args.out)?;
    write_json(&args.out.join("run.json"), &run)?;
    if let Some(dataset) = dataset {
        write_json(&args.out.join("dataset.json"), &dataset)?;
    }
    println!("mode\tmetric\tvalue\truntime_ms");
    if let Some(summary) = run["summary"].as_object() {
        for (metric, value) in summary.iter().filter(|(_, v)| v.is_number()) {
            println!("{mode}\t{metric}\t{value}\t{runtime_ms:.6}");
        }
    }
    Ok(())
}
