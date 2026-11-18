mod entry;
mod gadgets;
mod magnet;
mod queens;

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
    #[arg(long, conflicts_with_all = ["input", "l", "beta"])]
    n: Option<usize>,
    #[arg(long, requires = "l")]
    p_af: Option<f64>,
    #[arg(long)]
    beta: Option<f64>,
    #[arg(long, value_enum)]
    order: Option<Order>,
    #[arg(long)]
    samples: Option<usize>,
}

fn main() -> Result<()> {
    let start = Instant::now();
    let args = Args::parse();
    let mode = args.mode.name();
    let order = args.order.unwrap_or(Order::Treesa);
    let samples = sample_count(&args)?;
    let (mut data, metrics) = match args.mode {
        Mode::Gadgets => {
            ensure!(
                args.input.is_none()
                    && args.l.is_none()
                    && args.beta.is_none()
                    && args.n.is_none()
                    && args.order.is_none(),
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
        Mode::Plan | Mode::Count | Mode::Sample if args.n.is_some() => {
            let n = args.n.unwrap();
            if matches!(args.mode, Mode::Plan) {
                plan_queen_network(n, order)?
            } else {
                ensure!(
                    args.order.is_none(),
                    "N-queens count/sample uses sparse row order"
                );
                queens::analyze(n, samples, args.seed)?
            }
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
            magnet::analyze(&magnet, order, beta, samples, args.seed)?
        }
    };
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;
    data["schema_version"] = json!(1);
    data["mode"] = json!(mode);
    data["seed"] = json!(args.seed);
    data["runtime_ms"] = json!(runtime_ms);
    data["parameters"] = json!({
        "input": args.input, "l": args.l, "p_af": args.p_af,
        "beta": args.beta, "samples": args.samples, "n": args.n,
        "order": data.get("planner"),
    });
    save_run(args.out, mode, data, &metrics, runtime_ms)
}

fn sample_count(args: &Args) -> Result<usize> {
    if matches!(args.mode, Mode::Sample) {
        let count = args
            .samples
            .context("--samples is required in sample mode")?;
        ensure!(count > 0, "--samples must be positive");
        Ok(count)
    } else {
        ensure!(
            args.samples.is_none(),
            "--samples is only used in sample mode"
        );
        Ok(0)
    }
}

fn plan_queen_network(n: usize, order: Order) -> Result<(Value, Vec<(&'static str, Value)>)> {
    use omeco::{
        EinCode, GreedyMethod, TreeSA, contraction_complexity, optimize_code, uniform_size_dict,
    };
    ensure!((1..=12).contains(&n), "board size must be in 1..=12");
    let mut axes = Vec::with_capacity(4 * n * n);
    let index = |r: usize, c: usize| r * n + c;
    // Four local factors per square, sharing one occupancy bit. A carry
    // label is keyed by its line family and the square where the edge starts.
    // At a line endpoint we have already pinned or summed the outer carry.
    for (family, (dr, dc)) in [(0, 1), (1, 0), (1, -1), (1, 1)].into_iter().enumerate() {
        let base = (family + 1) * n * n;
        let adjacent = |r: usize, c: usize, sign: isize| {
            let r = r.checked_add_signed(sign * dr)?;
            let c = c.checked_add_signed(sign * dc)?;
            (r < n && c < n).then_some(index(r, c))
        };
        for r in 0..n {
            for c in 0..n {
                let square = index(r, c);
                let mut factor = vec![square];
                if let Some(previous) = adjacent(r, c, -1) {
                    factor.push(base + previous);
                }
                if adjacent(r, c, 1).is_some() {
                    factor.push(base + square);
                }
                axes.push(factor);
            }
        }
    }
    let code = EinCode::new(axes, vec![]);
    let sizes = uniform_size_dict(&code, 2);
    let tree = match order {
        Order::Greedy => optimize_code(&code, &sizes, &GreedyMethod::default()),
        Order::Treesa => optimize_code(&code, &sizes, &TreeSA::default()),
    }
    .context("OMECO could not plan the carry network")?;
    let cost = contraction_complexity(&tree, &sizes, &code.ixs);
    let data = json!({"n": n, "n_variables": sizes.len(), "n_tensors": code.ixs.len(),
        "planner": order.name(), "strategy": "dense_carry_topology_estimate", "sc": cost.sc, "tc": cost.tc});
    let metrics = vec![
        ("n_variables", json!(sizes.len())),
        ("tc", json!(cost.tc)),
        ("sc", json!(cost.sc)),
    ];
    Ok((data, metrics))
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
