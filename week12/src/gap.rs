use crate::problem::{self, SpinGlass, EPS};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

fn compare(graph: &SpinGlass) -> Result<Value> {
    let model = graph.model();
    let lp = problem::solve(&model, &[], false)?.context("missing relaxed spin optimum")?;
    let mip = graph.oracle(&model)?;
    let rounded_bits = graph.round_bits(&lp.x);
    let rounded_energy = graph.energy(&rounded_bits);
    let gap = mip.value - lp.value;
    let excess = rounded_energy - mip.value;
    ensure!(
        gap >= -EPS && excess >= -EPS,
        "bounds do not enclose ground energy"
    );
    let fractional_spins = lp.x[..graph.fields.len()]
        .iter()
        .filter(|&&x| x > EPS && x < 1.0 - EPS)
        .count();
    let ground_bits = graph.round_bits(&mip.x);
    Ok(
        json!({"graph":graph, "lp":lp, "mip":mip, "ground_bits":ground_bits,
              "gap":gap, "relative_gap": gap / mip.value.abs().max(EPS),
              "fractional_spins":fractional_spins, "rounded_bits":rounded_bits,
              "rounded_energy":rounded_energy, "rounding_excess":excess}),
    )
}

pub fn run(graphs: &[SpinGlass]) -> Result<Value> {
    let square = compare(&problem::square())?;
    let diet = problem::solve(&problem::diet(), &[], false)?.context("diet has no optimum")?;
    let mut instances = Vec::new();
    for (id, graph) in graphs.iter().enumerate() {
        let mut case = compare(graph).with_context(|| format!("gap instance {id}"))?;
        case["id"] = json!(id);
        instances.push(case);
    }
    let numbers = |pointer: &'static str| {
        instances.iter().map(move |r| {
            r.pointer(pointer)
                .and_then(Value::as_f64)
                .expect("numeric gap evidence")
        })
    };
    let mean = |pointer| numbers(pointer).sum::<f64>() / graphs.len() as f64;
    Ok(json!({
        "square":square, "diet":diet, "instances":instances,
        "summary":{
            "square_lp":square["lp"]["value"], "square_mip":square["mip"]["value"],
            "square_rounded":square["rounded_energy"], "diet_optimum":diet.value,
            "mean_lp_bound":mean("/lp/value"), "mean_ground_energy":mean("/mip/value"),
            "mean_integrality_gap":mean("/gap"), "mean_relative_gap":mean("/relative_gap"),
            "mean_fractional_spins":mean("/fractional_spins"),
            "suboptimal_roundings":numbers("/rounding_excess").filter(|&v| v > EPS).count(),
            "mean_rounding_excess":mean("/rounding_excess"),
        }
    }))
}
