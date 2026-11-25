//! AMAT5315 Week 12, Part 1: LP/MIP comparison, rounding, and greedy repair.

use crate::problem::{diet, solve, workshop, Problem, Solution, EPS};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

/// Row activity of `x` on row `r`.
fn row_activity(p: &Problem, r: usize, x: &[f64]) -> f64 {
    p.rows[r].iter().map(|&(c, v)| x[c] * v).sum()
}

/// Indices of covering rows violated by `x`.
fn uncovered_rows(p: &Problem, x: &[f64]) -> Vec<usize> {
    (0..p.rows.len())
        .filter(|&r| row_activity(p, r, x) < p.rhs[r] - EPS)
        .collect()
}

/// Greedy repair: visit rows in ascending index; when a row is uncovered,
/// add the cheapest covering column (ties broken by smallest index).
fn repair(p: &Problem, x0: &[f64]) -> Vec<f64> {
    let mut x = x0.to_vec();
    for r in 0..p.rows.len() {
        if row_activity(p, r, &x) >= p.rhs[r] - EPS {
            continue;
        }
        let best = p.rows[r]
            .iter()
            .min_by(|a, b| {
                p.costs[a.0]
                    .partial_cmp(&p.costs[b.0])
                    .unwrap()
                    .then(a.0.cmp(&b.0))
            })
            .map(|&(c, _)| c)
            .expect("each covering row must be non-empty");
        x[best] = 1.0;
    }
    x
}

fn lp_mip(p: &Problem) -> Result<(Solution, Solution)> {
    let lp = solve(p, &p.lower, &p.upper, false)?.context("expected a feasible LP")?;
    let mip = solve(p, &p.lower, &p.upper, true)?.context("expected a feasible MILP")?;
    ensure!(
        p.feasible(&mip.x) && mip.x.iter().all(|x| (x.round() - x).abs() <= EPS),
        "HiGHS integer incumbent violates bounds, integrality or rows"
    );
    Ok((lp, mip))
}

fn workshop_entry(p: &Problem) -> anyhow::Result<Value> {
    let (lp, mip) = lp_mip(p)?;
    let rounded_x: Vec<f64> = lp.x.iter().map(|&v| v.round()).collect();
    let floor_x: Vec<f64> = lp.x.iter().map(|&v| v.floor()).collect();
    let work_hours = row_activity(p, 1, &rounded_x);
    Ok(json!({
        "lp": lp,
        "mip": mip,
        "rounded": rounded_x,
        "rounded_feasible": p.feasible(&rounded_x),
        "rounded_work_hours": work_hours,
        "floor": floor_x,
        "floor_value": p.objective(&floor_x),
    }))
}

fn diet_entry(p: &Problem) -> anyhow::Result<Value> {
    let lp = solve(p, &p.lower, &p.upper, false)?.context("diet LP is infeasible")?;
    Ok(json!({ "lp": lp }))
}

fn instance_entry(id: usize, p: &Problem) -> anyhow::Result<Value> {
    let (lp, mip) = lp_mip(p).with_context(|| format!("covering instance {id}"))?;
    // Verify LP <= MIP before reporting.
    if lp.value > mip.value + EPS {
        anyhow::bail!(
            "instance {id}: LP value {} exceeds MIP value {}",
            lp.value,
            mip.value
        );
    }

    let rounded_x: Vec<f64> = lp.x.iter().map(|&v| v.round()).collect();
    let rounded_feasible = p.feasible(&rounded_x);
    let uncovered_before = uncovered_rows(p, &rounded_x);

    let repaired_x = repair(p, &rounded_x);
    let repaired_cost = p.objective(&repaired_x);
    if !p.feasible(&repaired_x) {
        anyhow::bail!("instance {id}: greedy repair produced an infeasible point");
    }
    if repaired_cost < mip.value - EPS {
        anyhow::bail!(
            "instance {id}: repaired cost {} below MIP optimum {}",
            repaired_cost,
            mip.value
        );
    }

    let denom = mip.value.abs().max(EPS);
    let relative_gap = (mip.value - lp.value) / denom;
    let rounding_overhead = (repaired_cost - mip.value) / denom;

    Ok(json!({
        "id": id,
        "problem": p,
        "lp": lp,
        "mip": mip,
        "rounded": rounded_x,
        "rounded_feasible": rounded_feasible,
        "uncovered_rows": uncovered_before,
        "repaired": repaired_x,
        "repaired_cost": repaired_cost,
        "relative_gap": relative_gap,
        "rounding_overhead": rounding_overhead,
    }))
}

/// Run the two analytical checks and the seeded set covering instances.
pub fn run(problems: &[Problem]) -> anyhow::Result<Value> {
    let instances = problems
        .iter()
        .enumerate()
        .map(|(id, p)| instance_entry(id, p))
        .collect::<Result<Vec<_>>>()?;
    let rounding_infeasible = instances
        .iter()
        .filter(|row| row["rounded_feasible"] == false)
        .count();
    let mean_relative_gap = instances
        .iter()
        .filter_map(|row| row["relative_gap"].as_f64())
        .sum::<f64>()
        / instances.len() as f64;

    Ok(json!({
        "workshop": workshop_entry(&workshop())?,
        "diet": diet_entry(&diet())?,
        "instances": instances,
        "summary": {
            "mean_relative_gap": mean_relative_gap,
            "rounding_infeasible": rounding_infeasible,
        }
    }))
}
