use crate::bnb::{self, Policy};
use crate::features::NAMES;
use crate::problem::{self, Problem};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub fn compare(
    problems: &[Problem],
    policies: &[(&str, Policy)],
    export_labels: bool,
    tolerance: f64,
) -> Result<(Value, Option<Value>)> {
    let workshop = problem::workshop();
    let example = bnb::run(&workshop, Policy::MostFrac, None)?;
    ensure!(
        example.nodes == 3 && (example.optimum - 27.0).abs() < tolerance,
        "workshop branch-and-bound must stop at the three-node optimum"
    );

    let train_end = 3 * problems.len() / 4;
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut instances = Vec::with_capacity(problems.len());
    let mut summary = BTreeMap::<String, f64>::new();
    let mut total_error: f64 = 0.0;

    for (id, p) in problems.iter().enumerate() {
        let oracle = problem::solve(p, &p.lower, &p.upper, true)?
            .context("generated instance has no feasible integer plan")?;
        let mut results = BTreeMap::new();
        for &(name, policy) in policies {
            let mut s = bnb::run(p, policy, export_labels.then_some(id))?;
            let error = (s.optimum - oracle.value).abs();
            total_error = total_error.max(error);
            ensure!(
                error < tolerance,
                "optimum mismatch for {name} on instance {id}: {error}"
            );
            ensure!(
                s.lp_solves >= s.nodes && (name == "strong" || s.lp_solves == s.nodes),
                "invalid LP count for {name} on instance {id}"
            );
            ensure!(
                s.lp_ms >= 0.0
                    && s.inference_ms >= 0.0
                    && s.lp_ms + s.inference_ms <= s.runtime_ms + 1e-4,
                "invalid timing for {name} on instance {id}"
            );
            if export_labels {
                ensure!(s.labels.len() <= 80, "too many labels on instance {id}");
                let records = if id < train_end {
                    &mut train
                } else {
                    &mut validation
                };
                records.append(&mut s.labels);
            }
            for (metric, value) in [
                ("nodes", s.nodes as f64),
                ("lp_solves", s.lp_solves as f64),
                ("runtime_ms", s.runtime_ms),
                ("lp_ms", s.lp_ms),
                ("inference_ms", s.inference_ms),
            ] {
                *summary.entry(format!("{name}_mean_{metric}")).or_default() +=
                    value / problems.len() as f64;
            }
            let mut record = serde_json::to_value(s)?;
            record["abs_optimum_error"] = json!(error);
            results.insert(name, record);
        }
        instances.push(json!({"id": id, "problem": p, "oracle": oracle, "results": results}));
    }
    summary.insert("max_abs_optimum_error".into(), total_error);
    let dataset = export_labels.then(|| {
        json!({
            "schema": "amat5315.week12.strong-labels.v1", "feature_names": NAMES,
            "train_instance_ids": (0..train_end).collect::<Vec<_>>(),
            "validation_instance_ids": (train_end..problems.len()).collect::<Vec<_>>(),
            "train": train, "validation": validation,
        })
    });
    Ok((
        json!({"workshop": example, "instances": instances, "summary": summary}),
        dataset,
    ))
}

pub fn best_saving(run: &Value) -> Option<Value> {
    run["instances"].as_array()?.iter().filter_map(|row| {
        let most = &row["results"]["most-frac"];
        let neural = &row["results"]["neural"];
        let saving = most["nodes"].as_i64()? - neural["nodes"].as_i64()?;
        let id = row["id"].as_i64()?;
        Some(((saving, -id), json!({"id": id, "nodes_saved": saving,
            "most-frac": {"nodes": most["nodes"], "runtime_ms": most["runtime_ms"], "splits": most["splits"]},
            "neural": {"nodes": neural["nodes"], "runtime_ms": neural["runtime_ms"], "splits": neural["splits"]}})))
    }).max_by_key(|(key, _)| *key).map(|(_, row)| row)
}
