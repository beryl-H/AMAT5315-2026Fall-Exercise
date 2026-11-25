use crate::bnb::{self, Policy};
use crate::problem::{self, SpinGlass};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub fn compare(graphs: &[SpinGlass], policies: &[(&str, Policy)], tolerance: f64) -> Result<Value> {
    let tetra = problem::tetrahedron();
    let example = bnb::run(&tetra, &tetra.model(), Policy::MostFrac, true)?;
    ensure!(
        example.nodes == 5 && (example.optimum + 3.0).abs() < tolerance,
        "tetrahedron must stop at the five-node optimum"
    );
    let mut instances = Vec::with_capacity(graphs.len());
    let mut summary = BTreeMap::<String, f64>::new();
    let mut max_error = 0.0_f64;
    for (id, graph) in graphs.iter().enumerate() {
        let model = graph.model();
        let oracle = graph.oracle(&model)?;
        let mut results = BTreeMap::new();
        for &(name, policy) in policies {
            let s = bnb::run(graph, &model, policy, false)?;
            let error = (s.optimum - oracle.value).abs();
            max_error = max_error.max(error);
            ensure!(
                error < tolerance,
                "optimum mismatch ({name}, instance {id}): {error}"
            );
            ensure!(
                s.lp_solves == s.nodes + 2 * s.probed_candidates
                    && (name == "strong" || s.probed_candidates == 0),
                "invalid LP count ({name}, instance {id})"
            );
            for (key, value) in [
                ("nodes", s.nodes as f64),
                ("lp_solves", s.lp_solves as f64),
                ("lp_ms", s.lp_ms),
                ("runtime_ms", s.runtime_ms),
            ] {
                *summary.entry(format!("{name}_mean_{key}")).or_default() += value;
            }
            let mut record = serde_json::to_value(s)?;
            record["abs_optimum_error"] = json!(error);
            results.insert(name, record);
        }
        instances.push(json!({"id":id, "graph":graph, "oracle":oracle, "results":results}));
    }
    for value in summary.values_mut() {
        *value /= graphs.len() as f64;
    }
    summary.insert("max_abs_optimum_error".into(), max_error);
    Ok(json!({"tetrahedron":example, "instances":instances, "summary":summary}))
}
