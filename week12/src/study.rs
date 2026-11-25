use crate::bnb::{self, Policy};
use crate::features::NAMES;
use crate::problem::{self, SpinGlass};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub fn compare(
    graphs: &[SpinGlass],
    policies: &[(&str, Policy<'_>)],
    tolerance: f64,
    export_labels: bool,
) -> Result<(Value, Option<Value>)> {
    let tetra = problem::tetrahedron();
    let example = bnb::run(&tetra, &tetra.model(), Policy::MostFrac, true, None)?;
    ensure!(
        example.nodes == 5 && (example.optimum + 3.0).abs() < tolerance,
        "tetrahedron must stop at the five-node optimum"
    );
    let mut instances = Vec::with_capacity(graphs.len());
    let mut summary = BTreeMap::<String, f64>::new();
    let mut max_error = 0.0_f64;
    let mut train = Vec::new();
    let mut validation = Vec::new();
    for (id, graph) in graphs.iter().enumerate() {
        let model = graph.model();
        let oracle = graph.oracle(&model)?;
        let mut results = BTreeMap::new();
        for &(name, policy) in policies {
            let mut s = bnb::run(graph, &model, policy, false, export_labels.then_some(id))?;
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
                ("inference_ms", s.inference_ms),
                ("runtime_ms", s.runtime_ms),
            ] {
                *summary.entry(format!("{name}_mean_{key}")).or_default() += value;
            }
            if export_labels {
                let split = if id < 36 { &mut train } else { &mut validation };
                split.append(&mut s.labels);
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
    let dataset = export_labels.then(|| {
        json!({
            "schema":"amat5315.week12.spin-glass.labels.v1", "feature_names": NAMES,
            "train_instance_ids": (0..36).collect::<Vec<_>>(),
            "validation_instance_ids": (36..48).collect::<Vec<_>>(),
            "train":train, "validation":validation,
        })
    });
    Ok((
        json!({"tetrahedron":example, "instances":instances, "summary":summary}),
        dataset,
    ))
}

pub fn best_saving(run: &Value) -> Option<Value> {
    let selected = run["instances"].as_array()?.iter().max_by_key(|row| {
        let most = row["results"]["most-frac"]["nodes"].as_i64().unwrap();
        let neural = row["results"]["neural"]["nodes"].as_i64().unwrap();
        (most - neural, -row["id"].as_i64().unwrap())
    })?;
    let extract = |name| {
        let result = &selected["results"][name];
        json!({"nodes":result["nodes"], "runtime_ms":result["runtime_ms"], "splits":result["splits"]})
    };
    let most = extract("most-frac");
    let neural = extract("neural");
    Some(
        json!({"id": selected["id"], "energy": selected["oracle"]["value"],
        "nodes_saved":most["nodes"].as_i64()? - neural["nodes"].as_i64()?,
        "most-frac":most, "neural":neural}),
    )
}
