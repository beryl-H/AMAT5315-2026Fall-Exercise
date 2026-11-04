use anyhow::{Context, Result, ensure};
use clap::Parser;
use infer::{
    exact,
    model::{Model, parse_evidence},
    plan::{self, Plan},
};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
#[derive(Parser, Debug)]
#[command(about = "CPU tensor-network inference; UAI last scope axis varies fastest")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    evidence: String,
    #[arg(long)]
    out: PathBuf,
    #[arg(long,conflicts_with_all=["plan","analyze"])]
    order: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, conflicts_with = "analyze")]
    plan: Option<PathBuf>,
    #[arg(long)]
    analyze: Option<PathBuf>,
    #[arg(long, default_value = "exact")]
    method: String,
    #[arg(long)]
    damping: Option<f64>,
    #[arg(long)]
    tolerance: Option<f64>,
    #[arg(long)]
    max_sweeps: Option<usize>,
    #[arg(long)]
    queries: Option<String>,
    #[arg(long, default_value_t = 4)]
    trials: usize,
    #[arg(long, default_value_t = 50)]
    iterations: usize,
}
fn write(path: &std::path::Path, value: &impl serde::Serialize) -> Result<()> {
    ensure!(
        !path.exists(),
        "refusing to overwrite existing run artifact {}",
        path.display()
    );
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn main() -> Result<()> {
    let a = Args::parse();
    let start = Instant::now();
    let m = Model::parse(&std::fs::read_to_string(&a.model)?)?;
    let asia = m.dims.len() == 8;
    let e = match a.evidence.as_str() {
        "prior" => BTreeMap::new(),
        "AD" => BTreeMap::from([(0, 0), (7, 0)]),
        "ADX" => BTreeMap::from([(0, 0), (7, 0), (6, 0)]),
        p => parse_evidence(
            &std::fs::read_to_string(p)
                .context("evidence must be prior/AD/ADX or a UAI .evid path")?,
        )?,
    };
    let fs = if asia && a.method == "exact" {
        m.with_evidence_caps(&e)?
    } else {
        m.condition(&e)?
    };
    std::fs::create_dir_all(&a.out)?;
    if a.method == "bp" {
        ensure!(
            a.order.is_none() && a.plan.is_none() && a.analyze.is_none() && a.seed.is_none(),
            "ordering options apply to exact only"
        );
        let mut result = infer::bp::run(
            &m,
            &e,
            a.damping.context("bp requires damping")?,
            a.tolerance.context("bp requires tolerance")?,
            a.max_sweeps.context("bp requires max-sweeps")?,
        )?;
        result["runtime_seconds"] = serde_json::json!(start.elapsed().as_secs_f64());
        write(&a.out.join("run.json"), &result)?;
        return Ok(());
    }
    ensure!(a.method == "exact", "method must be exact or bp");
    ensure!(
        a.damping.is_none() && a.tolerance.is_none() && a.max_sweeps.is_none(),
        "BP settings require method bp"
    );
    let method = a.order.as_deref().unwrap_or("greedy");
    ensure!(
        method == "random" || method == "treesa" || a.seed.is_none(),
        "seed only valid with random or treesa"
    );
    let p = if let Some(path) = a.plan.as_ref().or(a.analyze.as_ref()) {
        serde_json::from_str::<Plan>(&std::fs::read_to_string(path)?)?
    } else {
        plan::create(&m, &fs, &[], method, a.seed, a.trials, a.iterations)?
    };
    ensure!(p.output_labels.is_empty(), "scalar evidence plan required");
    ensure!(
        p.dimensions == m.dims
            && p.input_labels == fs.iter().map(|f| f.scope.clone()).collect::<Vec<_>>(),
        "plan/model mismatch"
    );
    let complexity = plan::analyze(&p)?;
    write(&a.out.join("complexity.json"), &complexity)?;
    if a.analyze.is_some() {
        println!(
            "analysis only: work={} largest_tensor={}",
            complexity.work, complexity.largest_tensor
        );
        return Ok(());
    }
    write(&a.out.join("order.json"), &p)?;
    let scalar = exact::contract(&m, &fs, &p)?;
    let qs = if let Some(q) = a.queries {
        q.split(',')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else if asia {
        (0..8).collect()
    } else {
        vec![59, 102, 139]
    };
    ensure!(qs.iter().all(|&v| v < m.dims.len()), "query out of bounds");
    ensure!(
        !asia || qs == (0..8).collect::<Vec<_>>(),
        "ASIA requires all node queries in ID order 0,1,2,3,4,5,6,7; omit --queries for the default"
    );
    let mut marginals = vec![serde_json::Value::Null; m.dims.len()];
    let mut query_records = Vec::new();
    for &q in &qs {
        if let Some(&state) = e.get(&q) {
            let mut v = vec![0.; m.dims[q]];
            v[state] = 1.;
            marginals[q] = serde_json::json!(v);
            continue;
        }
        let qp = plan::retarget(&p, &[q])?;
        write(&a.out.join(format!("query-{q}-order.json")), &qp)?;
        let qc = plan::analyze(&qp)?;
        write(&a.out.join(format!("query-{q}-complexity.json")), &qc)?;
        let r = exact::contract(&m, &fs, &qp)?;
        marginals[q] = serde_json::json!(r.normalized);
        query_records
            .push(serde_json::json!({"variable":q,"log10_sum":r.log10_sum,"work":qc.work}));
    }
    let mut result = serde_json::json!({"method":"exact","observations":e,"model_kind":m.kind,"evidence_mass":10f64.powf(scalar.log10_sum),"log10_evidence_partition_sum":scalar.log10_sum,"log10_scale":scalar.log10_scale,"marginals":marginals,"queries":qs,"query_contractions":query_records,"work":complexity.work});
    result["evidence_quantity"] = serde_json::json!(if asia {
        "probability_for_normalized_ASIA"
    } else {
        "unnormalized_partition_sum"
    });
    result["evidence_partition_sum"] = serde_json::json!(10f64.powf(scalar.log10_sum));
    if asia {
        let jp = plan::retarget(&p, &[1, 3])?;
        write(&a.out.join("joint-tl-order.json"), &jp)?;
        write(
            &a.out.join("joint-tl-complexity.json"),
            &plan::analyze(&jp)?,
        )?;
        let j = exact::contract(&m, &fs, &jp)?.normalized;
        result["joint_tl"] = serde_json::json!([[j[0], j[1]], [j[2], j[3]]]);
        let (ze, _) = m.enumerate(&e, &[])?;
        let em = (0..8)
            .map(|q| m.enumerate(&e, &[q]).map(|r| r.1))
            .collect::<Result<Vec<_>>>()?;
        let ej = m.enumerate(&e, &[1, 3])?.1;
        let maxerr = (0..8)
            .flat_map(|q| (0..2).map(move |s| (q, s)))
            .map(|(q, s)| (marginals[q][s].as_f64().unwrap() - em[q][s]).abs())
            .fold(0f64, f64::max);
        let rel = (10f64.powf(scalar.log10_sum) - ze).abs() / ze;
        ensure!(
            maxerr < 1e-10 && rel < 1e-10,
            "enumeration validation failed"
        );
        result["enumeration"] = serde_json::json!({"evidence_mass":ze,"marginals":em,"joint_tl":[[ej[0],ej[1]],[ej[2],ej[3]]]});
        result["max_absolute_marginal_error"] = serde_json::json!(maxerr);
        result["relative_evidence_mass_error"] = serde_json::json!(rel);
        let checks = exact::sanity_checks()?;
        println!(
            "sanity checks (contraction/enumeration and absolute errors): {}",
            checks
        );
        result["sanity_checks"] = checks;
        println!(
            "evidence mass={:.17} max marginal error={:.3e} relative mass error={:.3e}",
            ze, maxerr, rel
        );
        for q in [1, 3, 4] {
            println!("disease {q}: {}", marginals[q]);
        }
    }
    result["runtime_seconds"] = serde_json::json!(start.elapsed().as_secs_f64());
    write(&a.out.join("run.json"), &result)?;
    println!(
        "log10 Z_evidence={:.17}; work={}",
        scalar.log10_sum, complexity.work
    );
    Ok(())
}
