use infer::{exact, model::Model, plan};
use std::collections::BTreeMap;
#[test]
fn scope_order_and_evidence_mass() {
    let m = Model::parse("MARKOV 2 2 2 2 1 0 2 1 0 2 0.01 0.99 4 0.05 0.01 0.95 0.99").unwrap();
    let e = BTreeMap::new();
    let (z, p) = m.enumerate(&e, &[1]).unwrap();
    assert!((z - 1.0).abs() < 1e-14);
    assert!((p[0] - 0.0104).abs() < 1e-14);
    let fs = m.condition(&e).unwrap();
    let pl = plan::create(&m, &fs, &[1], "greedy", None, 1, 20).unwrap();
    let out = exact::contract(&m, &fs, &pl).unwrap();
    assert!((out.normalized[0] - 0.0104).abs() < 1e-12);
}
#[test]
fn scalar_scale_survives_and_random_replays() {
    let m = Model::parse("MARKOV 1 2 2 0 1 0 1 1e-200 2 2e-200 3e-200").unwrap();
    let fs = m.condition(&BTreeMap::new()).unwrap();
    let p = plan::create(&m, &fs, &[], "random", Some(42), 1, 20).unwrap();
    let q: plan::Plan = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    let r = exact::contract(&m, &fs, &q).unwrap();
    assert!((r.log10_sum - (-400.0 + 5f64.log10())).abs() < 1e-12);
}
#[test]
fn invalid_tables_are_rejected() {
    assert!(Model::parse("MARKOV 1 2 1 1 0 3 0.2 0.3 0.5").is_err());
    assert!(Model::parse("MARKOV 1 2 1 1 0 2 -1 2").is_err());
}
#[test]
fn matrix_and_ising_checks() {
    let v = exact::sanity_checks().unwrap();
    assert!(v["ising_absolute_error"].as_f64().unwrap() < 1e-12);
    assert_eq!(v["matrix_ab"], serde_json::json!([4.0, 4.0, 10.0, 8.0]));
}
#[test]
fn malformed_plan_returns_error_instead_of_panicking() {
    let m = Model::parse("MARKOV 1 2 2 1 0 1 0 2 0.2 0.8 2 1 1").unwrap();
    let fs = m.condition(&BTreeMap::new()).unwrap();
    let mut p = plan::create(&m, &fs, &[], "random", Some(42), 1, 20).unwrap();
    p.dimensions.clear();
    assert!(plan::analyze(&p).is_err());
}
#[test]
fn hyperedge_and_joint_output_preserve_dependence() {
    let m = Model::parse(
        "MARKOV 3 2 2 2 3 1 0 2 1 0 2 2 0 2 0.5 0.5 4 0.9 0.1 0.1 0.9 4 0.8 0.2 0.2 0.8",
    )
    .unwrap();
    let e = BTreeMap::new();
    let fs = m.condition(&e).unwrap();
    let p = plan::create(&m, &fs, &[1, 2], "greedy", None, 1, 20).unwrap();
    let r = exact::contract(&m, &fs, &p).unwrap();
    let (_, expected) = m.enumerate(&e, &[1, 2]).unwrap();
    for (a, b) in r.normalized.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!((r.normalized[0] - 0.25).abs() > 0.1);
}
#[test]
fn isolated_variable_is_explicitly_rejected() {
    assert!(Model::parse("MARKOV 2 2 2 1 1 0 2 0.2 0.8").is_err());
}
#[test]
fn evidence_parser_accepts_both_competition_encodings() {
    assert_eq!(
        infer::model::parse_evidence("2 0 1 3 0").unwrap(),
        BTreeMap::from([(0, 1), (3, 0)])
    );
    assert_eq!(
        infer::model::parse_evidence("1 2 0 1 3 0").unwrap(),
        BTreeMap::from([(0, 1), (3, 0)])
    );
    assert!(infer::model::parse_evidence("2 0 1 0 0").is_err());
}
#[test]
fn asia_caps_preserve_evidence_and_cost_accounting() {
    let m = Model::parse(include_str!("../asia.uai")).unwrap();
    let e = BTreeMap::from([(0, 0), (7, 0)]);
    let fs = m.with_evidence_caps(&e).unwrap();
    assert_eq!(fs.len(), 10);
    for (before, after) in m.factors.iter().zip(&fs) {
        assert_eq!(before.scope, after.scope);
        assert_eq!(before.values, after.values);
    }
    let p = plan::create(&m, &fs, &[], "greedy", None, 1, 20).unwrap();
    let c = plan::analyze(&p).unwrap();
    assert!(c.work > 0);
    assert_eq!(c.largest_tensor, 8);
    let r = exact::contract(&m, &fs, &p).unwrap();
    assert!((10f64.powf(r.log10_sum) - 0.004501375).abs() < 1e-14);
}
#[test]
fn asia_partial_query_is_a_clear_cli_error_not_a_panic() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out = std::env::temp_dir().join(format!(
        "infer-asia-partial-query-{}-{stamp}",
        std::process::id()
    ));
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_infer"))
        .args([
            "--model",
            concat!(env!("CARGO_MANIFEST_DIR"), "/asia.uai"),
            "--evidence",
            "prior",
            "--queries",
            "1",
            "--out",
        ])
        .arg(out)
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success());
    assert!(error.contains("ASIA requires all node queries"), "{error}");
    assert!(!error.contains("panicked"), "{error}");
}
#[test]
fn retarget_keeps_open_queries_with_binary_space_bound_and_exact_marginals() {
    let m = Model::parse(include_str!("../asia.uai")).unwrap();
    let e = BTreeMap::from([(0, 0), (7, 0)]);
    let fs = m.with_evidence_caps(&e).unwrap();
    let scalar = plan::create(&m, &fs, &[], "greedy", None, 1, 20).unwrap();
    let base = plan::analyze(&scalar).unwrap();
    for q in [1, 2, 3, 4, 5, 6] {
        let p = plan::retarget(&scalar, &[q]).unwrap();
        let c = plan::analyze(&p).unwrap();
        assert_eq!(scalar.tree.leaf_indices(), p.tree.leaf_indices());
        assert_eq!(
            base.steps.iter().map(|s| &s.children).collect::<Vec<_>>(),
            c.steps.iter().map(|s| &s.children).collect::<Vec<_>>()
        );
        assert!(c.largest_generated_tensor <= 2 * base.largest_generated_tensor);
        assert!(c.work <= 2 * base.work);
        let r = exact::contract(&m, &fs, &p).unwrap();
        let (z, expected) = m.enumerate(&e, &[q]).unwrap();
        assert!((r.log10_sum - z.log10()).abs() < 1e-12);
        for (a, b) in r.normalized.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
    }
    let joint = plan::retarget(&scalar, &[1, 3]).unwrap();
    let r = exact::contract(&m, &fs, &joint).unwrap();
    let (_, expected) = m.enumerate(&e, &[1, 3]).unwrap();
    for (a, b) in r.normalized.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(plan::retarget(&scalar, &[8]).is_err());
}
#[test]
fn retarget_preserves_subnormal_partition_log_scale() {
    let m = Model::parse("MARKOV 1 2 2 0 1 0 1 1e-200 2 2e-200 3e-200").unwrap();
    let fs = m.condition(&BTreeMap::new()).unwrap();
    let scalar = plan::create(&m, &fs, &[], "greedy", None, 1, 20).unwrap();
    let p = plan::retarget(&scalar, &[0]).unwrap();
    let r = exact::contract(&m, &fs, &p).unwrap();
    assert!((r.log10_sum - (-400.0 + 5f64.log10())).abs() < 1e-12);
    assert!((r.normalized[0] - 0.4).abs() < 1e-12);
    assert!((r.normalized[1] - 0.6).abs() < 1e-12);
}
