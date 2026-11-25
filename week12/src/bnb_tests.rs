use crate::bnb::{self, Policy};
use crate::problem::{self, Bond, SpinGlass};
use serde_json::json;

#[test]
fn exact_tetrahedron_trace_includes_both_bound_prunes() {
    let graph = problem::tetrahedron();
    let result = bnb::run(&graph, &graph.model(), Policy::MostFrac, true, None).unwrap();
    assert_eq!(result.nodes, 5);
    assert_eq!(result.lp_solves, 5);
    assert_eq!(result.optimum, -3.0);
    assert_eq!(result.ground_bits, [1, 1, 0, 0]);
    let trace = serde_json::to_value(result.trace).unwrap();
    for (index, (branch, bound, status)) in [
        (None, -6.0, "split"),
        (Some((0, 1)), -4.0, "split"),
        (Some((1, 1)), -3.0, "incumbent"),
        (Some((1, 0)), -3.0, "bound"),
        (Some((0, 0)), -2.0, "bound"),
    ]
    .into_iter()
    .enumerate()
    {
        let row = &trace[index];
        assert_eq!(
            row["branch"],
            branch.map_or(json!(null), |(spin, bit)| json!({"spin":spin,"bit":bit}))
        );
        assert_eq!(row["disposition"], status);
        assert!((row["bound"].as_f64().unwrap() - bound).abs() < 1e-9);
    }
}

#[test]
fn both_policy_optima_match_enumeration_on_weighted_graph() {
    let graph = SpinGlass {
        fields: vec![0.25, 0.0, -0.75, 0.5],
        bonds: [
            (0, 1, 1.2),
            (0, 2, 2.2),
            (0, 3, 0.3),
            (1, 2, 0.4),
            (1, 3, -0.7),
            (2, 3, 1.3),
        ]
        .into_iter()
        .map(|(u, v, j)| Bond { u, v, j })
        .collect(),
    };
    let model = graph.model();
    let brute = (0..16)
        .map(|v| {
            let bits: Vec<u8> = (0..4).map(|i| (v >> i) & 1).collect();
            graph.energy(&bits)
        })
        .reduce(f64::min)
        .unwrap();
    let oracle = graph.oracle(&model).unwrap();
    assert!((brute - oracle.value).abs() < 1e-9);
    for policy in [Policy::MostFrac, Policy::Strong] {
        let result = bnb::run(&graph, &model, policy, false, None).unwrap();
        assert!((result.optimum - brute).abs() < 1e-9);
        assert_eq!(
            result.lp_solves,
            result.nodes + 2 * result.probed_candidates
        );
        assert_eq!(
            result.probed_candidates > 0,
            matches!(policy, Policy::Strong)
        );
    }
}
