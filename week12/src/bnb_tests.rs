use crate::bnb::{self, Policy};
use crate::problem::{self, Problem};

#[test]
fn workshop_has_three_nodes_and_one_split() {
    let search = bnb::run(&problem::workshop(), Policy::MostFrac, None).unwrap();
    assert_eq!(search.nodes, 3);
    assert_eq!(search.lp_solves, 3);
    assert_eq!(search.splits.len(), 1);
    assert_eq!(search.plan, [3.0, 2.0]);
    assert!((search.optimum - 27.0).abs() < 1e-9);
}

#[test]
fn triangle_cover_strong_scores_are_quarter_at_root() {
    let triangle = Problem {
        costs: vec![1.0; 3],
        rows: vec![
            vec![(0, 1.0), (1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0), (2, 1.0)],
        ],
        rhs: vec![1.0; 3],
        lower: vec![0.0; 3],
        upper: vec![1.0; 3],
        maximize: false,
        covering: true,
    };
    let search = bnb::run(&triangle, Policy::Strong, Some(0)).unwrap();
    assert!((search.optimum - 2.0).abs() < 1e-9);
    assert!(triangle.feasible(&search.plan));
    assert_eq!(search.lp_solves, search.nodes + 6);
    let record = serde_json::to_value(&search.labels[0]).unwrap();
    let candidates = record["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 3);
    for candidate in candidates {
        assert!((candidate["score"].as_f64().unwrap() - 0.25).abs() < 1e-9);
    }
}
