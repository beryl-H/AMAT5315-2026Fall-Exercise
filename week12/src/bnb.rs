use crate::features::{features, fractionality};
use crate::neural::{z_scores, Model};
use crate::problem::{solve, Problem, Solution, EPS};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Policy<'a> {
    MostFrac,
    Strong,
    Neural(&'a Model),
    RootNeural(&'a Model),
}

#[derive(Clone)]
struct Node {
    lower: Vec<f64>,
    upper: Vec<f64>,
    depth: usize,
}

#[derive(Serialize)]
pub struct Candidate {
    column: usize,
    features: [f64; 12],
    gain0: f64,
    gain1: f64,
    score: f64,
}

#[derive(Serialize)]
pub struct Label {
    instance_id: usize,
    node: usize,
    candidates: Vec<Candidate>,
}

#[derive(Serialize)]
pub struct Split {
    depth: usize,
    column: usize,
    x: f64,
    bound: f64,
}

#[derive(Serialize)]
pub struct Trace {
    node: usize,
    depth: usize,
    bound: Option<f64>,
    x: Option<Vec<f64>>,
    disposition: &'static str,
}

#[derive(Default, Serialize)]
pub struct Search {
    pub optimum: f64,
    pub plan: Vec<f64>,
    pub root_bound: Option<f64>,
    pub nodes: usize,
    pub lp_solves: usize,
    pub probed_candidates: usize,
    pub feature_batches: usize,
    pub lp_ms: f64,
    pub inference_ms: f64,
    pub feature_ms: f64,
    pub score_ms: f64,
    pub runtime_ms: f64,
    pub splits: Vec<Split>,
    pub trace: Vec<Trace>,
    pub labels: Vec<Label>,
}

impl Search {
    fn lp(&mut self, problem: &Problem, node: &Node) -> Result<Option<Solution>> {
        let started = Instant::now();
        let solution = solve(problem, &node.lower, &node.upper, false);
        self.lp_ms += started.elapsed().as_secs_f64() * 1000.0;
        self.lp_solves += 1;
        solution
    }

    fn trace(&mut self, node: &Node, solution: Option<&Solution>, disposition: &'static str) {
        self.trace.push(Trace {
            node: self.nodes,
            depth: node.depth,
            bound: solution.map(|s| s.value),
            x: solution.map(|s| s.x.clone()),
            disposition,
        });
    }
}

fn children(node: &Node, j: usize, split: f64) -> [Node; 2] {
    let mut down = node.clone();
    let mut up = node.clone();
    down.depth += 1;
    up.depth += 1;
    down.upper[j] = split.floor();
    up.lower[j] = split.ceil();
    [down, up]
}

fn bound_prunes(problem: &Problem, bound: f64, incumbent: Option<f64>) -> bool {
    incumbent.is_some_and(|best| {
        if problem.maximize {
            bound <= best + EPS
        } else {
            bound >= best - EPS
        }
    })
}

fn gain(problem: &Problem, probe: Option<Solution>, parent: f64, incumbent: Option<f64>) -> f64 {
    match probe {
        Some(s) if !bound_prunes(problem, s.value, incumbent) => {
            if problem.maximize {
                parent - s.value
            } else {
                s.value - parent
            }
        }
        _ => 1e6,
    }
}

fn winner(candidates: &[usize], scores: impl IntoIterator<Item = f64>) -> usize {
    candidates
        .iter()
        .copied()
        .zip(scores)
        .max_by(|(j, a), (k, b)| a.total_cmp(b).then_with(|| k.cmp(j)))
        .expect("nonempty candidates")
        .0
}

pub fn run(problem: &Problem, policy: Policy<'_>, label_instance: Option<usize>) -> Result<Search> {
    let started = Instant::now();
    let mut search = Search::default();
    let mut incumbent = None;
    let mut root_scores = Vec::new();
    let mut stack = vec![Node {
        lower: problem.lower.clone(),
        upper: problem.upper.clone(),
        depth: 0,
    }];
    while let Some(node) = stack.pop() {
        search.nodes += 1;
        let Some(lp) = search.lp(problem, &node)? else {
            if problem.maximize {
                search.trace(&node, None, "infeasible");
            }
            continue;
        };
        if node.depth == 0 {
            search.root_bound = Some(lp.value);
        }
        if bound_prunes(problem, lp.value, incumbent) {
            if problem.maximize {
                search.trace(&node, Some(&lp), "bound");
            }
            continue;
        }
        if let Policy::RootNeural(model) = policy {
            if node.depth == 0 {
                let all: Vec<_> = (0..problem.costs.len()).collect();
                let started = Instant::now();
                let f = features(problem, &lp, &node.lower, &node.upper, &all, incumbent);
                search.feature_batches += 1;
                search.feature_ms += started.elapsed().as_secs_f64() * 1000.0;
                let started = Instant::now();
                let raw: Vec<_> = f.iter().map(|x| model.score(x)).collect();
                root_scores = z_scores(&raw);
                search.score_ms += started.elapsed().as_secs_f64() * 1000.0;
            }
        }
        let candidates: Vec<_> = lp
            .x
            .iter()
            .enumerate()
            .filter(|&(j, x)| node.upper[j] - node.lower[j] > EPS && (x - x.round()).abs() > EPS)
            .map(|(j, _)| j)
            .collect();

        if candidates.is_empty() {
            let plan: Vec<_> = lp.x.iter().map(|x| x.round()).collect();
            ensure!(
                problem.feasible(&plan)
                    && plan
                        .iter()
                        .enumerate()
                        .all(|(j, &x)| x >= node.lower[j] - EPS && x <= node.upper[j] + EPS),
                "integer LP point violates the model or branch bounds"
            );
            incumbent = Some(problem.objective(&plan));
            search.plan = plan;
            if problem.maximize {
                search.trace(&node, Some(&lp), "incumbent");
            }
            continue;
        }

        let (column, floor_first) = if problem.maximize
            && node.depth == 0
            && candidates.contains(&1)
        {
            (1, true)
        } else {
            let column = match policy {
                Policy::MostFrac => winner(
                    &candidates,
                    candidates.iter().map(|&j| fractionality(lp.x[j])),
                ),
                Policy::Neural(model) => {
                    let started = Instant::now();
                    let f = features(
                        problem,
                        &lp,
                        &node.lower,
                        &node.upper,
                        &candidates,
                        incumbent,
                    );
                    search.feature_batches += 1;
                    search.feature_ms += started.elapsed().as_secs_f64() * 1000.0;
                    let started = Instant::now();
                    let column = winner(&candidates, f.iter().map(|x| model.score(x)));
                    search.score_ms += started.elapsed().as_secs_f64() * 1000.0;
                    column
                }
                Policy::RootNeural(_) => {
                    let started = Instant::now();
                    let column = winner(
                        &candidates,
                        candidates
                            .iter()
                            .map(|&j| root_scores[j] + 3.0 * fractionality(lp.x[j])),
                    );
                    search.score_ms += started.elapsed().as_secs_f64() * 1000.0;
                    column
                }
                Policy::Strong => {
                    let mut gains = Vec::with_capacity(candidates.len());
                    for &j in &candidates {
                        let [down, up] = children(&node, j, lp.x[j]);
                        let gain0 = gain(problem, search.lp(problem, &down)?, lp.value, incumbent);
                        let gain1 = gain(problem, search.lp(problem, &up)?, lp.value, incumbent);
                        search.probed_candidates += 1;
                        gains.push((gain0, gain1, gain0.max(EPS) * gain1.max(EPS)));
                    }
                    if let Some(id) =
                        label_instance.filter(|_| candidates.len() >= 2 && search.labels.len() < 80)
                    {
                        let f = features(
                            problem,
                            &lp,
                            &node.lower,
                            &node.upper,
                            &candidates,
                            incumbent,
                        );
                        let records = candidates
                            .iter()
                            .zip(&gains)
                            .zip(f)
                            .map(|((&column, &(gain0, gain1, score)), features)| Candidate {
                                column,
                                features,
                                gain0,
                                gain1,
                                score,
                            })
                            .collect();
                        search.labels.push(Label {
                            instance_id: id,
                            node: search.nodes,
                            candidates: records,
                        });
                    }
                    winner(&candidates, gains.iter().map(|&(_, _, score)| score))
                }
            };
            (column, lp.x[column].round() == lp.x[column].floor())
        };
        if search.splits.len() < 8 {
            search.splits.push(Split {
                depth: node.depth,
                column,
                x: lp.x[column],
                bound: lp.value,
            });
        }
        if problem.maximize {
            search.trace(&node, Some(&lp), "split");
        }
        let [down, up] = children(&node, column, lp.x[column]);
        if floor_first {
            stack.extend([up, down]);
        } else {
            stack.extend([down, up]);
        }
    }
    search.optimum = incumbent.context("B&B exhausted all nodes without an integer solution")?;
    search.inference_ms = search.feature_ms + search.score_ms;
    search.runtime_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(search)
}
