use crate::features::FeatureGraph;
use crate::neural::Model;
use crate::problem::{solve, Problem, Solution, SpinGlass, EPS};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Policy<'a> {
    MostFrac,
    Strong,
    Neural(&'a Model),
}

#[derive(Clone, Copy, Serialize)]
pub struct Fix {
    spin: usize,
    bit: u8,
}

#[derive(Clone)]
struct Node {
    fixed: Vec<Option<u8>>,
    branch: Option<Fix>,
}

impl Node {
    fn children(&self, spin: usize) -> [Node; 2] {
        [0, 1].map(|bit| {
            let mut child = self.clone();
            child.fixed[spin] = Some(bit);
            child.branch = Some(Fix { spin, bit });
            child
        })
    }
}

#[derive(Serialize)]
pub struct Split {
    spin: usize,
    x: f64,
    bound: f64,
}

#[derive(Serialize)]
pub struct Trace {
    node: usize,
    branch: Option<Fix>,
    bound: Option<f64>,
    x: Option<Vec<f64>>,
    disposition: &'static str,
}

#[derive(Serialize)]
pub struct Candidate {
    spin: usize,
    features: [f64; 12],
    gain0: f64,
    gain1: f64,
    score: f64,
    pruned_children: u8,
}

#[derive(Serialize)]
pub struct Label {
    instance_id: usize,
    node: usize,
    candidates: Vec<Candidate>,
}

#[derive(Default, Serialize)]
pub struct Search {
    pub optimum: f64,
    pub ground_bits: Vec<u8>,
    pub nodes: usize,
    pub lp_solves: usize,
    pub probed_candidates: usize,
    pub lp_ms: f64,
    pub inference_ms: f64,
    pub feature_batches: usize,
    pub runtime_ms: f64,
    pub splits: Vec<Split>,
    pub trace: Vec<Trace>,
    #[serde(skip_serializing)]
    pub labels: Vec<Label>,
}

impl Search {
    fn lp(&mut self, model: &Problem, node: &Node) -> Result<Option<Solution>> {
        let t = Instant::now();
        let solved = solve(model, &node.fixed, false);
        self.lp_ms += elapsed_ms(t);
        self.lp_solves += 1;
        solved
    }

    fn trace(&mut self, node: &Node, lp: Option<&Solution>, disposition: &'static str) {
        self.trace.push(Trace {
            node: self.nodes,
            branch: node.branch,
            bound: lp.map(|s| s.value),
            x: lp.map(|s| s.x[..node.fixed.len()].to_vec()),
            disposition,
        });
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn prunable(bound: f64, incumbent: f64) -> bool {
    bound >= incumbent - EPS
}

fn probe_gain(probe: Option<Solution>, parent: f64, incumbent: f64) -> (f64, u8) {
    match probe {
        Some(lp) if !prunable(lp.value, incumbent) => ((lp.value - parent).max(0.0), 0),
        _ => (1e6, 1),
    }
}

pub fn fractionality(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    2.0 * x.min(1.0 - x)
}

fn winner(scored: impl IntoIterator<Item = (usize, f64)>, tolerance: f64) -> usize {
    let mut best = None::<(usize, f64)>;
    for (spin, score) in scored {
        if best.is_none_or(|(_, old)| score > old + tolerance) {
            best = Some((spin, score));
        }
    }
    best.expect("at least one candidate").0
}

pub fn run(
    graph: &SpinGlass,
    model: &Problem,
    policy: Policy<'_>,
    trace: bool,
    label_instance: Option<usize>,
) -> Result<Search> {
    let started = Instant::now();
    let mut search = Search::default();
    let setup = Instant::now();
    let feature_graph = (label_instance.is_some() || matches!(policy, Policy::Neural(_)))
        .then(|| FeatureGraph::new(graph));
    if matches!(policy, Policy::Neural(_)) {
        search.inference_ms += elapsed_ms(setup);
    }
    let n = graph.fields.len();
    let mut incumbent = f64::INFINITY;
    let mut stack = vec![Node {
        fixed: vec![None; n],
        branch: None,
    }];
    while let Some(node) = stack.pop() {
        search.nodes += 1;
        let Some(lp) = search.lp(model, &node)? else {
            if trace {
                search.trace(&node, None, "infeasible");
            }
            continue;
        };
        if prunable(lp.value, incumbent) {
            if trace {
                search.trace(&node, Some(&lp), "bound");
            }
            continue;
        }
        let candidates: Vec<_> = (0..n)
            .filter(|&i| node.fixed[i].is_none() && lp.x[i] > EPS && lp.x[i] < 1.0 - EPS)
            .collect();

        if candidates.is_empty() {
            let bits = graph
                .check_integer_point(&lp)
                .context("B&B returned an invalid integer LP point")?;
            ensure!(
                bits.iter()
                    .zip(&node.fixed)
                    .all(|(&bit, fix)| fix.is_none_or(|v| bit == v)),
                "integer point violates branch bounds"
            );
            let energy = graph.energy(&bits);
            if energy < incumbent {
                incumbent = energy;
                search.ground_bits = bits;
            }
            if trace {
                search.trace(&node, Some(&lp), "incumbent");
            }
            continue;
        }

        let spin = match policy {
            Policy::MostFrac => {
                winner(candidates.iter().map(|&i| (i, fractionality(lp.x[i]))), EPS)
            }
            Policy::Neural(network) => {
                let t = Instant::now();
                let features = feature_graph.as_ref().expect("neural features").extract(
                    graph,
                    &lp,
                    &node.fixed,
                    &candidates,
                    incumbent,
                );
                let spin = winner(
                    candidates
                        .iter()
                        .copied()
                        .zip(features.iter().map(|f| network.score(f))),
                    0.0,
                );
                search.inference_ms += elapsed_ms(t);
                search.feature_batches += 1;
                spin
            }
            Policy::Strong => {
                let mut probes = Vec::with_capacity(candidates.len());
                for &i in &candidates {
                    let [down, up] = node.children(i);
                    let (gain0, prune0) = probe_gain(search.lp(model, &down)?, lp.value, incumbent);
                    let (gain1, prune1) = probe_gain(search.lp(model, &up)?, lp.value, incumbent);
                    search.probed_candidates += 1;
                    probes.push((
                        gain0,
                        gain1,
                        gain0.max(EPS) * gain1.max(EPS),
                        prune0 + prune1,
                    ));
                }
                if let Some(instance_id) =
                    label_instance.filter(|_| candidates.len() >= 2 && search.labels.len() < 80)
                {
                    let features = feature_graph.as_ref().expect("label features").extract(
                        graph,
                        &lp,
                        &node.fixed,
                        &candidates,
                        incumbent,
                    );
                    let records = candidates
                        .iter()
                        .copied()
                        .zip(features)
                        .zip(&probes)
                        .map(
                            |((spin, features), &(gain0, gain1, score, pruned_children))| {
                                Candidate {
                                    spin,
                                    features,
                                    gain0,
                                    gain1,
                                    score,
                                    pruned_children,
                                }
                            },
                        )
                        .collect();
                    search.labels.push(Label {
                        instance_id,
                        node: search.nodes,
                        candidates: records,
                    });
                }
                winner(
                    candidates.iter().copied().zip(probes.iter().map(|p| p.2)),
                    1e-9,
                )
            }
        };

        if search.splits.len() < 2 {
            search.splits.push(Split {
                spin,
                x: lp.x[spin],
                bound: lp.value,
            });
        }
        if trace {
            search.trace(&node, Some(&lp), "split");
        }
        let [down, up] = node.children(spin);
        if lp.x[spin] >= 0.5 - EPS {
            stack.extend([down, up]);
        } else {
            stack.extend([up, down]);
        }
    }
    ensure!(incumbent.is_finite(), "B&B found no integer spin solution");
    search.optimum = incumbent;
    search.runtime_ms = elapsed_ms(started);
    Ok(search)
}
