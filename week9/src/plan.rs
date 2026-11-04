use crate::model::{Factor, Model, entries};
use anyhow::{Context, Result, ensure};
use omeco::{EinCode, GreedyMethod, NestedEinsum, ScoreFunction, TreeSA, optimize_code};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub schema: u32,
    pub method: String,
    pub seed: Option<u64>,
    pub settings: serde_json::Value,
    pub dimensions: Vec<usize>,
    pub input_labels: Vec<Vec<usize>>,
    pub output_labels: Vec<usize>,
    pub tree: NestedEinsum<usize>,
}
#[derive(Serialize)]
pub struct Step {
    pub id: usize,
    pub children: Vec<usize>,
    pub input_labels: Vec<Vec<usize>>,
    pub retained_labels: Vec<usize>,
    pub union_labels: Vec<usize>,
    pub work: u64,
    pub entries: usize,
}
#[derive(Serialize)]
pub struct Complexity {
    pub work: u64,
    pub largest_tensor: usize,
    pub largest_generated_tensor: usize,
    pub steps: Vec<Step>,
}
fn relabel(
    t: NestedEinsum<usize>,
    ixs: &[Vec<usize>],
    out: &[usize],
    root: bool,
) -> NestedEinsum<usize> {
    let below = t.leaf_indices().into_iter().collect::<BTreeSet<_>>();
    match t {
        NestedEinsum::Leaf { .. } => t,
        NestedEinsum::Node { args, .. } => {
            let args = args
                .into_iter()
                .map(|a| relabel(a, ixs, out, false))
                .collect::<Vec<_>>();
            let inputs = args
                .iter()
                .map(|a| a.output_labels(ixs))
                .collect::<Vec<_>>();
            let outside = ixs
                .iter()
                .enumerate()
                .filter(|(i, _)| !below.contains(i))
                .flat_map(|(_, s)| s.iter())
                .chain(out.iter())
                .copied()
                .collect::<BTreeSet<_>>();
            let kept = if root {
                out.to_vec()
            } else {
                inputs
                    .iter()
                    .flatten()
                    .copied()
                    .filter(|v| outside.contains(v))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect()
            };
            NestedEinsum::node(args, EinCode::new(inputs, kept))
        }
    }
}
pub fn create(
    m: &Model,
    fs: &[Factor],
    out: &[usize],
    method: &str,
    seed: Option<u64>,
    trials: usize,
    iters: usize,
) -> Result<Plan> {
    ensure!(!fs.is_empty(), "empty factor network");
    ensure!(out.iter().all(|&v| v < m.dims.len()), "query outside model");
    ensure!(
        method != "treesa" || (trials > 0 && iters > 0),
        "TreeSA trials/iterations must be positive"
    );
    let ixs = fs.iter().map(|f| f.scope.clone()).collect::<Vec<_>>();
    let code = EinCode::new(ixs.clone(), out.to_vec());
    let sizes = m
        .dims
        .iter()
        .copied()
        .enumerate()
        .collect::<HashMap<_, _>>();
    let mut settings = serde_json::json!({});
    let tree = match method {
        "greedy" => {
            optimize_code(&code, &sizes, &GreedyMethod::default()).context("greedy failed")?
        }
        "treesa" => {
            ensure!(
                seed == Some(42),
                "omeco 0.2.6 TreeSA uses fixed trial seeds 42+trial_index; specify --seed 42"
            );
            let config = TreeSA {
                ntrials: trials,
                niters: iters,
                score: ScoreFunction::new(1.0, 0.0, 0.0, 20.0),
                ..TreeSA::default()
            };
            settings = serde_json::json!({"ntrials":trials,"niters":iters,"betas":config.betas,"initializer":"Greedy","tc_weight":1.0,"sc_weight":0.0,"rw_weight":0.0,"sc_target":20.0,"trial_seeds":(42..42+trials as u64).collect::<Vec<_>>()});
            optimize_code(&code, &sizes, &config).context("TreeSA failed")?
        }
        "random" => {
            let mut rng =
                rand::rngs::SmallRng::seed_from_u64(seed.context("random requires seed")?);
            let mut pool = (0..fs.len())
                .map(|i| (NestedEinsum::leaf(i), ixs[i].clone()))
                .collect::<Vec<_>>();
            while pool.len() > 1 {
                let i = rng.random_range(0..pool.len());
                let (a, ia) = pool.remove(i);
                let j = rng.random_range(0..pool.len());
                let (b, ib) = pool.remove(j);
                let outside = pool
                    .iter()
                    .flat_map(|(_, s)| s.iter())
                    .chain(out.iter())
                    .copied()
                    .collect::<BTreeSet<_>>();
                let iy = ia
                    .iter()
                    .chain(&ib)
                    .copied()
                    .filter(|v| outside.contains(v))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                pool.push((
                    NestedEinsum::node(vec![a, b], EinCode::new(vec![ia, ib], iy.clone())),
                    iy,
                ));
            }
            pool.remove(0).0
        }
        _ => anyhow::bail!("unknown ordering"),
    };
    // omeco can retain a label appearing in only one child until the root.
    // Keep its chosen binary topology, and apply the lecture's earliest-safe
    // summation rule consistently to both execution and independent costing.
    let tree = relabel(tree, &ixs, out, true);
    settings["label_policy"] =
        serde_json::json!("earliest safe sum; optimizer binary topology retained");
    settings["topology_source"] = serde_json::json!(if method == "random" {
        "seeded SmallRng pair selection"
    } else {
        "omeco 0.2.6 optimize_code"
    });
    settings["search_cost_vs_executed_cost"] = serde_json::json!(
        "omeco searches its native label-retention work; saved/recounted execution uses earliest-safe-sum labels; these costs need not coincide"
    );
    settings["evidence_representation"] = serde_json::json!(if fs.len() > m.factors.len() {
        "original factors plus unary one-hot evidence caps; original scopes retained"
    } else {
        "conditioned/sliced factors; observed axes removed and scalar factors retained (no change for prior)"
    });
    let p = Plan {
        schema: 1,
        method: method.into(),
        seed,
        settings,
        dimensions: m.dims.clone(),
        input_labels: ixs,
        output_labels: out.to_vec(),
        tree,
    };
    analyze(&p)?;
    Ok(p)
}
/// Reuse a scalar evidence tree with query axes left open. No optimizer is
/// rerun: only index retention changes. For binary q, every intermediate and
/// pair-union contains at most one additional size-two axis, giving <=2x bounds.
pub fn retarget(scalar: &Plan, output: &[usize]) -> Result<Plan> {
    ensure!(
        scalar.output_labels.is_empty(),
        "retarget requires a scalar evidence plan"
    );
    let source = analyze(scalar)?;
    let available = scalar
        .input_labels
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let unique = output.iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        unique.len() == output.len() && output.iter().all(|v| available.contains(v)),
        "invalid or duplicate open query axis"
    );
    let mut p = scalar.clone();
    p.output_labels = output.to_vec();
    p.tree = relabel(p.tree, &p.input_labels, output, true);
    p.settings["query_topology_source"] = serde_json::json!(
        "scalar evidence plan; identical binary children, re-canonicalized open output axes; no independent query optimizer run"
    );
    p.settings["scalar_planning_method"] = serde_json::json!(scalar.method);
    p.settings["scalar_work"] = serde_json::json!(source.work);
    p.settings["scalar_largest_generated_tensor"] =
        serde_json::json!(source.largest_generated_tensor);
    let multiplier = output
        .iter()
        .try_fold(1usize, |a, &v| a.checked_mul(p.dimensions[v]))
        .context("query bound overflow")?;
    p.settings["query_axis_size_multiplier"] = serde_json::json!(multiplier);
    let actual = analyze(&p)?;
    ensure!(
        actual.largest_generated_tensor
            <= source.largest_generated_tensor.saturating_mul(multiplier),
        "retarget storage bound violated"
    );
    ensure!(
        actual.work <= source.work.saturating_mul(multiplier as u64),
        "retarget work bound violated"
    );
    Ok(p)
}
pub fn analyze(p: &Plan) -> Result<Complexity> {
    ensure!(
        p.input_labels
            .iter()
            .flatten()
            .chain(p.output_labels.iter())
            .all(|&v| v < p.dimensions.len()),
        "plan label outside dimensions"
    );
    ensure!(p.dimensions.iter().all(|&d| d > 0), "zero plan dimension");
    let mut leaves = p.tree.leaf_indices();
    leaves.sort_unstable();
    ensure!(
        leaves == (0..p.input_labels.len()).collect::<Vec<_>>(),
        "plan must use each factor exactly once"
    );
    ensure!(
        p.tree.output_labels(&p.input_labels) == p.output_labels,
        "wrong plan output"
    );
    let mut c = Complexity {
        work: 0,
        largest_tensor: p
            .input_labels
            .iter()
            .map(|s| entries(s, &p.dimensions))
            .max()
            .unwrap_or(1),
        largest_generated_tensor: 0,
        steps: vec![],
    };
    fn visit(t: &NestedEinsum<usize>, p: &Plan, c: &mut Complexity) -> Result<(usize, Vec<usize>)> {
        match t {
            NestedEinsum::Leaf { tensor_index } => {
                Ok((*tensor_index, p.input_labels[*tensor_index].clone()))
            }
            NestedEinsum::Node { args, eins } => {
                ensure!(args.len() == 2, "nonbinary plan");
                ensure!(
                    eins.ixs
                        .iter()
                        .flatten()
                        .chain(eins.iy.iter())
                        .all(|&v| v < p.dimensions.len()),
                    "internal plan label outside dimensions"
                );
                let children = args
                    .iter()
                    .map(|a| visit(a, p, c))
                    .collect::<Result<Vec<_>>>()?;
                let labels = children.iter().map(|a| a.1.clone()).collect::<Vec<_>>();
                ensure!(labels == eins.ixs, "plan child labels disagree");
                let below = t.leaf_indices().into_iter().collect::<BTreeSet<_>>();
                let outside = p
                    .input_labels
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !below.contains(i))
                    .flat_map(|(_, s)| s.iter())
                    .chain(p.output_labels.iter())
                    .copied()
                    .collect::<BTreeSet<_>>();
                let union = labels.iter().flatten().copied().collect::<BTreeSet<_>>();
                let expected = union
                    .intersection(&outside)
                    .copied()
                    .collect::<BTreeSet<_>>();
                ensure!(
                    eins.iy.iter().copied().collect::<BTreeSet<_>>() == expected
                        && eins.iy.len() == expected.len(),
                    "plan prematurely sums or retains labels"
                );
                let union_labels = union.into_iter().collect::<Vec<_>>();
                let work = union_labels
                    .iter()
                    .try_fold(1u64, |a, &v| a.checked_mul(p.dimensions[v] as u64))
                    .context("cost overflow")?;
                let n = entries(&eins.iy, &p.dimensions);
                c.work = c.work.checked_add(work).context("work overflow")?;
                c.largest_tensor = c.largest_tensor.max(n);
                c.largest_generated_tensor = c.largest_generated_tensor.max(n);
                let id = p.input_labels.len() + c.steps.len();
                c.steps.push(Step {
                    id,
                    children: children.iter().map(|x| x.0).collect(),
                    input_labels: labels,
                    retained_labels: eins.iy.clone(),
                    union_labels,
                    work,
                    entries: n,
                });
                Ok((id, eins.iy.clone()))
            }
        }
    }
    visit(&p.tree, p, &mut c)?;
    Ok(c)
}
