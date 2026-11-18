use crate::entry::Entry;
use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use omeco::{
    EinCode, GreedyMethod, NestedEinsum, TreeSA, contraction_complexity, optimize_code,
    uniform_size_dict,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Order {
    Greedy,
    Treesa,
}

impl Order {
    pub fn name(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Treesa => "treesa",
        }
    }
}

#[derive(Deserialize)]
struct Bundle {
    target: Target,
}

#[derive(Deserialize)]
struct Target {
    #[serde(rename = "type")]
    kind: String,
    data: MagnetData,
}

#[derive(Deserialize)]
struct MagnetData {
    graph: Graph,
    couplings: Vec<i64>,
    fields: Vec<i64>,
}

#[derive(Deserialize)]
struct Graph {
    num_vertices: usize,
    edges: Vec<[usize; 2]>,
}

#[derive(Serialize)]
pub struct Magnet {
    n: usize,
    // Coefficients of +K*s_i*s_j and +h*s_i in the Hamiltonian.
    bonds: Vec<(usize, usize, i64)>,
    fields: Vec<i64>,
}

impl Magnet {
    pub fn bundle(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let bundle: Bundle = serde_json::from_slice(&bytes).context("decoding pred bundle")?;
        ensure!(
            bundle.target.kind == "SpinGlass",
            "bundle target must be SpinGlass"
        );
        let data = bundle.target.data;
        let n = data.graph.num_vertices;
        ensure!(
            n > 0 && data.fields.len() == n,
            "invalid graph/field dimensions"
        );
        ensure!(
            data.couplings.len() == data.graph.edges.len(),
            "missing bond coefficients"
        );
        let mut bonds = Vec::new();
        for ([a, b], k) in data.graph.edges.into_iter().zip(data.couplings) {
            ensure!(
                a < n && b < n && a != b,
                "invalid bond ({a}, {b}) for {n} spins"
            );
            bonds.push((a, b, k));
        }
        let magnitude = data
            .fields
            .iter()
            .chain(bonds.iter().map(|(_, _, k)| k))
            .map(|k| i128::from(*k).abs())
            .sum::<i128>();
        ensure!(
            magnitude <= i128::from(i64::MAX),
            "Hamiltonian coefficients exceed i64"
        );
        Ok(Self {
            n,
            bonds,
            fields: data.fields,
        })
    }

    pub fn lattice(l: usize, p_af: f64, seed: u64) -> Result<Self> {
        ensure!(l > 0 && l <= 64, "lattice size must be in 1..=64");
        ensure!((0.0..=1.0).contains(&p_af), "--p-af must be in [0, 1]");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut bonds = Vec::new();
        for row in 0..l {
            for col in 0..l {
                let a = row * l + col;
                // K = -J for the lecture's open lattice.
                for b in [
                    (col + 1 < l).then_some(a + 1),
                    (row + 1 < l).then_some(a + l),
                ]
                .into_iter()
                .flatten()
                {
                    let k = if rng.random::<f64>() < p_af { 1 } else { -1 };
                    bonds.push((a, b, k));
                }
            }
        }
        Ok(Self {
            n: l * l,
            bonds,
            fields: vec![0; l * l],
        })
    }

    fn factors(&self) -> Vec<Factor> {
        let mut factors = Vec::new();
        let mut connected = vec![false; self.n];
        for &(a, b, k) in &self.bonds {
            if k != 0 {
                connected[a] = true;
                connected[b] = true;
                factors.push(Factor {
                    axes: vec![a, b],
                    energies: vec![k, -k, -k, k],
                });
            }
        }
        for (a, &h) in self.fields.iter().enumerate() {
            if h != 0 || !connected[a] {
                factors.push(Factor {
                    axes: vec![a],
                    energies: vec![-h, h],
                });
            }
        }
        factors
    }

    fn energy(&self, bits: &[u8]) -> i64 {
        let spin = |a: usize| 2 * i64::from(bits[a]) - 1;
        let bonds = self
            .bonds
            .iter()
            .map(|&(a, b, k)| k * spin(a) * spin(b))
            .sum::<i64>();
        bonds
            + self
                .fields
                .iter()
                .enumerate()
                .map(|(a, &h)| h * spin(a))
                .sum::<i64>()
    }
}

struct Factor {
    axes: Vec<usize>,
    energies: Vec<i64>,
}

struct Child {
    node: Tensor,
    // Where the child's axes occur in the concatenation (output, summed).
    positions: Vec<usize>,
}

struct Tensor {
    axes: Vec<usize>,
    entries: Vec<Entry>,
    eliminated: Vec<usize>,
    children: Vec<Child>,
}

impl Tensor {
    fn contract(tree: &NestedEinsum<usize>, factors: &[Factor], beta: f64) -> Result<Self> {
        match tree {
            NestedEinsum::Leaf { tensor_index } => {
                let f = &factors[*tensor_index];
                Ok(Self {
                    axes: f.axes.clone(),
                    entries: f.energies.iter().copied().map(Entry::leaf).collect(),
                    eliminated: vec![],
                    children: vec![],
                })
            }
            NestedEinsum::Node { args, eins } => {
                let children = args
                    .iter()
                    .map(|t| Self::contract(t, factors, beta))
                    .collect::<Result<Vec<_>>>()?;
                Self::join(eins.iy.clone(), children, beta)
            }
        }
    }

    fn join(axes: Vec<usize>, nodes: Vec<Self>, beta: f64) -> Result<Self> {
        let mut eliminated: Vec<_> = nodes
            .iter()
            .flat_map(|n| &n.axes)
            .copied()
            .filter(|a| !axes.contains(a))
            .collect();
        eliminated.sort_unstable();
        eliminated.dedup();
        let union: Vec<_> = axes.iter().chain(&eliminated).copied().collect();
        ensure!(
            union.len() <= 23,
            "contraction node needs 2^{} local assignments; use plan mode",
            union.len()
        );
        let children: Vec<_> = nodes
            .into_iter()
            .map(|node| {
                let positions = node
                    .axes
                    .iter()
                    .map(|a| union.iter().position(|x| x == a).unwrap())
                    .collect();
                Child { node, positions }
            })
            .collect();
        let width = axes.len();
        let mut entries = Vec::with_capacity(1 << width);
        for out in 0..(1 << width) {
            let mut sum = Self::product(out, &children)?;
            for hidden in 1..(1 << eliminated.len()) {
                let product = Self::product(out | (hidden << width), &children)?;
                sum = sum.add_alternative(product, beta)?;
            }
            entries.push(sum);
        }
        Ok(Self {
            axes,
            entries,
            eliminated,
            children,
        })
    }

    fn product(full: usize, children: &[Child]) -> Result<Entry> {
        let mut result = Entry::leaf(0);
        for child in children {
            let index = child
                .positions
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &p)| acc | (((full >> p) & 1) << i));
            result = result.product(child.node.entries[index])?;
        }
        Ok(result)
    }

    fn draw(&self, bits: &mut [u8], beta: f64, rng: &mut ChaCha8Rng) -> Result<()> {
        if self.children.is_empty() {
            return Ok(());
        }
        if !self.eliminated.is_empty() {
            let width = self.axes.len();
            let out = self
                .axes
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &a)| acc | (usize::from(bits[a]) << i));
            let parent = self.entries[out];
            let mut remaining = rng.random::<f64>();
            let mut choice = None;
            for hidden in 0..(1 << self.eliminated.len()) {
                let product = Self::product(out | (hidden << width), &self.children)?;
                let delta = (i128::from(product.energy) - i128::from(parent.energy)) as f64;
                let weight = (product.log_relative - parent.log_relative - beta * delta).exp();
                if weight > 0.0 {
                    choice = Some(hidden);
                    remaining -= weight;
                    if remaining <= 0.0 {
                        break;
                    }
                }
            }
            // The last positive alternative absorbs any last-ulp rounding error.
            let choice = choice.context("conditional slice has no positive weight")?;
            for (i, &a) in self.eliminated.iter().enumerate() {
                bits[a] = ((choice >> i) & 1) as u8;
            }
        }
        for child in &self.children {
            child.node.draw(bits, beta, rng)?;
        }
        Ok(())
    }
}

pub fn analyze(
    magnet: &Magnet,
    order: Order,
    beta: Option<f64>,
    samples: usize,
    seed: u64,
) -> Result<(Value, Vec<(&'static str, Value)>)> {
    let factors = magnet.factors();
    let code = EinCode::new(factors.iter().map(|f| f.axes.clone()).collect(), vec![]);
    let sizes = uniform_size_dict(&code, 2);
    let tree_search = TreeSA::default();
    let plan = if factors.len() == 1 {
        NestedEinsum::node(vec![NestedEinsum::leaf(0)], code.clone())
    } else {
        match order {
            Order::Greedy => optimize_code(&code, &sizes, &GreedyMethod::default()),
            Order::Treesa => optimize_code(&code, &sizes, &tree_search),
        }
        .context("OMECO failed to plan contraction")?
    };
    let cost = contraction_complexity(&plan, &sizes, &code.ixs);
    let mut data = json!({
        "n_variables": magnet.n, "n_tensors": factors.len(), "magnet": magnet,
        "planner": order.name(), "omeco": "0.2.6", "sc": cost.sc, "tc": cost.tc, "rwc": cost.rwc,
    });
    let mut metrics = vec![
        ("n_variables", json!(magnet.n)),
        ("tc", json!(cost.tc)),
        ("sc", json!(cost.sc)),
    ];
    if let Some(beta) = beta {
        let tensor = Tensor::contract(&plan, &factors, beta)?;
        ensure!(
            tensor.axes.is_empty() && tensor.entries.len() == 1,
            "plan has a non-scalar root"
        );
        let res = tensor.entries[0];
        let log_z = res.log_z(beta);
        ensure!(log_z.is_finite(), "log Z is outside f64 range");
        data["e_min"] = json!(res.energy);
        data["g0"] = json!(res.count);
        data["log_z"] = json!(log_z);
        data["p0"] = json!(res.ground_probability());
        metrics.extend([
            ("e_min", json!(res.energy)),
            ("g0", json!(res.count)),
            ("log_z", json!(log_z)),
        ]);
        if samples > 0 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut draws = Vec::with_capacity(samples);
            let mut ground = 0;
            for _ in 0..samples {
                let mut bits = vec![2; magnet.n];
                tensor.draw(&mut bits, beta, &mut rng)?;
                ensure!(bits.iter().all(|&b| b < 2), "tree has an unassigned spin");
                let energy = magnet.energy(&bits);
                ground += usize::from(energy == res.energy);
                draws.push(json!({"energy": energy, "bits": bits}));
            }
            data["samples"] = json!(draws);
            data["ground_samples"] = json!(ground);
            metrics.push(("ground_samples", json!(ground)));
        }
    }
    Ok((data, metrics))
}
