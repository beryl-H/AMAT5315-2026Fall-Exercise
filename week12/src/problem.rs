use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::Serialize;
use std::collections::BTreeSet;

pub const EPS: f64 = 1e-6;

pub struct Problem {
    pub costs: Vec<f64>,
    pub rows: Vec<Vec<(usize, f64)>>,
    pub rhs: Vec<f64>,
    pub covering: bool,
    pub upper: Vec<f64>,
    pub integer_columns: usize,
    pub offset: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Solution {
    pub value: f64,
    pub x: Vec<f64>,
    #[serde(skip_serializing)]
    pub dual: Vec<f64>,
}

impl Problem {
    fn objective(&self, x: &[f64]) -> f64 {
        self.offset + self.costs.iter().zip(x).map(|(c, x)| c * x).sum::<f64>()
    }
}

/// When supplied, fixes only spin columns. Bond columns always stay continuous.
pub fn solve(p: &Problem, fixed: &[Option<u8>], integer: bool) -> Result<Option<Solution>> {
    ensure!(
        fixed.len() <= p.integer_columns,
        "too many fixed spin columns"
    );
    let mut pb = highs::RowProblem::default();
    let mut columns = Vec::with_capacity(p.costs.len());
    for (j, (&cost, &upper)) in p.costs.iter().zip(&p.upper).enumerate() {
        let bounds = match fixed.get(j).copied().flatten() {
            Some(bit) => f64::from(bit)..=f64::from(bit),
            None => 0.0..=upper,
        };
        let column = if integer && j < p.integer_columns {
            pb.add_integer_column(cost, bounds)
        } else {
            pb.add_column(cost, bounds)
        };
        columns.push(column);
    }
    for (r, row) in p.rows.iter().enumerate() {
        let rhs = p.rhs[r];
        let bounds = if p.covering {
            rhs..=f64::INFINITY
        } else {
            f64::NEG_INFINITY..=rhs
        };
        pb.add_row(bounds, row.iter().map(|&(j, a)| (columns[j], a)));
    }

    let mut model = pb
        .try_optimise(highs::Sense::Minimise)
        .map_err(|e| anyhow!("HiGHS model: {e:?}"))?;
    model
        .try_set_option("presolve", "off")
        .map_err(|e| anyhow!("HiGHS presolve: {e:?}"))?;
    model
        .try_set_option("threads", 1_i32)
        .map_err(|e| anyhow!("HiGHS threads: {e:?}"))?;
    model
        .try_set_option("output_flag", false)
        .map_err(|e| anyhow!("HiGHS output setting: {e:?}"))?;
    if integer {
        model
            .try_set_option("mip_rel_gap", 0.0)
            .map_err(|e| anyhow!("HiGHS MIP relative gap: {e:?}"))?;
        model
            .try_set_option("mip_abs_gap", 0.0)
            .map_err(|e| anyhow!("HiGHS MIP absolute gap: {e:?}"))?;
    }
    let response = model
        .try_solve()
        .map_err(|e| anyhow!("HiGHS solve: {e:?}"))?;
    match response.status() {
        highs::HighsModelStatus::Optimal => {
            let solution = response.get_solution();
            let x = solution.columns().to_vec();
            Ok(Some(Solution {
                value: p.objective(&x),
                x,
                dual: if integer {
                    Vec::new()
                } else {
                    solution.dual_rows().to_vec()
                },
            }))
        }
        highs::HighsModelStatus::Infeasible => Ok(None),
        other => bail!("unexpected HiGHS status: {other:?}"),
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Bond {
    pub u: usize,
    pub v: usize,
    pub j: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpinGlass {
    pub bonds: Vec<Bond>,
    pub fields: Vec<f64>,
}

impl SpinGlass {
    pub fn model(&self) -> Problem {
        let n = self.fields.len();
        let mut costs: Vec<_> = self.fields.iter().map(|h| -2.0 * h).collect();
        let mut rows = Vec::with_capacity(4 * self.bonds.len());
        let mut rhs = Vec::with_capacity(rows.capacity());
        let mut offset = self.fields.iter().sum::<f64>();
        for (e, &Bond { u, v, j }) in self.bonds.iter().enumerate() {
            let z = n + e;
            costs.push(-2.0 * j);
            offset += j;
            rows.extend([
                vec![(z, 1.0), (u, -1.0), (v, -1.0)],
                vec![(z, 1.0), (u, 1.0), (v, 1.0)],
                vec![(z, -1.0), (u, 1.0), (v, -1.0)],
                vec![(z, -1.0), (u, -1.0), (v, 1.0)],
            ]);
            rhs.extend([0.0, 2.0, 0.0, 0.0]);
        }
        Problem {
            upper: vec![1.0; costs.len()],
            costs,
            rows,
            rhs,
            covering: false,
            integer_columns: n,
            offset,
        }
    }

    pub fn round_bits(&self, x: &[f64]) -> Vec<u8> {
        x[..self.fields.len()]
            .iter()
            .map(|&bit| u8::from(bit >= 0.5))
            .collect()
    }

    pub fn energy(&self, bits: &[u8]) -> f64 {
        let spin = |i| 1.0 - 2.0 * f64::from(bits[i]);
        let bonds: f64 = self.bonds.iter().map(|b| b.j * spin(b.u) * spin(b.v)).sum();
        bonds
            + self
                .fields
                .iter()
                .enumerate()
                .map(|(i, h)| h * spin(i))
                .sum::<f64>()
    }

    pub fn oracle(&self, model: &Problem) -> Result<Solution> {
        let optimum = solve(model, &[], true)?.context("missing spin-glass MILP optimum")?;
        self.check_integer_point(&optimum)?;
        Ok(optimum)
    }

    pub fn check_integer_point(&self, sol: &Solution) -> Result<Vec<u8>> {
        let n = self.fields.len();
        let bits = self.round_bits(&sol.x);
        ensure!(
            bits.iter()
                .zip(&sol.x[..n])
                .all(|(&b, &x)| (f64::from(b) - x).abs() <= EPS),
            "nonintegral spin in integer point"
        );
        for (i, bond) in self.bonds.iter().enumerate() {
            let xor = f64::from(bits[bond.u] ^ bits[bond.v]);
            ensure!(
                (sol.x[n + i] - xor).abs() <= EPS,
                "bond {i} disagrees with XOR"
            );
        }
        let energy = self.energy(&bits);
        ensure!(
            (energy - sol.value).abs() <= EPS,
            "linear objective and spin energy disagree"
        );
        Ok(bits)
    }
}

pub fn square() -> SpinGlass {
    SpinGlass {
        bonds: [(0, 1, 1.0), (0, 3, -1.0), (1, 2, 1.0), (2, 3, 1.0)]
            .into_iter()
            .map(|(u, v, j)| Bond { u, v, j })
            .collect(),
        fields: vec![0.0; 4],
    }
}

pub fn diet() -> Problem {
    Problem {
        costs: vec![6.0, 3.5],
        rows: vec![
            vec![(1, 7.0)],
            vec![(0, 4.0), (1, 2.0)],
            vec![(0, 2.0), (1, 1.0)],
        ],
        rhs: vec![8.0, 15.0, 3.0],
        covering: true,
        upper: vec![f64::INFINITY; 2],
        integer_columns: 0,
        offset: 0.0,
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn draw(&mut self, k: usize) -> usize {
        (self.next() % k as u64) as usize
    }
}

fn insert_edge(edges: &mut BTreeSet<(usize, usize)>, degree: &mut [usize], i: usize, j: usize) {
    if i != j && edges.insert((i.min(j), i.max(j))) {
        degree[i] += 1;
        degree[j] += 1;
    }
}

pub fn instances(n: usize, m: usize, count: usize, seed: u64) -> Result<Vec<SpinGlass>> {
    ensure!(n > 0 && count > 0, "spins and count must be positive");
    let max_edges = n.checked_mul(n - 1).context("spin count too large")? / 2;
    let min_degree = (n - 1).min(2);
    let min_edges = n * min_degree / 2;
    ensure!(
        (min_edges..=max_edges).contains(&m),
        "edge count must be in {min_edges}..={max_edges}"
    );

    (0..count)
        .map(|id| {
            let mixed = seed ^ (id as u64).wrapping_mul(0xD1B54A32D192ED03);
            let mut rng = SplitMix64(SplitMix64(mixed).next());
            let mut edges = BTreeSet::new();
            let mut degree = vec![0; n];
            for i in 0..n {
                while degree[i] < min_degree {
                    insert_edge(&mut edges, &mut degree, i, rng.draw(n));
                    ensure!(
                        edges.len() <= m,
                        "instance {id}: degree construction requires more than {m} edges"
                    );
                }
            }
            while edges.len() < m {
                let i = rng.draw(n);
                let j = rng.draw(n);
                insert_edge(&mut edges, &mut degree, i, j);
            }
            let bonds = edges
                .into_iter()
                .map(|(u, v)| Bond {
                    u,
                    v,
                    j: if rng.next() & 1 == 0 { 1.0 } else { -1.0 },
                })
                .collect();
            let fields = (0..n).map(|_| rng.draw(3) as f64 - 1.0).collect();
            Ok(SpinGlass { bonds, fields })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diet_and_square_match_their_analytical_optima() {
        let diet = solve(&diet(), &[], false).unwrap().unwrap();
        assert!((diet.value - 323.0 / 14.0).abs() < EPS);
        let square = square();
        let model = square.model();
        let lp = solve(&model, &[], false).unwrap().unwrap();
        assert_eq!(lp.dual.len(), 16);
        assert!((lp.value + 4.0).abs() < EPS);
        let mip = square.oracle(&model).unwrap();
        assert!((mip.value + 2.0).abs() < EPS);
    }

    #[test]
    fn a_continuous_bond_is_xor_at_integral_spins() {
        let graph = SpinGlass {
            bonds: vec![Bond { u: 0, v: 1, j: 2.5 }],
            fields: vec![0.3, -0.8],
        };
        for a in 0..2 {
            for b in 0..2 {
                let sol = solve(&graph.model(), &[Some(a), Some(b)], false)
                    .unwrap()
                    .unwrap();
                assert_eq!(graph.check_integer_point(&sol).unwrap(), [a, b]);
                assert_eq!(sol.x[2], f64::from(a ^ b));
            }
        }
    }
}

pub fn tetrahedron() -> SpinGlass {
    let bonds = (0..4)
        .flat_map(|u| ((u + 1)..4).map(move |v| Bond { u, v, j: 1.0 }))
        .collect();
    SpinGlass {
        bonds,
        fields: vec![1.0, 0.0, 0.0, 0.0],
    }
}
