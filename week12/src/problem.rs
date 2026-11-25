//! AMAT5315 Week 12, Part 1: problem representation and solving.

use anyhow::{anyhow, bail, ensure};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

pub const EPS: f64 = 1e-6;

/// A self-contained optimization problem.
///
/// - `maximize == true`: maximize `costs . x`; otherwise minimize it.
/// - `covering == true`: rows are `sum a_ij x_j >= rhs[i]` (set covering / diet).
///   Otherwise rows are `sum a_ij x_j <= rhs[i]` (workshop capacity constraints).
/// - Rows are stored sparsely as `(column index, coefficient)` pairs.
#[derive(Clone, Debug, Serialize)]
pub struct Problem {
    pub costs: Vec<f64>,
    pub rows: Vec<Vec<(usize, f64)>>,
    pub rhs: Vec<f64>,
    pub maximize: bool,
    pub covering: bool,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// An optimal solution: objective value, primal point, and LP row duals.
/// Duals are only meaningful for continuous solves; MIP duals are zero-filled.
#[derive(Clone, Debug, Serialize)]
pub struct Solution {
    pub value: f64,
    pub x: Vec<f64>,
    pub dual: Vec<f64>,
}

impl Problem {
    /// Original objective evaluated at `x` (no sign flip).
    pub fn objective(&self, x: &[f64]) -> f64 {
        self.costs.iter().zip(x.iter()).map(|(c, xi)| c * xi).sum()
    }

    /// Feasibility check within [`EPS`] tolerance.
    pub fn feasible(&self, x: &[f64]) -> bool {
        if x.len() != self.costs.len() {
            return false;
        }
        for (i, &xi) in x.iter().enumerate() {
            if !xi.is_finite() || xi < self.lower[i] - EPS || xi > self.upper[i] + EPS {
                return false;
            }
        }
        for (r, row) in self.rows.iter().enumerate() {
            let lhs: f64 = row.iter().map(|&(c, v)| x[c] * v).sum();
            if self.covering {
                if lhs < self.rhs[r] - EPS {
                    return false;
                }
            } else if lhs > self.rhs[r] + EPS {
                return false;
            }
        }
        true
    }
}

/// Solve `p` with the given variable bounds under a fresh HiGHS model.
/// `integer == true` enforces integrality on every column.
/// Returns `Ok(None)` when the problem is infeasible.
pub fn solve(
    p: &Problem,
    lower: &[f64],
    upper: &[f64],
    integer: bool,
) -> anyhow::Result<Option<Solution>> {
    ensure!(
        lower.len() == p.costs.len() && upper.len() == p.costs.len(),
        "wrong bounds length"
    );
    let mut pb = highs::RowProblem::default();
    let mut columns = Vec::with_capacity(p.costs.len());
    for (j, &obj) in p.costs.iter().enumerate() {
        let bounds = lower[j]..=upper[j];
        let column = if integer {
            pb.add_integer_column(obj, bounds)
        } else {
            pb.add_column(obj, bounds)
        };
        columns.push(column);
    }
    for (r, row) in p.rows.iter().enumerate() {
        let bounds = if p.covering {
            p.rhs[r]..=f64::INFINITY
        } else {
            f64::NEG_INFINITY..=p.rhs[r]
        };
        pb.add_row(bounds, row.iter().map(|&(j, a)| (columns[j], a)));
    }

    let sense = if p.maximize {
        highs::Sense::Maximise
    } else {
        highs::Sense::Minimise
    };
    let mut model = pb
        .try_optimise(sense)
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
            .map_err(|e| anyhow!("HiGHS relative MIP gap: {e:?}"))?;
        model
            .try_set_option("mip_abs_gap", 0.0)
            .map_err(|e| anyhow!("HiGHS absolute MIP gap: {e:?}"))?;
    }
    let response = model
        .try_solve()
        .map_err(|e| anyhow!("HiGHS solve: {e:?}"))?;
    match response.status() {
        highs::HighsModelStatus::Optimal => {
            let sol = response.get_solution();
            let x = sol.columns().to_vec();
            let dual = if integer {
                vec![0.0; p.rows.len()]
            } else {
                sol.dual_rows().to_vec()
            };
            Ok(Some(Solution {
                value: p.objective(&x),
                x,
                dual,
            }))
        }
        highs::HighsModelStatus::Infeasible => Ok(None),
        other => bail!("unexpected HiGHS status: {:?}", other),
    }
}

/// Workshop: maximize 5*x1 + 6*x2, x1+x2 <= 5, 4*x1+7*x2 <= 28, x >= 0.
pub fn workshop() -> Problem {
    Problem {
        costs: vec![5.0, 6.0],
        rows: vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 4.0), (1, 7.0)]],
        rhs: vec![5.0, 28.0],
        maximize: true,
        covering: false,
        lower: vec![0.0, 0.0],
        upper: vec![f64::INFINITY, f64::INFINITY],
    }
}

/// Diet: minimize 6*x1 + 3.5*x2, 7*x2 >= 8, 4*x1+2*x2 >= 15, 2*x1+x2 >= 3, x >= 0.
pub fn diet() -> Problem {
    Problem {
        costs: vec![6.0, 3.5],
        rows: vec![
            vec![(1, 7.0)],
            vec![(0, 4.0), (1, 2.0)],
            vec![(0, 2.0), (1, 1.0)],
        ],
        rhs: vec![8.0, 15.0, 3.0],
        maximize: false,
        covering: true,
        lower: vec![0.0, 0.0],
        upper: vec![f64::INFINITY, f64::INFINITY],
    }
}

/// Generate `count` random binary set-covering instances.
/// One ChaCha8Rng is seeded once, so instances with the same `(vars, rows, seed)`
/// share prefixes even if `count` changes.
pub fn instances(
    vars: usize,
    rows: usize,
    count: usize,
    seed: u64,
) -> anyhow::Result<Vec<Problem>> {
    if vars < 3 {
        bail!("need at least 3 variables");
    }
    if rows < 2 {
        bail!("need at least 2 rows");
    }
    if count == 0 {
        bail!("count must be positive");
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Independent Bernoulli(0.24) entries.
        let mut m = vec![vec![false; vars]; rows];
        for r in 0..rows {
            for c in 0..vars {
                if rng.gen_bool(0.24) {
                    m[r][c] = true;
                }
            }
        }
        // Add ones (never remove) until each row has >= 3 ones.
        for r in 0..rows {
            let need = 3usize.saturating_sub(m[r].iter().filter(|&&e| e).count());
            let zeros: Vec<usize> = (0..vars).filter(|&c| !m[r][c]).collect();
            for &c in zeros.choose_multiple(&mut rng, need) {
                m[r][c] = true;
            }
        }
        // Add ones until each column has >= 2 ones.
        for c in 0..vars {
            let col_degree = (0..rows).filter(|&r| m[r][c]).count();
            let need = 2usize.saturating_sub(col_degree);
            let zeros: Vec<usize> = (0..rows).filter(|&r| !m[r][c]).collect();
            for &r in zeros.choose_multiple(&mut rng, need) {
                m[r][c] = true;
            }
        }

        // Integer costs derived from the final column degrees.
        let mut col_degree = vec![0usize; vars];
        for r in 0..rows {
            for c in 0..vars {
                if m[r][c] {
                    col_degree[c] += 1;
                }
            }
        }
        let costs: Vec<f64> = col_degree
            .iter()
            .map(|&d| (2 * d + 1 + rng.gen_range(0..=3)) as f64)
            .collect();

        let mat: Vec<Vec<(usize, f64)>> = (0..rows)
            .map(|r| (0..vars).filter(|&c| m[r][c]).map(|c| (c, 1.0)).collect())
            .collect();

        out.push(Problem {
            costs,
            rows: mat,
            rhs: vec![1.0; rows],
            maximize: false,
            covering: true,
            lower: vec![0.0; vars],
            upper: vec![1.0; vars],
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{diet, solve, workshop};

    #[test]
    fn highs_matches_analytical_optima() {
        let workshop = workshop();
        let lp = solve(&workshop, &workshop.lower, &workshop.upper, false)
            .unwrap()
            .unwrap();
        let mip = solve(&workshop, &workshop.lower, &workshop.upper, true)
            .unwrap()
            .unwrap();
        assert!((lp.value - 83.0 / 3.0).abs() < 1e-4);
        assert!((mip.value - 27.0).abs() < 1e-4);
        let diet = diet();
        let lp = solve(&diet, &diet.lower, &diet.upper, false)
            .unwrap()
            .unwrap();
        assert!((lp.value - 323.0 / 14.0).abs() < 1e-4);
    }

    #[test]
    fn highs_reports_workshop_infeasible_with_rounded_lower_bounds() {
        let workshop = workshop();
        assert!(solve(&workshop, &[2.0, 3.0], &workshop.upper, false)
            .unwrap()
            .is_none());
    }
}
