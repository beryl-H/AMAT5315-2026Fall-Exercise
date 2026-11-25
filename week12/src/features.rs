use crate::problem::{Problem, Solution, EPS};

pub const NAMES: [&str; 12] = [
    "x_val",
    "frac_dist",
    "cost_norm",
    "cost_per_uncovered",
    "uncovered_deg_norm",
    "tight_frac",
    "mean_slack",
    "mean_dual",
    "max_dual",
    "mean_nbr_frac",
    "gap_ratio",
    "frac_ratio",
];

pub fn fractionality(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    2.0 * x.min(1.0 - x)
}

#[derive(Default)]
struct ActiveRows {
    degree: usize,
    tight: usize,
    slack_sum: f64,
    dual_sum: f64,
    max_dual: f64,
    neighbor_sum: f64,
}

pub fn features(
    problem: &Problem,
    solution: &Solution,
    lower: &[f64],
    upper: &[f64],
    candidates: &[usize],
    incumbent: Option<f64>,
) -> Vec<[f64; 12]> {
    let mut by_column: Vec<_> = problem
        .costs
        .iter()
        .map(|_| ActiveRows::default())
        .collect();
    let free: Vec<_> = lower.iter().zip(upper).map(|(l, u)| u - l > EPS).collect();
    let fractional: Vec<_> = solution.x.iter().map(|x| fractionality(*x)).collect();
    let max_dual = solution.dual.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let max_cost = problem.costs.iter().copied().fold(0.0, f64::max);

    for (i, row) in problem.rows.iter().enumerate() {
        if row.iter().any(|&(j, _)| !free[j] && lower[j] >= 1.0 - EPS) {
            continue;
        }
        let activity: f64 = row
            .iter()
            .map(|&(j, coefficient)| coefficient * solution.x[j])
            .sum();
        let slack = (activity - 1.0).clamp(0.0, 2.0);
        let dual = if max_dual > 0.0 {
            solution.dual[i].abs() / max_dual
        } else {
            0.0
        };
        let (free_count, frac_sum) = row
            .iter()
            .filter(|&&(j, _)| free[j])
            .fold((0, 0.0), |(count, sum), &(j, _)| {
                (count + 1, sum + fractional[j])
            });
        for &(j, _) in row {
            let stats = &mut by_column[j];
            stats.degree += 1;
            stats.tight += usize::from(slack < 1e-5);
            stats.slack_sum += slack;
            stats.dual_sum += dual;
            stats.max_dual = stats.max_dual.max(dual);
            let neighbors = free_count - usize::from(free[j]);
            if neighbors != 0 {
                let own = if free[j] { fractional[j] } else { 0.0 };
                stats.neighbor_sum += (frac_sum - own) / neighbors as f64;
            }
        }
    }

    let max_degree = candidates
        .iter()
        .map(|&j| by_column[j].degree)
        .max()
        .unwrap_or(0)
        .max(1);
    let (unit_sum, unit_count) = candidates.iter().fold((0.0, 0), |(sum, count), &j| {
        let degree = by_column[j].degree;
        if degree == 0 {
            (sum, count)
        } else {
            (sum + problem.costs[j] / degree as f64, count + 1)
        }
    });
    let mean_unit = if unit_count == 0 {
        1.0
    } else {
        unit_sum / unit_count as f64
    };
    let gap_ratio = incumbent.map_or(1.0, |value| {
        ((value - solution.value) / value.abs().max(1.0)).clamp(0.0, 2.0)
    });
    let frac_ratio = candidates.len() as f64 / free.iter().filter(|&&v| v).count().max(1) as f64;
    candidates
        .iter()
        .map(|&j| {
            let s = &by_column[j];
            let norm = s.degree.max(1) as f64;
            let unit = if s.degree == 0 {
                0.0
            } else {
                (problem.costs[j] / norm / mean_unit).clamp(0.0, 4.0)
            };
            [
                solution.x[j].clamp(0.0, 1.0),
                fractional[j],
                problem.costs[j] / max_cost,
                unit,
                s.degree as f64 / max_degree as f64,
                s.tight as f64 / norm,
                s.slack_sum / norm,
                s.dual_sum / norm,
                s.max_dual,
                s.neighbor_sum / norm,
                gap_ratio,
                frac_ratio,
            ]
        })
        .collect()
}
