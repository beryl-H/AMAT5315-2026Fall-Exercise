use crate::bnb::fractionality;
use crate::problem::{Solution, SpinGlass};

pub const NAMES: [&str; 12] = [
    "x_val",
    "frac_dist",
    "field_h",
    "abs_field_h",
    "deg_norm",
    "bond_bias",
    "eff_field",
    "abs_eff_field",
    "mean_bond_dual",
    "mean_bond_gap",
    "mean_nbr_frac",
    "gap_ratio",
];

pub struct FeatureGraph {
    incident: Vec<Vec<(usize, usize)>>,
    max_degree: f64,
}

impl FeatureGraph {
    pub fn new(graph: &SpinGlass) -> Self {
        let mut incident = vec![Vec::new(); graph.fields.len()];
        for (e, bond) in graph.bonds.iter().enumerate() {
            incident[bond.u].push((e, bond.v));
            incident[bond.v].push((e, bond.u));
        }
        let max_degree = incident.iter().map(Vec::len).max().unwrap_or(0).max(1) as f64;
        Self {
            incident,
            max_degree,
        }
    }

    pub fn extract(
        &self,
        graph: &SpinGlass,
        lp: &Solution,
        fixed: &[Option<u8>],
        candidates: &[usize],
        incumbent: f64,
    ) -> Vec<[f64; 12]> {
        let n = graph.fields.len();
        let primal = |i: usize| lp.x[i].clamp(0.0, 1.0);
        let spin = |i| 1.0 - 2.0 * primal(i);
        let dual_scale = lp.dual.iter().fold(1e-9_f64, |scale, d| scale.max(d.abs()));
        let gap_ratio = if incumbent.is_finite() {
            ((incumbent - lp.value) / incumbent.abs().max(1.0)).clamp(0.0, 2.0)
        } else {
            1.0
        };
        candidates
            .iter()
            .map(|&i| {
                let local = &self.incident[i];
                let degree = local.len() as f64;
                let denominator = degree.max(1.0);
                let h = graph.fields[i];
                let mut bias = 0.0;
                let mut local_field = h;
                let mut dual = 0.0;
                let mut gap = 0.0;
                let mut nbr_frac = 0.0;
                for &(e, j) in local {
                    let coupling = graph.bonds[e].j;
                    bias +=
                        f64::from(u8::from(coupling > 0.0)) - f64::from(u8::from(coupling < 0.0));
                    local_field += coupling * spin(j);
                    dual += lp.dual[4 * e..4 * e + 4]
                        .iter()
                        .map(|d| d.abs())
                        .sum::<f64>()
                        / dual_scale;
                    gap += (lp.x[n + e] - 0.5 * (1.0 - spin(i) * spin(j))).abs();
                    if fixed[j].is_none() {
                        nbr_frac += fractionality(primal(j));
                    }
                }
                let effective = local_field / (degree + 1.0);
                [
                    primal(i),
                    fractionality(primal(i)),
                    h,
                    h.abs(),
                    degree / self.max_degree,
                    bias / denominator,
                    effective,
                    effective.abs(),
                    dual / denominator,
                    gap / denominator,
                    nbr_frac / denominator,
                    gap_ratio,
                ]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::Bond;

    #[test]
    fn features_use_fixed_neighbor_and_include_all_bond_duals() {
        let graph = SpinGlass {
            fields: vec![1.0, 0.0, -1.0],
            bonds: vec![Bond {
                u: 0,
                v: 1,
                j: -1.0,
            }],
        };
        let lp = Solution {
            value: -4.0,
            x: vec![0.5, 1.0, 0.5, 0.75],
            dual: vec![0.0, 4.0, -2.0, 0.0],
        };
        let features =
            FeatureGraph::new(&graph).extract(&graph, &lp, &[None, Some(1), None], &[0, 2], -2.0);
        assert_eq!(
            features[0],
            [0.5, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.5, 0.25, 0.0, 1.0]
        );
        assert_eq!(
            features[1],
            [0.5, 1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }
}
