//! AMAT5315 Week 12, Part 4: Rust inference for the Part 3 trained model.

use std::path::Path;

use anyhow::{bail, Context};
use serde::Deserialize;

const FEATURES: usize = 12;
const HIDDEN: usize = 48;
const PARITY_TOL: f64 = 1e-4;

/// Raw model JSON as stored by the Python trainer.
#[derive(Deserialize)]
struct ModelFile {
    schema: String,
    feature_names: Vec<String>,
    mean: Vec<f64>,
    std: Vec<f64>,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
    probes: Vec<ProbeFile>,
}

#[derive(Deserialize)]
struct ProbeFile {
    instance_id: usize,
    node: usize,
    columns: Vec<usize>,
    features: Vec<[f64; FEATURES]>,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
}

/// Validated model ready for inference.
pub struct Model {
    mean: [f64; FEATURES],
    std: [f64; FEATURES],
    w1: [[f64; HIDDEN]; FEATURES],
    b1: [f64; HIDDEN],
    w2: [f64; HIDDEN],
    b2: f64,
    probes: Vec<ProbeFile>,
}

fn check_finite(vals: &[f64], what: &str) -> anyhow::Result<()> {
    if vals.iter().any(|v| !v.is_finite()) {
        bail!("non-finite value in {what}");
    }
    Ok(())
}

fn probe_shape_err(i: usize, what: &str) -> anyhow::Error {
    anyhow::anyhow!("probe {i}: inconsistent {what} shape")
}

/// Load and fully validate a model.json produced by Part 3.
pub fn load(path: &Path) -> anyhow::Result<Model> {
    let raw: ModelFile = serde_json::from_str(
        &std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )
    .context("cannot parse model JSON")?;

    if raw.schema != "amat5315.week12.spin-glass.model.v1" {
        bail!("unexpected schema {:?}", raw.schema);
    }
    if raw.feature_names != crate::features::NAMES.to_vec() {
        bail!("feature_names do not match crate::features::NAMES");
    }
    if raw.mean.len() != FEATURES || raw.std.len() != FEATURES {
        bail!("mean/std must have length {FEATURES}");
    }
    check_finite(&raw.mean, "mean")?;
    check_finite(&raw.std, "std")?;
    if raw.std.iter().any(|&s| s <= 0.0) {
        bail!("std entries must be positive");
    }
    if raw.w1.len() != FEATURES || raw.w1.iter().any(|r| r.len() != HIDDEN) {
        bail!("w1 must be {FEATURES}x{HIDDEN}");
    }
    if raw.b1.len() != HIDDEN || raw.w2.len() != HIDDEN {
        bail!("b1/w2 must have length {HIDDEN}");
    }
    for row in &raw.w1 {
        check_finite(row, "w1")?;
    }
    check_finite(&raw.b1, "b1")?;
    check_finite(&raw.w2, "w2")?;
    check_finite(&[raw.b2], "b2")?;
    if raw.probes.len() != 6 {
        bail!("expected exactly 6 probes, found {}", raw.probes.len());
    }
    for (i, pr) in raw.probes.iter().enumerate() {
        if pr.features.len() < 2
            || pr.features.len() != pr.logits.len()
            || pr.features.len() != pr.probabilities.len()
            || pr.features.len() != pr.columns.len()
        {
            return Err(probe_shape_err(
                i,
                "candidate rows, logits and probabilities",
            ));
        }
        for features in &pr.features {
            check_finite(features, "probe features")?;
        }
        check_finite(&pr.logits, "probe logits")?;
        check_finite(&pr.probabilities, "probe probabilities")?;
    }

    let to_arr12 = |v: Vec<f64>| -> [f64; FEATURES] { v.try_into().unwrap() };
    let to_arr48 = |v: Vec<f64>| -> [f64; HIDDEN] { v.try_into().unwrap() };
    let mut w1_boxed = [[0.0; HIDDEN]; FEATURES];
    for (r, row) in raw.w1.into_iter().enumerate() {
        w1_boxed[r] = to_arr48(row);
    }

    Ok(Model {
        mean: to_arr12(raw.mean),
        std: to_arr12(raw.std),
        w1: w1_boxed,
        b1: to_arr48(raw.b1),
        w2: to_arr48(raw.w2),
        b2: raw.b2,
        probes: raw.probes,
    })
}

impl Model {
    /// Forward score of one normalized feature vector: tanh MLP scalar head.
    pub fn score(&self, input: &[f64; FEATURES]) -> f64 {
        let mut hidden = self.b1;
        for (i, x) in input.iter().enumerate() {
            let normalized = (x - self.mean[i]) / self.std[i];
            for (activation, weight) in hidden.iter_mut().zip(self.w1[i]) {
                *activation += normalized * weight;
            }
        }
        self.b2
            + hidden
                .iter()
                .zip(self.w2)
                .map(|(h, w)| h.tanh() * w)
                .sum::<f64>()
    }

    /// Numerically stable softmax over a logit slice.
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / z).collect()
    }

    /// Recompute every probe in Rust and compare against the stored JAX outputs.
    pub fn parity(&self) -> anyhow::Result<serde_json::Value> {
        let mut max_logit_err = 0.0f64;
        let mut max_prob_err = 0.0f64;
        let mut per_probe = Vec::with_capacity(self.probes.len());

        for (i, pr) in self.probes.iter().enumerate() {
            let logits: Vec<_> = pr.features.iter().map(|input| self.score(input)).collect();
            let probs = Self::softmax(&logits);
            let mut logit_error = 0.0_f64;
            let mut probability_error = 0.0_f64;
            for k in 0..logits.len() {
                if !logits[k].is_finite() || !probs[k].is_finite() {
                    bail!("probe {i}: non-finite recomputation");
                }
                let le = (logits[k] - pr.logits[k]).abs();
                let pe = (probs[k] - pr.probabilities[k]).abs();
                if le >= PARITY_TOL || pe >= PARITY_TOL {
                    bail!(
                        "probe {i} candidate {k}: parity error logit {le:.3e} prob {pe:.3e} exceeds {PARITY_TOL}"
                    );
                }
                logit_error = logit_error.max(le);
                probability_error = probability_error.max(pe);
            }
            max_logit_err = max_logit_err.max(logit_error);
            max_prob_err = max_prob_err.max(probability_error);
            per_probe.push(serde_json::json!({
                "probe": i,
                "instance_id": pr.instance_id,
                "node": pr.node,
                "columns": pr.columns,
                "rust_logits": logits,
                "rust_probabilities": probs,
                "max_abs_logit_error": logit_error,
                "max_abs_probability_error": probability_error,
            }));
        }

        Ok(serde_json::json!({
            "max_abs_logit_error": max_logit_err,
            "max_abs_probability_error": max_prob_err,
            "tolerance": PARITY_TOL,
            "probes": per_probe,
            "passed": true,
        }))
    }
}

/// Population z-scores of `scores`; all zeros if the scores are constant.
pub fn z_scores(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = scores.iter().sum::<f64>() / n as f64;
    let var = scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt().max(1e-6);
    scores.iter().map(|s| (s - mean) / std).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_scores_constant_is_zero() {
        assert_eq!(z_scores(&[3.0, 3.0, 3.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn root_normalization_floors_standard_deviation() {
        assert_eq!(z_scores(&[0.0, 1e-8]), [-0.005, 0.005]);
    }

    #[test]
    fn z_scores_population_std() {
        let z = z_scores(&[1.0, 2.0, 3.0]);
        let endpoint = (3.0_f64 / 2.0).sqrt();
        assert!((z[0] + endpoint).abs() < 1e-12);
        assert_eq!(z[1], 0.0);
        assert!((z[2] - endpoint).abs() < 1e-12);
    }
}
