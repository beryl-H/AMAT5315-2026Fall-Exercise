//! Sum-product belief propagation (BP) on discrete factor graphs.
//!
//! Conventions (learning sheet, part 4):
//! - Observations are applied by slicing the observed dimensions out of every
//!   factor table (C-order flattening: last scope variable fastest). Observed
//!   variables leave the factor graph; their beliefs are one-hot. Sliced
//!   factors are normalized to unit mass (BP is invariant to per-factor
//!   scaling, this only helps numerical stability). The input model is never
//!   mutated.
//! - One message lives on each directed edge of the factor graph induced by
//!   the unobserved variables. Every message is a normalized nonnegative
//!   vector over the states of its connecting variable. Initial messages are
//!   uniform.
//! - Updates are synchronous: each undamped proposal map evaluation F(m)
//!   reads ONLY the old messages `m`, and both update types (variable->factor
//!   and factor->variable) omit the recipient's own incoming message.
//! - One sweep applies the damped map m' = (1-damping)*m + damping*F(m).
//! - Every recorded state (initial state 0, each post-sweep state, and the
//!   returned final state) reports beliefs computed from that state's
//!   factor->variable messages together with the undamped fixed-point
//!   residual max|F(m) - m| evaluated at that SAME state.
//!
//! Work convention (`update_work`, cumulative): the count of floating-point
//! multiplications performed while forming message and belief products, i.e.
//! every time a message value or factor-table entry is multiplied into a
//! running product or accumulator during (a) variable->factor proposals,
//! (b) factor->variable proposals and (c) belief products. Because each
//! recorded residual max|F(m)-m| requires one extra undamped proposal
//! evaluation, those multiplications are included as well (one F evaluation
//! per recorded state; it doubles as the proposal for the next damped step).
//! Normalization divisions, damping arithmetic and the residual's absolute
//! differences are not counted.

use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::time::Instant;

use crate::model::{Factor, Model};

/// Run synchronous damped sum-product BP.
///
/// Returns a JSON report with method, observations, settings, the final
/// marginals for all original variables (in variable order, one probability
/// vector each), a per-sweep trace `{sweep, residual, marginals,
/// cumulative_update_work}`, the final residual, a `converged` flag (residual
/// dropped below `tolerance`), the number of applied sweeps, the runtime in
/// seconds and the cumulative `update_work`.
pub fn run(
    model: &Model,
    evidence: &BTreeMap<usize, usize>,
    damping: f64,
    tolerance: f64,
    max_sweeps: usize,
) -> Result<Value> {
    // --- settings validation ------------------------------------------------
    ensure!(
        damping.is_finite() && (0.0..=1.0).contains(&damping),
        "damping must be a finite fraction in [0, 1], got {damping}"
    );
    ensure!(
        tolerance.is_finite() && tolerance > 0.0,
        "tolerance must be finite and positive, got {tolerance}"
    );
    ensure!(max_sweeps >= 1, "max_sweeps must be at least 1");
    for (&v, &s) in evidence {
        ensure!(v < model.dims.len(), "evidence variable {v} out of range");
        ensure!(
            s < model.dims[v],
            "evidence state {s} out of range for variable {v}"
        );
    }

    let start = Instant::now();

    // --- factor graph after slicing observed dimensions ---------------------
    let factors = slice_factors(model, evidence)?;
    let dims = &model.dims;
    // incident[v] = list of (factor index, position of v in that factor's scope)
    let mut incident: Vec<Vec<(usize, usize)>> = vec![Vec::new(); dims.len()];
    for (fi, f) in factors.iter().enumerate() {
        for (p, &v) in f.scope.iter().enumerate() {
            incident[v].push((fi, p));
        }
    }

    // --- messages: uniform, normalized on every directed edge ---------------
    let empty = || Messages {
        v2f: factors
            .iter()
            .map(|f| {
                f.scope
                    .iter()
                    .map(|&v| vec![1.0 / dims[v] as f64; dims[v]])
                    .collect()
            })
            .collect(),
        f2v: factors
            .iter()
            .map(|f| {
                f.scope
                    .iter()
                    .map(|&v| vec![1.0 / dims[v] as f64; dims[v]])
                    .collect()
            })
            .collect(),
    };
    let mut m = empty();
    let mut scratch = empty(); // reused buffer for F(m)

    let mut work: u64 = 0;
    let mut trace: Vec<Value> = Vec::new();

    // Record the initial state: beliefs from m's factor->variable messages and
    // the undamped residual max|F(m) - m| at the SAME state. The F evaluation
    // is kept as the proposal for the first damped sweep.
    propose(dims, &factors, &incident, &m, &mut scratch, &mut work)?;
    let (entry, mut residual) =
        record_state(0, dims, &incident, evidence, &m, &scratch, &mut work)?;
    trace.push(entry);
    let mut converged = residual < tolerance;
    let mut sweeps: usize = 0;

    if !converged {
        for s in 1..=max_sweeps {
            // Damped synchronous step; `scratch` currently holds F(m).
            apply_damped(&mut m, &scratch, damping);
            sweeps = s;
            // F(m_new) for this state's residual; reused as the next proposal.
            propose(dims, &factors, &incident, &m, &mut scratch, &mut work)?;
            let (entry, res) = record_state(s, dims, &incident, evidence, &m, &scratch, &mut work)?;
            println!(
                "bp sweep {}: residual={res:.6e} cumulative_update_work={work}",
                entry["sweep"]
            );
            trace.push(entry);
            residual = res;
            if res < tolerance {
                converged = true;
                break;
            }
        }
    }

    let final_marginals = trace
        .last()
        .and_then(|e| e.get("marginals").cloned())
        .unwrap_or_else(|| json!(beliefs_for(dims, &incident, evidence, &m, &mut 0).ok()));

    let observations: Map<String, Value> = evidence
        .iter()
        .map(|(&v, &s)| (v.to_string(), json!(s)))
        .collect();

    Ok(json!({
        "method": "bp",
        "observations": observations,
        "settings": {
            "damping": damping,
            "tolerance": tolerance,
            "max_sweeps": max_sweeps,
        },
        "marginals": final_marginals,
        "trace": trace,
        "residual": residual,
        "converged": converged,
        "sweeps": sweeps,
        "runtime_seconds": start.elapsed().as_secs_f64(),
        "update_work": work,
        "update_work_convention": "cumulative count of floating-point multiplications used to form message and belief products during undamped proposal evaluations F(m) and belief products; each recorded residual requires one F evaluation, so its multiplications are included; normalization divisions, damping arithmetic and residual differences are not counted",
    }))
}

/// All messages of one BP state, indexed [factor][scope position][state].
struct Messages {
    v2f: Vec<Vec<Vec<f64>>>,
    f2v: Vec<Vec<Vec<f64>>>,
}

/// Slice observed dimensions out of every factor table (C-order: last scope
/// variable fastest) and normalize each sliced factor to unit mass. Scalar
/// factors are retained. Zero-mass sliced factors mean impossible evidence.
fn slice_factors(model: &Model, evidence: &BTreeMap<usize, usize>) -> Result<Vec<Factor>> {
    let mut out = Vec::with_capacity(model.factors.len());
    for f in &model.factors {
        let new_scope: Vec<usize> = f
            .scope
            .iter()
            .copied()
            .filter(|v| !evidence.contains_key(v))
            .collect();
        let n: usize = new_scope.iter().map(|&v| model.dims[v]).product();
        let mut vals = vec![0.0f64; n];
        for (i, &p) in f.values.iter().enumerate() {
            let x = decode_index(i, &f.scope, &model.dims);
            let mut keep = true;
            for (pos, &v) in f.scope.iter().enumerate() {
                if let Some(&s) = evidence.get(&v) {
                    if x[pos] != s {
                        keep = false;
                        break;
                    }
                }
            }
            if !keep {
                continue;
            }
            let mut j = 0usize;
            for (pos, &v) in f.scope.iter().enumerate() {
                if !evidence.contains_key(&v) {
                    j = j * model.dims[v] + x[pos];
                }
            }
            vals[j] = p;
        }
        let sum: f64 = vals.iter().sum();
        ensure!(
            sum.is_finite() && sum > 0.0,
            "factor on scope {:?} has zero or non-finite mass under the observations",
            f.scope
        );
        for x in vals.iter_mut() {
            *x /= sum;
        }
        out.push(Factor {
            scope: new_scope,
            values: vals,
        });
    }
    Ok(out)
}

/// C-order decode of a flat table index into per-variable coordinates.
fn decode_index(mut i: usize, scope: &[usize], dims: &[usize]) -> Vec<usize> {
    let mut x = vec![0usize; scope.len()];
    for k in (0..scope.len()).rev() {
        x[k] = i % dims[scope[k]];
        i /= dims[scope[k]];
    }
    x
}

/// Normalize in place; error on zero or non-finite sums.
fn normalize(vec: &mut [f64], what: &str) -> Result<()> {
    let sum: f64 = vec.iter().sum();
    ensure!(
        sum.is_finite() && sum > 0.0,
        "{what} has zero or non-finite sum"
    );
    for x in vec.iter_mut() {
        *x /= sum;
    }
    Ok(())
}

/// One undamped synchronous proposal evaluation F(m). Reads ONLY `m` and
/// writes normalized proposals into `out`; both update types omit the
/// recipient's own incoming message. Counts multiplications into `work`.
fn propose(
    dims: &[usize],
    factors: &[Factor],
    incident: &[Vec<(usize, usize)>],
    m: &Messages,
    out: &mut Messages,
    work: &mut u64,
) -> Result<()> {
    // Variable -> factor: product of all incoming factor->variable messages
    // except the recipient's own (empty product is the uniform vector).
    for (v, edges) in incident.iter().enumerate() {
        let d = dims[v];
        for &(f, p) in edges {
            let mut acc = vec![1.0f64; d];
            for &(g, q) in edges {
                if g == f && q == p {
                    continue;
                }
                let src = &m.f2v[g][q];
                for (x, a) in acc.iter_mut().enumerate() {
                    *a *= src[x];
                    *work += 1;
                }
            }
            normalize(&mut acc, "variable->factor message proposal")?;
            out.v2f[f][p].copy_from_slice(&acc);
        }
    }

    // Factor -> variable: one pass over the table entries. Prefix/suffix
    // products give, for every entry and every recipient, the product of all
    // incoming variable->factor messages except the recipient's own.
    for (fi, f) in factors.iter().enumerate() {
        let k = f.scope.len();
        // Clear the reused output buffers before accumulating F(m).
        for p in 0..k {
            for x in out.f2v[fi][p].iter_mut() {
                *x = 0.0;
            }
        }
        let mut idx = vec![0usize; k];
        let mut pre = vec![1.0f64; k + 1];
        let mut suf = vec![1.0f64; k + 1];
        for (i, &val) in f.values.iter().enumerate() {
            if i > 0 {
                // Odometer increment over the C-order entry index.
                for j in (0..k).rev() {
                    idx[j] += 1;
                    if idx[j] < dims[f.scope[j]] {
                        break;
                    }
                    idx[j] = 0;
                }
            }
            pre[0] = 1.0;
            for j in 0..k {
                pre[j + 1] = pre[j] * m.v2f[fi][j][idx[j]];
                *work += 1;
            }
            suf[k] = 1.0;
            for j in (0..k).rev() {
                suf[j] = suf[j + 1] * m.v2f[fi][j][idx[j]];
                *work += 1;
            }
            for p in 0..k {
                let t = pre[p] * suf[p + 1];
                let t = val * t;
                *work += 2;
                out.f2v[fi][p][idx[p]] += t;
            }
        }
        for p in 0..k {
            normalize(&mut out.f2v[fi][p], "factor->variable message proposal")?;
        }
    }
    Ok(())
}

/// m <- (1 - damping) * m + damping * F(m), elementwise on every message.
fn apply_damped(m: &mut Messages, f_m: &Messages, damping: f64) {
    let keep = 1.0 - damping;
    let mix = |dst: &mut Vec<Vec<f64>>, src: &Vec<Vec<f64>>| {
        for (a, b) in dst.iter_mut().zip(src) {
            for (x, y) in a.iter_mut().zip(b) {
                *x = keep * *x + damping * *y;
            }
        }
    };
    for (dst, src) in m.v2f.iter_mut().zip(&f_m.v2f) {
        mix(dst, src);
    }
    for (dst, src) in m.f2v.iter_mut().zip(&f_m.f2v) {
        mix(dst, src);
    }
}

/// Beliefs of every ORIGINAL variable (in variable order) from the current
/// state's factor->variable messages. Observed variables are one-hot;
/// unobserved isolated variables get the empty product, i.e. uniform.
fn beliefs_for(
    dims: &[usize],
    incident: &[Vec<(usize, usize)>],
    evidence: &BTreeMap<usize, usize>,
    m: &Messages,
    work: &mut u64,
) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::with_capacity(dims.len());
    for (v, d) in dims.iter().enumerate() {
        if let Some(&s) = evidence.get(&v) {
            let mut b = vec![0.0f64; *d];
            b[s] = 1.0;
            out.push(b);
            continue;
        }
        let mut b = vec![1.0f64; *d];
        for &(g, q) in &incident[v] {
            let src = &m.f2v[g][q];
            for (x, a) in b.iter_mut().enumerate() {
                *a *= src[x];
                *work += 1;
            }
        }
        normalize(&mut b, "variable belief")?;
        out.push(b);
    }
    Ok(out)
}

/// Undamped fixed-point residual max|F(m) - m| over every message component.
fn residual(m: &Messages, f_m: &Messages) -> f64 {
    let mut r = 0.0f64;
    for (side_m, side_f) in [(&m.v2f, &f_m.v2f), (&m.f2v, &f_m.f2v)] {
        for (a, b) in side_m.iter().zip(side_f) {
            for (x, y) in a.iter().zip(b) {
                for (u, v) in x.iter().zip(y) {
                    r = r.max((u - v).abs());
                }
            }
        }
    }
    r
}

/// One trace entry: sweep index, undamped residual at THIS state, beliefs from
/// THIS state's factor->variable messages, and cumulative update work.
fn record_state(
    sweep: usize,
    dims: &[usize],
    incident: &[Vec<(usize, usize)>],
    evidence: &BTreeMap<usize, usize>,
    m: &Messages,
    f_m: &Messages, // must be F(m), evaluated from the same state
    work: &mut u64,
) -> Result<(Value, f64)> {
    let b = beliefs_for(dims, incident, evidence, m, work)?;
    let res = residual(m, f_m);
    Ok((
        json!({
            "sweep": sweep,
            "residual": res,
            "marginals": b,
            "cumulative_update_work": *work,
        }),
        res,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_model() -> Model {
        // Nonuniform pairwise tree: 0 - 1 - 2, all binary.
        Model {
            kind: "MARKOV".to_string(),
            dims: vec![2, 2, 2],
            factors: vec![
                Factor {
                    scope: vec![0, 1],
                    values: vec![2.0, 1.0, 3.0, 1.0],
                },
                Factor {
                    scope: vec![1, 2],
                    values: vec![1.0, 4.0, 2.0, 1.0],
                },
            ],
        }
    }

    /// Independent brute-force posterior marginals by full enumeration.
    fn exact_marginals(model: &Model, evidence: &BTreeMap<usize, usize>) -> Vec<Vec<f64>> {
        let n = model.dims.len();
        let total: usize = model.dims.iter().product();
        let mut acc = vec![vec![0.0f64; 0]; n];
        for v in 0..n {
            if let Some(&s) = evidence.get(&v) {
                // Observed nodes get one-hot expectations, matching BP output.
                let mut onehot = vec![0.0; model.dims[v]];
                onehot[s] = 1.0;
                acc[v] = onehot;
            } else {
                acc[v] = vec![0.0; model.dims[v]];
            }
        }
        for i in 0..total {
            let x = decode_index(i, &(0..n).collect::<Vec<_>>(), &model.dims);
            if evidence.iter().any(|(&v, &s)| x[v] != s) {
                continue;
            }
            let w: f64 = model
                .factors
                .iter()
                .map(|f| {
                    let j = f
                        .scope
                        .iter()
                        .fold(0usize, |a, &v| a * model.dims[v] + x[v]);
                    f.values[j]
                })
                .product();
            for v in 0..n {
                if !evidence.contains_key(&v) {
                    if let Some(col) = acc[v].get_mut(x[v]) {
                        *col += w;
                    }
                }
            }
        }
        for v in 0..n {
            if !evidence.contains_key(&v) {
                let z: f64 = acc[v].iter().sum();
                for x in acc[v].iter_mut() {
                    *x /= z;
                }
            }
        }
        acc
    }

    fn assert_close(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.len(), y.len(), "variable {i} arity mismatch");
            for (u, v) in x.iter().zip(y) {
                assert!(
                    (u - v).abs() <= tol,
                    "variable {i}: {x:?} vs {y:?} (|diff| = {})",
                    (u - v).abs()
                );
            }
        }
    }

    #[test]
    fn tree_matches_exact_enumeration() {
        let model = tree_model();
        let evidence = BTreeMap::new();
        let expected = exact_marginals(&model, &evidence);
        // Undamped run (damping 1.0 = full proposal fraction) must converge on the tree...
        let out = run(&model, &evidence, 1.0, 1e-12, 100).expect("bp run");
        assert_eq!(out["converged"], json!(true), "tree BP must converge");
        let got: Vec<Vec<f64>> = serde_json::from_value(out["marginals"].clone()).unwrap();
        assert_close(&got, &expected, 1e-9);
        // ...and so must a damped run, to the same marginals.
        let out_d = run(&model, &evidence, 0.5, 1e-12, 500).expect("damped bp run");
        assert_eq!(out_d["converged"], json!(true));
        let got_d: Vec<Vec<f64>> = serde_json::from_value(out_d["marginals"].clone()).unwrap();
        assert_close(&got_d, &expected, 1e-9);
    }

    #[test]
    fn observed_variables_are_one_hot_and_condition_marginals_match() {
        let model = tree_model();
        let evidence = BTreeMap::from([(1usize, 0usize)]);
        let out = run(&model, &evidence, 1.0, 1e-12, 500).expect("bp run");
        assert_eq!(out["converged"], json!(true));
        let got: Vec<Vec<f64>> = serde_json::from_value(out["marginals"].clone()).unwrap();
        // Observed variable: exact one-hot belief.
        assert_eq!(got[1], vec![1.0, 0.0]);
        // Unobserved variables match conditioned enumeration.
        let expected = exact_marginals(&model, &evidence);
        assert_close(&got, &expected, 1e-9);
        // Trace states agree with the final state; settings echoed.
        assert_eq!(out["observations"], json!({"1": 0}));
        assert_eq!(
            out["settings"],
            json!({"damping": 1.0, "tolerance": 1e-12, "max_sweeps": 500})
        );
    }

    #[test]
    fn isolated_unobserved_variable_is_uniform() {
        let model = Model {
            kind: "MARKOV".to_string(),
            dims: vec![2, 2, 2],
            factors: vec![Factor {
                scope: vec![0, 1],
                values: vec![3.0, 1.0, 0.5, 2.0],
            }],
        };
        let out = run(&model, &BTreeMap::new(), 1.0, 1e-12, 100).expect("bp run");
        let got: Vec<Vec<f64>> = serde_json::from_value(out["marginals"].clone()).unwrap();
        assert_eq!(got[2], vec![0.5, 0.5]);
    }

    #[test]
    fn unary_initial_residual_is_exact_and_final_matches_analytic_proposal() {
        // One binary variable with a nonuniform unary factor. With uniform
        // messages the initial undamped residual is analytically
        // max|normalize(f) - uniform| = |(0.8, 0.2) - (0.5, 0.5)| = 0.3.
        let model = Model {
            kind: "MARKOV".to_string(),
            dims: vec![2],
            factors: vec![Factor {
                scope: vec![0],
                values: vec![0.8, 0.2],
            }],
        };
        let out = run(&model, &BTreeMap::new(), 0.5, 0.2, 1).expect("bp run");
        let trace = out["trace"].as_array().expect("trace array");
        // At eta=0.5 the initial residual is 0.3, distinct from the
        // proposed damped step size 0.15. It must not stop at sweep zero.
        assert_eq!(trace[0]["sweep"], json!(0));
        assert!((trace[0]["residual"].as_f64().unwrap() - 0.3).abs() <= 1e-12);
        // The unary proposal is constant: F(m) = normalize(f) = (0.8, 0.2).
        // One damped sweep returns beliefs (0.65,0.35). The residual at
        // that returned state is 0.15, while its next damped step is 0.075.
        assert_eq!(out["converged"], json!(true));
        assert_eq!(out["sweeps"], json!(1));
        assert!((out["residual"].as_f64().unwrap() - 0.15).abs() < 1e-12);
        let got: Vec<Vec<f64>> = serde_json::from_value(out["marginals"].clone()).unwrap();
        assert!((got[0][0] - 0.65).abs() < 1e-12);
        assert!((got[0][1] - 0.35).abs() < 1e-12);
    }

    #[test]
    fn conflicting_supports_error_with_zero_sum() {
        // Factor f supports only (x=0, y=1); g pins x=1 and h pins y=0, so the
        // posterior is empty and a message proposal must reach zero sum.
        let model = Model {
            kind: "MARKOV".to_string(),
            dims: vec![2, 2],
            factors: vec![
                Factor {
                    scope: vec![0, 1],
                    values: vec![0.0, 1.0, 0.0, 0.0],
                },
                Factor {
                    scope: vec![0],
                    values: vec![0.0, 1.0],
                },
                Factor {
                    scope: vec![1],
                    values: vec![1.0, 0.0],
                },
            ],
        };
        let err = run(&model, &BTreeMap::new(), 1.0, 1e-12, 10).expect_err("zero support");
        assert!(
            err.to_string().to_lowercase().contains("zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn zero_mass_factor_errors() {
        let model = Model {
            kind: "MARKOV".to_string(),
            dims: vec![2],
            factors: vec![Factor {
                scope: vec![0],
                values: vec![0.0, 0.0],
            }],
        };
        assert!(run(&model, &BTreeMap::new(), 1.0, 1e-12, 10).is_err());
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let model = tree_model();
        let none = BTreeMap::new();
        for damping in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert!(
                run(&model, &none, damping, 1e-10, 10).is_err(),
                "damping {damping}"
            );
        }
        for tol in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(run(&model, &none, 0.5, tol, 10).is_err(), "tolerance {tol}");
        }
        assert!(run(&model, &none, 0.5, 1e-10, 0).is_err(), "max_sweeps 0");
        assert!(run(&model, &BTreeMap::from([(0usize, 5usize)]), 0.5, 1e-10, 10).is_err());
    }

    #[test]
    fn trace_reports_every_recorded_state() {
        let model = tree_model();
        let out = run(&model, &BTreeMap::new(), 0.5, 1e-6, 1000).expect("bp run");
        let trace = out["trace"].as_array().expect("trace array");
        assert!(trace.len() >= 2);
        assert_eq!(trace[0]["sweep"], json!(0), "initial state recorded");
        let last = trace.last().unwrap();
        assert_eq!(last["sweep"], json!(out["sweeps"]));
        assert_eq!(last["residual"], json!(out["residual"]));
        assert_eq!(last["marginals"], json!(out["marginals"]));
        // Work is nondecreasing across the trace and included in every entry.
        let mut prev = 0u64;
        for e in trace {
            let w = e["cumulative_update_work"].as_u64().expect("work");
            assert!(w > prev, "work must strictly increase");
            prev = w;
        }
        assert_eq!(out["update_work"], json!(prev));
        assert_eq!(out["method"], json!("bp"));
        assert!(out["runtime_seconds"].as_f64().unwrap() >= 0.0);
    }
}
