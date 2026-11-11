//! Tensor renormalization group (TRG) for the periodic square-lattice Ising model.
//!
//! Source algorithm port:
//! <https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl>
//!
//! The Julia example splits the bond tensor twice (two independent SVD splits) and
//! contracts four half-tensors into a new lattice with half as many sites; two
//! rounds halve the linear size. Two deliberate deviations from that example are mandated by the lecture
//! notes and are implemented here:
//!
//! 1. **Closure**: the Julia example sums the four legs that are *not* opposite
//!    (`all four equal`), which is incorrect for the square lattice. Here the
//!    closure uses the two *independent opposite-leg* sums
//!    `z = sum_i sum_j A[i, j, i, j]` (indices `r == l` and `u == d`).
//! 2. **Normalization**: before every round the tensor is normalized by its
//!    maximum absolute entry `c`, and the corresponding factor `c^(N)` is
//!    accumulated in the running log scale `ell` (`ell += N * ln c`).
//!
//! Tensor conventions
//! -----------------
//! The lattice tensor `A` has shape `[r, u, l, d]` in C order (last axis fastest),
//! where `r`/`l` are the right/left legs and `u`/`d` the up/down legs of the same
//! lattice site. `D` is the bond dimension; the tensor has `D^4` entries.
//!
//! Each round performs two SVD splits of the flattened matrices
//!
//! * `Ma[(l, u), (r, d)] = A[r, u, l, d]`   (rows `l*D + u`, columns `r*D + d`)
//! * `Mb[(d, l), (u, r)] = A[r, u, l, d]`   (rows `d*D + l`, columns `u*D + r`)
//!
//! and reconstructs
//!
//! ```text
//! Anew[r, u, l, d] = sum_{w,a,b,g} S1[w, a, r] * S2[a, b, u] * S3[l, b, g] * S4[d, g, w]
//! ```
//!
//! with `Dnew = q = min(chi, D*D)`. The four-index contraction is evaluated with
//! grouped matrix products (BLAS-style `mm` from `crate::linalg`), never with
//! eight nested loops:
//!
//! ```text
//! X[(r, w), (b, u)] = sum_a S1[w, a, r] * S2[a, b, u]
//! Y[(l, b), (w, d)] = sum_g S3[l, b, g] * S4[d, g, w]
//! Anew[(r, u), (l, d)] = sum_{w, b} X[(r, w), (b, u)] * Y[(l, b), (w, d)]
//! ```
//!
//! Evidence files
//! --------------
//! When `out` is `Some(dir)` the routine writes one JSON file per round plus a
//! summary `trg-evidence.json`. Each step file contains the split factors
//! `S1..S4` (with their shapes), the singular value spectra, the discarded
//! relative squared weight, the normalization constant `c`, the site count `N`
//! before the round and the running log scale `ell`, which together are
//! sufficient to replay the contraction independently.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::linalg::{mm, svd};
use crate::model::site;

/// One renormalization round's worth of data (kept small: no full tensors).
#[derive(Debug, Clone)]
struct StepData {
    step: usize,
    c: f64,
    n_before: usize,
    ell: f64,
    old_dimension: usize,
    q: usize,
    spectrum_ma: Vec<f64>,
    spectrum_mb: Vec<f64>,
    discarded_rel_sq_ma: f64,
    discarded_rel_sq_mb: f64,
    s1: Vec<f64>,
    s2: Vec<f64>,
    s3: Vec<f64>,
    s4: Vec<f64>,
}

/// A dense tensor with its shape, stored row-major (last axis fastest).
#[derive(Debug, Clone)]
struct Tensor {
    dims: Vec<usize>,
    data: Vec<f64>,
}

impl Tensor {
    fn new(dims: Vec<usize>, data: Vec<f64>) -> Self {
        let n: usize = dims.iter().product();
        assert_eq!(data.len(), n, "tensor data length mismatch");
        Tensor { dims, data }
    }
}

/// Rank-4 tensor view used inside the round: shape `[r, u, l, d]`.
struct Tensor4 {
    d: usize,
    data: Vec<f64>,
}

impl Tensor4 {
    #[inline]
    fn get(&self, r: usize, u: usize, l: usize, d: usize) -> f64 {
        self.data[((r * self.d + u) * self.d + l) * self.d + d]
    }
}

/// Validate the public inputs.
fn check_inputs(l: usize, k: f64, chi: usize) -> Result<()> {
    if l < 2 || !l.is_power_of_two() {
        bail!("l must be a power of two >= 2, got {l}");
    }
    if !k.is_finite() {
        bail!("k must be finite, got {k}");
    }
    if k < 0.0 {
        bail!("k must be >= 0, got {k}");
    }
    if chi < 1 {
        bail!("chi must be >= 1, got {chi}");
    }
    Ok(())
}

/// Write a JSON value to `path` with a buffered writer.
fn write_json(path: &Path, value: &Value) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(file);
    serde_json::to_writer(&mut w, value)
        .with_context(|| format!("serializing {}", path.display()))?;
    w.flush()
        .with_context(|| format!("flushing {}", path.display()))?;
    Ok(())
}

/// JSON array of f64 with full precision.
fn jvec(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|&x| json!(x)).collect())
}

/// Relative squared weight discarded by truncating the singular values.
///
/// `discarded = sum_{i >= q} s_i^2 / sum_i s_i^2` (0 when the spectrum is empty).
fn discarded_rel_sq(s: &[f64], q: usize) -> f64 {
    let total: f64 = s.iter().map(|x| x * x).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let kept: f64 = s.iter().take(q).map(|x| x * x).sum();
    let disc = total - kept;
    if disc <= 0.0 { 0.0 } else { disc / total }
}

/// Perform the two SVD splits of one round and build `S1..S4`.
///
/// Returns the split factors together with the two spectra.
fn split(
    a: &Tensor4,
    d: usize,
    chi: usize,
) -> Result<(
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
)> {
    let d2 = d * d;
    // Ma[(l, u), (r, d)] = A[r, u, l, d]  (rows l*D+u, cols r*D+d)
    let mut ma = vec![0.0f64; d2 * d2];
    for r in 0..d {
        for u in 0..d {
            for l in 0..d {
                for dd in 0..d {
                    ma[(l * d + u) * d2 + (r * d + dd)] = a.get(r, u, l, dd);
                }
            }
        }
    }
    // Mb[(d, l), (u, r)] = A[r, u, l, d]  (rows d*D+l, cols u*D+r)
    let mut mb = vec![0.0f64; d2 * d2];
    for r in 0..d {
        for u in 0..d {
            for l in 0..d {
                for dd in 0..d {
                    mb[(dd * d + l) * d2 + (u * d + r)] = a.get(r, u, l, dd);
                }
            }
        }
    }

    let sda = svd(&ma, d2, d2);
    let sdb = svd(&mb, d2, d2);

    let q = chi.min(d2);

    // Thin SVD: U is m x q (row-major), VT is q x n (row-major), s descending.
    // Trim only exact numerical zeros of the *retained* spectrum is not done:
    // the cap is honoured exactly (zero singular values are kept), so that the
    // new bond dimension is always exactly q = min(chi, D*D).
    let sqrt_s: Vec<f64> = sda.s.iter().take(q).map(|x| x.sqrt()).collect();
    let sqrt_t: Vec<f64> = sdb.s.iter().take(q).map(|x| x.sqrt()).collect();

    // Old legs have dimension d, retained legs dimension q.
    let mut s1 = vec![0.; d * d * q];
    let mut s2 = vec![0.; d * d * q];
    let mut s3 = vec![0.; q * d * d];
    let mut s4 = vec![0.; q * d * d];
    for i in 0..d * d {
        for t in 0..q {
            s1[i * q + t] = sda.u[i * d2 + t] * sqrt_s[t];
            s2[i * q + t] = sdb.u[i * d2 + t] * sqrt_t[t];
            s3[t * d2 + i] = sqrt_s[t] * sda.vt[t * d2 + i];
            s4[t * d2 + i] = sqrt_t[t] * sdb.vt[t * d2 + i];
        }
    }

    let spec_a: Vec<f64> = sda.s.clone();
    let spec_b: Vec<f64> = sdb.s.clone();
    Ok((s1, s2, s3, s4, spec_a, spec_b, q))
}

/// Contract the four split factors back into the renormalized tensor.
///
/// `Anew[r, u, l, d] = sum_{w,a,b,g} S1[w,a,r] S2[a,b,u] S3[l,b,g] S4[d,g,w]`
///
/// Implemented with grouped matrix products:
///
/// * `X[(r, w), (b, u)] = sum_a S1[w, a, r] S2[a, b, u]`
/// * `Y[(l, b), (w, d)] = sum_g S3[l, b, g] S4[d, g, w]`
/// * `Anew[(r, u), (l, d)] = sum_{w, b} X[(r, w), (b, u)] Y[(l, b), (w, d)]`
fn contract(s1: &[f64], s2: &[f64], s3: &[f64], s4: &[f64], d: usize, q: usize) -> Vec<f64> {
    // X[(r,w),(b,u)] = sum_a S1[w,a,r] S2[a,b,u].
    let mut m1 = vec![0.; q * d * d];
    for r in 0..q {
        for w in 0..d {
            for a in 0..d {
                m1[(r * d + w) * d + a] = s1[(w * d + a) * q + r];
            }
        }
    }
    let x = mm(&m1, s2, q * d, d, d * q);
    let mut p = vec![0.; q * q * d * d];
    for r in 0..q {
        for u in 0..q {
            for w in 0..d {
                for b in 0..d {
                    p[(r * q + u) * d * d + w * d + b] = x[(r * d + w) * d * q + b * q + u];
                }
            }
        }
    }
    // Y[(l,b),(w,dnew)] = sum_g S3[l,b,g] S4[dnew,g,w].
    let mut m4 = vec![0.; d * d * q];
    for g in 0..d {
        for w in 0..d {
            for dn in 0..q {
                m4[g * d * q + w * q + dn] = s4[(dn * d + g) * d + w];
            }
        }
    }
    let y = mm(s3, &m4, q * d, d, d * q);
    let mut v = vec![0.; d * d * q * q];
    for w in 0..d {
        for b in 0..d {
            for l in 0..q {
                for dn in 0..q {
                    v[(w * d + b) * q * q + l * q + dn] = y[(l * d + b) * d * q + w * q + dn];
                }
            }
        }
    }
    mm(&p, &v, q * q, d * d, q * q)
}

/// Run the TRG for a periodic square lattice of linear size `l`.
///
/// Returns `g = log(Z) / l^2`; free energy per site is -g/k for k>0.
pub fn run(l: usize, k: f64, chi: usize, out: Option<&Path>) -> Result<f64> {
    check_inputs(l, k, chi)?;

    let n_rounds = 2 * l.trailing_zeros() as usize;

    let mut d: usize = 2;
    let mut a = Tensor4 { d, data: site(k) };
    if a.data.len() != d * d * d * d {
        bail!("site tensor has unexpected length {}", a.data.len());
    }
    if a.data.iter().any(|x| !x.is_finite()) {
        bail!("site tensor contains non-finite entries");
    }

    let mut ell: f64 = 0.0;
    let mut n_sites: usize = l * l;
    let mut steps: Vec<StepData> = Vec::with_capacity(n_rounds);
    let mut final_tensor: Option<Tensor> = None;

    for step in 0..n_rounds {
        // normalize: c = max |A|; A /= c; ell += N ln c
        let c = a.data.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        if !(c > 0.0) || !c.is_finite() {
            bail!("step {step}: tensor has non-positive or non-finite max abs entry {c}");
        }
        for x in a.data.iter_mut() {
            *x /= c;
        }
        ell += (n_sites as f64) * c.ln();

        let (s1, s2, s3, s4, spec_a, spec_b, q) = split(&a, d, chi)?;

        let disc_a = discarded_rel_sq(&spec_a, q);
        let disc_b = discarded_rel_sq(&spec_b, q);

        let anew = contract(&s1, &s2, &s3, &s4, d, q);
        if anew.iter().any(|x| !x.is_finite()) {
            bail!("step {step}: renormalized tensor contains non-finite entries");
        }

        let saved_step = StepData {
            step,
            c,
            n_before: n_sites,
            ell,
            old_dimension: d,
            q,
            spectrum_ma: spec_a,
            spectrum_mb: spec_b,
            discarded_rel_sq_ma: disc_a,
            discarded_rel_sq_mb: disc_b,
            s1,
            s2,
            s3,
            s4,
        };
        if let Some(dir) = out {
            write_step(dir, l, k, chi, &saved_step)?;
        }
        let mut diagnostic = saved_step;
        diagnostic.s1.clear();
        diagnostic.s2.clear();
        diagnostic.s3.clear();
        diagnostic.s4.clear();
        steps.push(diagnostic);

        d = q;
        a = Tensor4 { d, data: anew };
        n_sites /= 2;
        final_tensor = Some(Tensor::new(vec![d, d, d, d], a.data.clone()));
    }

    assert_eq!(n_sites, 1, "two rounds per halving of the original side");
    // Closure: two independent opposite-leg sums, z = sum_i sum_j A[i, j, i, j].
    let mut z = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            z += a.get(i, j, i, j);
        }
    }
    if !z.is_finite() {
        bail!("final closure is non-finite");
    }
    if z <= 0.0 {
        bail!("final closure is non-positive: {z}");
    }

    let log_z = ell + z.ln();
    let g = log_z / (l as f64 * l as f64);

    if let Some(dir) = out {
        write_evidence(
            dir,
            l,
            k,
            chi,
            &steps,
            final_tensor.as_ref().unwrap(),
            z,
            ell,
            log_z,
            g,
        )?;
    }

    Ok(g)
}

/// Write the per-step files and the summary evidence file.
#[allow(clippy::too_many_arguments)]
fn write_evidence(
    dir: &Path,
    l: usize,
    k: f64,
    chi: usize,
    steps: &[StepData],
    final_tensor: &Tensor,
    closure: f64,
    ell: f64,
    log_z: f64,
    g: f64,
) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    let summary = json!({
        "algorithm": "TRG (two-split SVD, grouped GEMM contraction)",
        "source_port": "https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl",
        "l": l,
        "k": k,
        "chi": chi,
        "rounds": steps.len(),
        "final_dimension": final_tensor.dims[0],
        "log_scale": ell,
        "final_closure": closure,
        "log_z": log_z,
        "g": g,
        "steps": steps.iter().map(|st| json!({
            "step": st.step,
            "scale_c": st.c,
            "n_sites_before": st.n_before,
            "ell_after": st.ell,
            "dimension_before": st.old_dimension,
            "dimension_after": st.q,
            "discarded_rel_sq_ma": st.discarded_rel_sq_ma,
            "discarded_rel_sq_mb": st.discarded_rel_sq_mb,
        })).collect::<Vec<Value>>(),
        "final_tensor": {
            "shape": final_tensor.dims,
            "data": jvec(&final_tensor.data),
        },
    });
    write_json(&dir.join("trg-evidence.json"), &summary)?;
    Ok(())
}

fn write_step(dir: &Path, l: usize, k: f64, chi: usize, st: &StepData) -> Result<()> {
    let d = st.old_dimension;
    let q = st.q;
    let value = json!({
        "step": st.step,
        "k": k,
        "l": l,
        "chi": chi,
        "dimension_before": d,
        "dimension_after": q,
        "q": q,
        "n_sites_before": st.n_before,
        "scale_c": st.c,
        "ell_after": st.ell,
        "spectrum_ma": jvec(&st.spectrum_ma),
        "spectrum_mb": jvec(&st.spectrum_mb),
        "discarded_rel_sq_ma": st.discarded_rel_sq_ma,
        "discarded_rel_sq_mb": st.discarded_rel_sq_mb,
        "s1": {
            "shape": [d, d, q],
            "data": jvec(&st.s1),
        },
        "s2": {
            "shape": [d, d, q],
            "data": jvec(&st.s2),
        },
        "s3": {
            "shape": [q, d, d],
            "data": jvec(&st.s3),
        },
        "s4": {
            "shape": [q, d, d],
            "data": jvec(&st.s4),
        },
        "layouts": {
            "s1": "[old_w,old_a,new_r], C order",
            "s2": "index (d, l, q) row-major over [d, d, q]",
            "s3": "[new_l,old_b,old_g], C order",
            "s4": "[new_d,old_g,old_w], C order",
        },
    });
    let path = dir.join(format!("step-{:04}.json", st.step));
    write_json(&path, &value)?;
    Ok(())
}
