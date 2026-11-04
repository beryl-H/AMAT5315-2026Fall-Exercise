use crate::{
    model::{Factor, Model, decode},
    plan::Plan,
};
use anyhow::{Result, ensure};
use omeco::NestedEinsum;
use serde::Serialize;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::Tensor;
#[derive(Serialize)]
pub struct ContractResult {
    pub log10_sum: f64,
    pub log10_scale: f64,
    pub normalized: Vec<f64>,
}
fn normalize(t: Tensor, scale: f64) -> Result<(Tensor, f64)> {
    let data = t.as_slice::<f64>()?;
    let max = data.iter().copied().fold(0f64, f64::max);
    ensure!(
        max > 0.0 && max.is_finite(),
        "zero or invalid contraction weight"
    );
    let values = data.iter().map(|x| x / max).collect::<Vec<_>>();
    Ok((
        Tensor::from_vec_col_major(t.shape().to_vec(), values)?,
        scale + max.log10(),
    ))
}
pub fn contract(m: &Model, fs: &[Factor], p: &Plan) -> Result<ContractResult> {
    ensure!(
        m.dims == p.dimensions
            && fs.iter().map(|f| f.scope.clone()).collect::<Vec<_>>() == p.input_labels,
        "saved plan does not match model/evidence scopes"
    );
    crate::plan::analyze(p)?;
    fn eval(
        t: &NestedEinsum<usize>,
        m: &Model,
        fs: &[Factor],
        cpu: &mut CpuEngine,
    ) -> Result<(Tensor, f64)> {
        match t {
            NestedEinsum::Leaf { tensor_index } => {
                let f = &fs[*tensor_index];
                normalize(
                    Tensor::from_vec_row_major(
                        f.scope.iter().map(|&v| m.dims[v]).collect::<Vec<_>>(),
                        f.values.clone(),
                    )?,
                    0.0,
                )
            }
            NestedEinsum::Node { args, eins } => {
                let (a, sa) = eval(&args[0], m, fs, cpu)?;
                let (b, sb) = eval(&args[1], m, fs, cpu)?;
                let i = eins
                    .ixs
                    .iter()
                    .map(|s| s.iter().map(|&v| v as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                let o = eins.iy.iter().map(|&v| v as u32).collect::<Vec<_>>();
                let sub =
                    EinsumSubscripts::new(&i.iter().map(Vec::as_slice).collect::<Vec<_>>(), &o);
                normalize([&a, &b].einsum_subscripts(&sub, cpu)?, sa + sb)
            }
        }
    }
    let mut cpu = CpuEngine::serial();
    let (t, scale) = eval(&p.tree, m, fs, &mut cpu)?;
    let data = t.as_slice::<f64>()?;
    let sum = data.iter().sum::<f64>();
    let mut normalized = Vec::new();
    for i in 0..data.len() {
        let x = decode(i, &p.output_labels, &m.dims);
        let mut stride = 1;
        let mut j = 0;
        for (&v, &s) in p.output_labels.iter().zip(&x) {
            j += stride * s;
            stride *= m.dims[v];
        }
        normalized.push(data[j] / sum);
    }
    Ok(ContractResult {
        log10_sum: scale + sum.log10(),
        log10_scale: scale,
        normalized,
    })
}
pub fn sanity_checks() -> Result<serde_json::Value> {
    let mut cpu = CpuEngine::serial();
    let a = Tensor::from_vec_row_major(vec![2, 2], vec![1f64, 2., 3., 4.])?;
    let b = Tensor::from_vec_row_major(vec![2, 2], vec![2f64, 0., 1., 2.])?;
    let ab = [&a, &b].einsum("ij,jk->ik", &mut cpu)?;
    let trace = [&a, &b].einsum("ij,ji->", &mut cpu)?.as_slice::<f64>()?[0];
    let matrix = vec![
        ab.as_slice::<f64>()?[0],
        ab.as_slice::<f64>()?[2],
        ab.as_slice::<f64>()?[1],
        ab.as_slice::<f64>()?[3],
    ];
    let k = Tensor::from_vec_row_major(
        vec![2, 2],
        vec![0.5f64.exp(), (-0.5f64).exp(), (-0.5f64).exp(), 0.5f64.exp()],
    )?;
    let z = [&k, &k, &k]
        .einsum("ij,jk,ki->", &mut cpu)?
        .as_slice::<f64>()?[0];
    let mut ze = 0.;
    for a in [-1f64, 1.] {
        for b in [-1f64, 1.] {
            for c in [-1f64, 1.] {
                ze += (0.5 * (a * b + b * c + c * a)).exp();
            }
        }
    }
    let err = (z - ze).abs();
    ensure!(
        matrix == vec![4., 4., 10., 8.] && (trace - 12.).abs() < 1e-12 && err < 1e-12,
        "sanity checks failed"
    );
    Ok(
        serde_json::json!({"matrix_ab":matrix,"matrix_ab_absolute_error":0.0,"trace_ab":trace,"trace_absolute_error":(trace-12.).abs(),"ising_contraction":z,"ising_enumeration":ze,"ising_absolute_error":err}),
    )
}
