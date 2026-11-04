use anyhow::{Context, Result};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::Tensor;
fn rss() -> String {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .filter(|s| s.starts_with("VmRSS:") || s.starts_with("VmPeak:"))
        .collect::<Vec<_>>()
        .join("; ")
}
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let file = args.get(1).context("complexity path")?;
    let id = args.get(2).context("step ID")?.parse::<u64>()?;
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(file)?)?;
    let s = v["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"].as_u64() == Some(id))
        .context("step not found")?;
    let inputs = s["input_labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            s.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let out = s["retained_labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    let mut cpu = CpuEngine::serial();
    if args.get(3).is_some_and(|a| a == "no-cache") {
        cpu.set_buffer_pool_limit_bytes(0);
    }
    eprintln!(
        "step{id} input_ranks={:?} output_rank={} pool_limit={} {}",
        inputs.iter().map(Vec::len).collect::<Vec<_>>(),
        out.len(),
        cpu.buffer_pool_limit_bytes(),
        rss()
    );
    let a = Tensor::from_vec_col_major(vec![2; inputs[0].len()], vec![1f64; 1 << inputs[0].len()])?;
    let b = Tensor::from_vec_col_major(vec![2; inputs[1].len()], vec![1f64; 1 << inputs[1].len()])?;
    eprintln!("inputs allocated {}", rss());
    let start = std::time::Instant::now();
    let sub = EinsumSubscripts::new(&inputs.iter().map(Vec::as_slice).collect::<Vec<_>>(), &out);
    let c = [&a, &b].einsum_subscripts(&sub, &mut cpu)?;
    eprintln!(
        "done elapsed={:?} result_len={} sample={} cache={:?} {}",
        start.elapsed(),
        c.as_slice::<f64>()?.len(),
        c.as_slice::<f64>()?[0],
        cpu.runtime_cache_stats(),
        rss()
    );
    Ok(())
}
