use anyhow::Result;
use infer::{
    model::{Model, parse_evidence},
    plan,
};
fn main() -> Result<()> {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let m = Model::parse(&std::fs::read_to_string(
        base.join("promedus/Promedus_14.uai"),
    )?)?;
    let e = parse_evidence(&std::fs::read_to_string(
        base.join("promedus/Promedus_14.uai.evid"),
    )?)?;
    let fs = m.condition(&e)?;
    let p = plan::create(&m, &fs, &[139], "greedy", None, 1, 20)?;
    let c = plan::analyze(&p)?;
    let dest = base
        .parent()
        .unwrap()
        .join("week9-support/promedus-q139-plan-only");
    std::fs::create_dir_all(&dest)?;
    let order = dest.join("order.json");
    anyhow::ensure!(!order.exists(), "do not overwrite probe");
    std::fs::write(order, serde_json::to_vec_pretty(&p)?)?;
    std::fs::write(dest.join("complexity.json"), serde_json::to_vec_pretty(&c)?)?;
    println!(
        "plan-only: work={} largest_tensor={} largest_generated_tensor={}; no contraction invoked",
        c.work, c.largest_tensor, c.largest_generated_tensor
    );
    Ok(())
}
