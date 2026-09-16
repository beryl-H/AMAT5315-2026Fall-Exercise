//! run.json / series.jsonl / spins.jsonl artifact writing.

use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::ramp::{SpinFrame, TemperatureResult};

/// run.json metadata; field order matches the design contract.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunConfig {
    #[serde(rename = "L")]
    pub l: usize,
    pub update: String,
    pub t_grid: Vec<f64>,
    pub discard: usize,
    pub measure: usize,
    pub seed: u64,
    pub sample_every: usize,
    pub time_unit: String,
}

/// Round `x` to 6 decimal places, mapping -0.0 (and anything that rounds to
/// zero) to +0.0 so the artifact never shows "-0.000000".
fn six(x: f64) -> f64 {
    let r = (x * 1e6).round() / 1e6;
    if r == 0.0 { 0.0 } else { r }
}

/// Write run.json, series.jsonl and (when any frames exist) spins.jsonl
/// into `dir`, creating the directory as needed.
///
/// series.jsonl rows: {"L","T","sweep","M","E"} with M and E formatted to
/// exactly 6 decimal places and T a plain numeric JSON value.
/// spins.jsonl rows: {"L","T","sweep","m","spins"} with `spins` a string of
/// l*l characters, '1' for +1 and '0' for -1, row after row from the top.
pub fn write_artifacts(
    dir: &Path,
    run: &RunConfig,
    results: &[TemperatureResult],
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join("run.json"), serde_json::to_string_pretty(run).unwrap())?;

    let mut series = fs::File::create(dir.join("series.jsonl"))?;
    for r in results {
        for row in &r.series {
            match row.cluster_size {
                // Wolff rows add cluster_size, the spins flipped in that move.
                Some(cs) => writeln!(
                    series,
                    r#"{{"L":{},"T":{},"sweep":{},"M":{:.6},"E":{:.6},"cluster_size":{}}}"#,
                    run.l,
                    serde_json::to_string(&row.t).unwrap(),
                    row.sweep,
                    six(row.m),
                    six(row.e),
                    cs,
                )?,
                None => writeln!(
                    series,
                    r#"{{"L":{},"T":{},"sweep":{},"M":{:.6},"E":{:.6}}}"#,
                    run.l,
                    serde_json::to_string(&row.t).unwrap(),
                    row.sweep,
                    six(row.m),
                    six(row.e),
                )?,
            }
        }
    }

    let frames: Vec<&SpinFrame> = results.iter().flat_map(|r| r.frames.iter()).collect();
    if !frames.is_empty() {
        let mut spins = fs::File::create(dir.join("spins.jsonl"))?;
        for f in frames {
            let mut text = String::with_capacity(f.spins.len());
            for &s in &f.spins {
                text.push(if s == 1 { '1' } else { '0' });
            }
            writeln!(
                spins,
                r#"{{"L":{},"T":{},"sweep":{},"m":{},"spins":{}}}"#,
                run.l,
                serde_json::to_string(&f.t).unwrap(),
                f.global_sweep,
                serde_json::to_string(&f.m).unwrap(),
                serde_json::to_string(&text).unwrap(),
            )?;
        }
    }
    Ok(())
}
