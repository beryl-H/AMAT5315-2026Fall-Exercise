use serde_json::{Value, json};
use std::{fs, path::PathBuf, process::Command};

#[test]
fn tropical_count_and_boltzmann_samples_match_small_open_lattice() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(format!("test-magnet-{}", std::process::id()));
    let command = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args([
            "--mode",
            "sample",
            "--l",
            "2",
            "--p-af",
            "0",
            "--beta",
            "0.3",
            "--order",
            "greedy",
            "--samples",
            "1200",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        command.status.success(),
        "{}",
        String::from_utf8_lossy(&command.stderr)
    );
    let run: Value = serde_json::from_slice(&fs::read(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["e_min"], -4);
    assert_eq!(run["g0"], 2);
    let z = 2.0 * 1.2_f64.exp() + 12.0 + 2.0 * (-1.2_f64).exp();
    assert!((run["log_z"].as_f64().unwrap() - z.ln()).abs() < 1e-12);
    let samples = run["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 1200);
    let mut observed = 0;
    for sample in samples {
        let s: Vec<_> = sample["bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|bit| 2 * bit.as_i64().unwrap() - 1)
            .collect();
        assert_eq!(s.len(), 4);
        assert!(s.iter().all(|x| x.abs() == 1));
        let e = -(s[0] * s[1] + s[0] * s[2] + s[1] * s[3] + s[2] * s[3]);
        assert_eq!(sample["energy"], json!(e));
        observed += usize::from(e == -4);
    }
    let p = 2.0 * 1.2_f64.exp() / z;
    let sigma = (1200.0 * p * (1.0 - p)).sqrt();
    assert!((observed as f64 - 1200.0 * p).abs() < 4.0 * sigma);
}

#[test]
fn counts_an_isolated_spin_in_both_states() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(format!("test-isolated-{}", std::process::id()));
    let command = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args([
            "--mode", "count", "--l", "1", "--p-af", "0", "--beta", "2", "--order", "greedy",
            "--seed", "2026", "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        command.status.success(),
        "{}",
        String::from_utf8_lossy(&command.stderr)
    );
    let run: Value = serde_json::from_slice(&fs::read(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["e_min"], 0);
    assert_eq!(run["g0"], 2);
    assert!((run["log_z"].as_f64().unwrap() - std::f64::consts::LN_2).abs() < 1e-12);
}
