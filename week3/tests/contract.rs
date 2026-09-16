//! Integration tests for the file/stdout contract and physics sanity.

use std::path::PathBuf;
use std::process::{Command, Output};

fn ising_binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ising"))
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ising-contract-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn run_ising(args: &[String]) -> Output {
    ising_binary().args(args).output().expect("failed to run ising binary")
}

/// Default valid argument list; individual tests override values.
fn base_args(out: &PathBuf) -> Vec<String> {
    vec![
        "--update".into(), "metropolis".into(),
        "--l".into(), "4".into(),
        "--t-from".into(), "1.0".into(),
        "--t-to".into(), "1.0".into(),
        "--t-step".into(), "0.5".into(),
        "--discard".into(), "0".into(),
        "--measure".into(), "5".into(),
        "--seed".into(), "2026".into(),
        "--every".into(), "0".into(),
        "--out".into(), out.to_str().unwrap().into(),
    ]
}

fn set(args: &mut Vec<String>, flag: &str, value: &str) {
    let pos = args.iter().position(|a| a == flag).expect("flag present in base args");
    args[pos + 1] = value.to_string();
}

fn read_jsonl(path: &PathBuf) -> Vec<serde_json::Value> {
    let text = std::fs::read_to_string(path).expect("jsonl file missing");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("jsonl line malformed"))
        .collect()
}

fn sorted_keys(v: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    keys
}

/// The raw text of a JSON field (everything between `"key":` and the next
/// ',' or '}').
fn raw_field(line: &str, key: &str) -> String {
    let needle = format!("\"{key}\":");
    let pos = line.find(&needle).unwrap_or_else(|| panic!("missing field {key} in {line}"));
    let rest = &line[pos + needle.len()..];
    let end = rest.find(|c| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].to_string()
}

fn assert_six_decimals(raw: &str, key: &str) {
    let parsed: f64 = raw
        .parse()
        .unwrap_or_else(|_| panic!("field {key} is not a number: {raw:?}"));
    assert!(parsed.is_finite(), "field {key} not finite: {raw:?}");
    let frac = raw.split('.').nth(1).unwrap_or_else(|| panic!("field {key} has no decimal point: {raw:?}"));
    assert_eq!(frac.len(), 6, "field {key} must have exactly 6 decimals: {raw:?}");
    assert!(frac.chars().all(|c| c.is_ascii_digit()), "field {key} fractional digits: {raw:?}");
}

#[test]
fn single_temperature_run_writes_contract_files() {
    let out = temp_dir("single");
    let args = base_args(&out); // l=4, single temperature 1.0, measure=5, every=0
    let output = run_ising(&args);
    assert!(
        output.status.success(),
        "ising failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // stdout: header, then one tab-separated row per temperature.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "expected header + 1 temperature row");
    assert_eq!(lines[0], "T\tmean_abs_M\tacceptance");
    let fields: Vec<&str> = lines[1].split('\t').collect();
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0], "1.000000");

    // run.json: exactly the contract keys.
    let run: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).expect("run.json missing"),
    )
    .expect("run.json malformed");
    assert_eq!(
        sorted_keys(&run),
        ["L", "discard", "measure", "sample_every", "seed", "t_grid", "time_unit", "update"]
    );
    assert_eq!(run["L"], 4);
    assert_eq!(run["update"], "metropolis");
    assert_eq!(run["t_grid"], serde_json::json!([1.0]));
    assert_eq!(run["discard"], 0);
    assert_eq!(run["measure"], 5);
    assert_eq!(run["seed"], 2026);
    assert_eq!(run["sample_every"], 1);
    assert_eq!(run["time_unit"], "sweep");

    // series.jsonl: one row per measured step, sweeps 1..=measure.
    let series = read_jsonl(&out.join("series.jsonl"));
    assert_eq!(series.len(), 5, "one row per measured sweep");
    for (i, row) in series.iter().enumerate() {
        assert_eq!(sorted_keys(row), ["E", "L", "M", "T", "sweep"]);
        assert_eq!(row["L"], 4);
        assert_eq!(row["sweep"], (i + 1) as u64);
        assert!(row["T"].is_f64(), "T must be a numeric JSON value");
        assert!((row["T"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    }
    // M and E are written with exactly 6 decimal places.
    let series_text = std::fs::read_to_string(out.join("series.jsonl")).unwrap();
    for line in series_text.lines() {
        assert_six_decimals(&raw_field(line, "M"), "M");
        assert_six_decimals(&raw_field(line, "E"), "E");
    }

    // every=0: no spins.jsonl at all.
    assert!(!out.join("spins.jsonl").exists(), "no spins.jsonl when every=0");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn spins_frames_cumulative_and_series_sweeps_reset() {
    let out = temp_dir("spins");
    let mut args = base_args(&out);
    set(&mut args, "--t-to", "2.0"); // grid [1.0, 2.0]
    set(&mut args, "--t-step", "1.0");
    set(&mut args, "--measure", "6");
    set(&mut args, "--every", "2");
    let output = run_ising(&args);
    assert!(
        output.status.success(),
        "ising failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // stdout has one row per temperature (header + 2).
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.lines().count(), 3);

    // run.json records the full t_grid.
    let run: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(run["t_grid"], serde_json::json!([1.0, 2.0]));

    // series.jsonl: sweep resets at each temperature (1..=6 twice).
    let series = read_jsonl(&out.join("series.jsonl"));
    assert_eq!(series.len(), 12);
    let sweeps: Vec<u64> = series.iter().map(|r| r["sweep"].as_u64().unwrap()).collect();
    assert_eq!(sweeps, vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]);

    // spins.jsonl: frames at measured steps 2,4,6 with a *cumulative*
    // global sweep counter (2,4,6 then 8,10,12), spins a 16-char string.
    assert!(out.join("spins.jsonl").exists(), "spins.jsonl expected when every>0");
    let frames = read_jsonl(&out.join("spins.jsonl"));
    assert_eq!(frames.len(), 6);
    let global: Vec<u64> = frames.iter().map(|f| f["sweep"].as_u64().unwrap()).collect();
    assert_eq!(global, vec![2, 4, 6, 8, 10, 12]);
    for f in &frames {
        assert_eq!(sorted_keys(f), ["L", "T", "m", "spins", "sweep"]);
        assert_eq!(f["L"], 4);
        assert!(f["T"].is_f64());
        assert!(f["m"].is_f64());
        let spins = f["spins"].as_str().expect("spins must be a string");
        assert_eq!(spins.len(), 16, "spins string has exactly l*l characters");
        assert!(spins.chars().all(|c| c == '0' || c == '1'), "spins chars are 0/1");
        // The string encoding is consistent with the reported m.
        let ones = spins.chars().filter(|&c| c == '1').count();
        let computed_m = (ones as f64 - (16 - ones) as f64) / 16.0;
        assert!(
            (computed_m - f["m"].as_f64().unwrap()).abs() < 1e-12,
            "spins string vs m mismatch: {spins}, m={}",
            f["m"]
        );
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn low_temperature_from_all_up_never_flips() {
    let out = temp_dir("lowt");
    let mut args = base_args(&out);
    set(&mut args, "--l", "8");
    set(&mut args, "--t-from", "0.01");
    set(&mut args, "--t-to", "0.01");
    set(&mut args, "--t-step", "0.01");
    set(&mut args, "--measure", "20");
    let output = run_ising(&args);
    assert!(
        output.status.success(),
        "ising failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Every flip from all-up costs +8, rejected at T=0.01: lattice stays up.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let row = stdout.lines().nth(1).unwrap();
    let fields: Vec<&str> = row.split('\t').collect();
    assert_eq!(fields, ["0.010000", "1.000000", "0.000000"]);

    let series = read_jsonl(&out.join("series.jsonl"));
    for row in &series {
        assert!((row["M"].as_f64().unwrap() - 1.0).abs() < 1e-12, "M must stay 1.0: {row}");
        assert!((row["E"].as_f64().unwrap() + 2.0).abs() < 1e-12, "E must stay -2.0: {row}");
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn high_temperature_randomizes_and_accepts_often() {
    let out = temp_dir("hight");
    let mut args = base_args(&out);
    set(&mut args, "--l", "8");
    set(&mut args, "--t-from", "5.0");
    set(&mut args, "--t-to", "5.0");
    set(&mut args, "--t-step", "1.0");
    set(&mut args, "--discard", "10");
    set(&mut args, "--measure", "50");
    let output = run_ising(&args);
    assert!(
        output.status.success(),
        "ising failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let row = stdout.lines().nth(1).unwrap();
    let fields: Vec<&str> = row.split('\t').collect();
    let mean_abs_m: f64 = fields[1].parse().unwrap();
    let acceptance: f64 = fields[2].parse().unwrap();
    assert!(acceptance > 0.5, "high-T acceptance should be large, got {acceptance}");
    assert!(mean_abs_m < 0.6, "high-T |M| should be small, got {mean_abs_m}");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn same_seed_reproduces_identical_artifacts() {
    let out1 = temp_dir("seed1");
    let out2 = temp_dir("seed2");
    let mut args = base_args(&out1);
    let mut args2 = base_args(&out2);
    set(&mut args, "--measure", "10");
    set(&mut args, "--every", "2");
    set(&mut args2, "--measure", "10");
    set(&mut args2, "--every", "2");
    assert!(run_ising(&args).status.success());
    assert!(run_ising(&args2).status.success());
    for name in ["run.json", "series.jsonl", "spins.jsonl"] {
        let a = std::fs::read(out1.join(name)).unwrap();
        let b = std::fs::read(out2.join(name)).unwrap();
        assert_eq!(a, b, "{name} differs between identical-seed runs");
    }
    let _ = std::fs::remove_dir_all(&out1);
    let _ = std::fs::remove_dir_all(&out2);
}

#[test]
fn wolff_low_temperature_flips_whole_lattice_each_step() {
    let out = temp_dir("wolff-lt");
    let mut args = base_args(&out);
    set(&mut args, "--update", "wolff");
    // T = 1e-9: p_add = 1 - exp(-2/T) = 1 exactly, so the cluster is the
    // whole same-spin connected component: size l*l, |M| = 1, E = -2 per
    // site, and the lattice flips all-up/all-down every step.
    set(&mut args, "--t-from", "0.000000001");
    set(&mut args, "--t-to", "0.000000001");
    set(&mut args, "--t-step", "1.0");
    set(&mut args, "--measure", "6");
    let output = run_ising(&args);
    assert!(
        output.status.success(),
        "ising failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let row = stdout.lines().nth(1).unwrap();
    let fields: Vec<&str> = row.split('\t').collect();
    assert_eq!(fields[0], "0.000000");
    assert_eq!(fields[1], "1.000000", "mean |M| stays 1 at T=1e-9");
    let mean_cluster: f64 = fields[2].parse().unwrap();
    assert_eq!(mean_cluster, 16.0, "every cluster is the whole l*l lattice");

    let series = read_jsonl(&out.join("series.jsonl"));
    assert_eq!(series.len(), 6);
    let ms: Vec<f64> = series.iter().map(|r| r["M"].as_f64().unwrap()).collect();
    for m in &ms {
        assert!((m.abs() - 1.0).abs() < 1e-12, "|M| must stay 1: {ms:?}");
    }
    for (i, m) in ms.iter().enumerate() {
        let expected = if i % 2 == 0 { -1.0 } else { 1.0 };
        assert_eq!(*m, expected, "lattice flips all-down/all-up: {ms:?}");
        let e = series[i]["E"].as_f64().unwrap();
        assert!((e + 2.0).abs() < 1e-12, "E stays -2 per site: {e}");
        assert_eq!(series[i]["cluster_size"].as_u64().unwrap(), 16);
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn wolff_high_temperature_makes_single_spin_clusters() {
    let out = temp_dir("wolff-ht");
    let mut args = base_args(&out);
    set(&mut args, "--update", "wolff");
    // T = 1e12: p_add = 1 - exp(-2/T) ~ 2e-12 ~ 0, so no neighbour is ever
    // added and every cluster is the seed site alone.
    set(&mut args, "--t-from", "1000000000000");
    set(&mut args, "--t-to", "1000000000000");
    set(&mut args, "--t-step", "1.0");
    set(&mut args, "--measure", "20");
    let output = run_ising(&args);
    assert!(
        output.status.success(),
        "ising failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let series = read_jsonl(&out.join("series.jsonl"));
    assert_eq!(series.len(), 20);
    for row in &series {
        assert_eq!(row["cluster_size"].as_u64().unwrap(), 1);
        assert!(row["M"].as_f64().unwrap().abs() <= 1.0);
        assert!(row["E"].as_f64().unwrap() >= -2.0);
    }
    // Single-spin flips mean mean |M| stays well below the ordered value.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let fields: Vec<&str> = stdout.lines().nth(1).unwrap().split('\t').collect();
    let mean_abs_m: f64 = fields[1].parse().unwrap();
    assert!(mean_abs_m < 0.6, "high-T |M| should be small: {mean_abs_m}");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn wolff_critical_temperature_disorders_lattice() {
    let out = temp_dir("wolff-tc");
    let mut args = base_args(&out);
    set(&mut args, "--update", "wolff");
    set(&mut args, "--l", "16");
    set(&mut args, "--t-from", "3.0");
    set(&mut args, "--t-to", "3.0");
    set(&mut args, "--t-step", "1.0");
    set(&mut args, "--discard", "500");
    set(&mut args, "--measure", "4000");
    let output = run_ising(&args);
    assert!(
        output.status.success(),
        "ising failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let fields: Vec<&str> = stdout.lines().nth(1).unwrap().split('\t').collect();
    let mean_abs_m: f64 = fields[1].parse().unwrap();
    assert!(mean_abs_m < 0.4, "T=3.0 |M| should be small: {mean_abs_m}");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn wolff_same_seed_reproduces_identical_artifacts() {
    let out1 = temp_dir("wolff-seed1");
    let out2 = temp_dir("wolff-seed2");
    let mut args = base_args(&out1);
    let mut args2 = base_args(&out2);
    set(&mut args, "--update", "wolff");
    set(&mut args, "--measure", "10");
    set(&mut args, "--every", "2");
    set(&mut args2, "--update", "wolff");
    set(&mut args2, "--measure", "10");
    set(&mut args2, "--every", "2");
    assert!(run_ising(&args).status.success());
    assert!(run_ising(&args2).status.success());
    for name in ["run.json", "series.jsonl", "spins.jsonl"] {
        let a = std::fs::read(out1.join(name)).unwrap();
        let b = std::fs::read(out2.join(name)).unwrap();
        assert_eq!(a, b, "{name} differs between identical-seed wolff runs");
    }
    let _ = std::fs::remove_dir_all(&out1);
    let _ = std::fs::remove_dir_all(&out2);
}
