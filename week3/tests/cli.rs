//! Integration tests for argument validation and error paths.

use std::path::PathBuf;
use std::process::{Command, Output};

fn ising_binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ising"))
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ising-cli-test-{tag}-{}", std::process::id()));
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

fn remove(args: &mut Vec<String>, flag: &str) {
    let pos = args.iter().position(|a| a == flag).expect("flag present in base args");
    args.drain(pos..=pos + 1);
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

#[test]
fn wolff_runs_and_writes_contract_artifacts() {
    let out = temp_dir("wolff");
    let mut args = base_args(&out);
    set(&mut args, "--update", "wolff");
    let output = run_ising(&args);
    assert!(
        output.status.success(),
        "ising failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // run.json records the wolff update and its cluster_flip time unit.
    let run: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).expect("run.json missing"),
    )
    .expect("run.json malformed");
    assert_eq!(run["update"], "wolff");
    assert_eq!(run["time_unit"], "cluster_flip");

    // series.jsonl: wolff rows add cluster_size (>= 1, at least the seed).
    let series = read_jsonl(&out.join("series.jsonl"));
    assert_eq!(series.len(), 5, "one row per measured cluster flip");
    for row in &series {
        assert_eq!(
            sorted_keys(row),
            ["E", "L", "M", "T", "cluster_size", "sweep"]
        );
        assert!(row["cluster_size"].as_u64().unwrap() >= 1);
    }

    // stdout header names the wolff third column.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.lines().next().unwrap(),
        "T\tmean_abs_M\tmean_cluster_size"
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn unknown_update_rejected() {
    let out = temp_dir("unknown");
    let mut args = base_args(&out);
    set(&mut args, "--update", "swendsen");
    let output = run_ising(&args);
    assert_ne!(output.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown"));
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn grid_not_integer_number_of_steps_rejected() {
    let out = temp_dir("grid");
    let mut args = base_args(&out);
    set(&mut args, "--t-to", "1.53"); // 1.0 -> 1.53 is not a multiple of 0.05
    let output = run_ising(&args);
    assert_ne!(output.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("integer number"), "stderr: {stderr}");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn zero_l_rejected() {
    let out = temp_dir("zerol");
    let mut args = base_args(&out);
    set(&mut args, "--l", "0");
    let output = run_ising(&args);
    assert_ne!(output.status.code(), Some(0));
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn measure_zero_rejected() {
    let out = temp_dir("zeromeasure");
    let mut args = base_args(&out);
    set(&mut args, "--measure", "0");
    let output = run_ising(&args);
    assert_ne!(output.status.code(), Some(0));
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn descending_ramp_rejected() {
    let out = temp_dir("desc");
    let mut args = base_args(&out);
    set(&mut args, "--t-from", "2.0");
    set(&mut args, "--t-to", "1.0");
    let output = run_ising(&args);
    assert_ne!(output.status.code(), Some(0));
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn missing_required_out_rejected() {
    let out = temp_dir("noout");
    let mut args = base_args(&out);
    remove(&mut args, "--out");
    let output = run_ising(&args);
    assert_ne!(output.status.code(), Some(0));
    let _ = std::fs::remove_dir_all(&out);
}
