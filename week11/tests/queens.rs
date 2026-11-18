use serde_json::Value;
use std::{collections::BTreeSet, fs, path::PathBuf, process::Command};

fn sample_boards(n: usize, samples: usize) -> Value {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(format!("test-queens-{n}-{}", std::process::id()));
    let result = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args([
            "--mode",
            "sample",
            "--n",
            &n.to_string(),
            "--samples",
            &samples.to_string(),
            "--seed",
            "2026",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&fs::read(out.join("run.json")).unwrap()).unwrap()
}

#[test]
fn sparse_carries_count_and_sample_both_four_queen_boards() {
    let data = sample_boards(4, 100);
    assert_eq!(data["board_count"], 2);
    let expected = BTreeSet::from([vec![1, 3, 0, 2], vec![2, 0, 3, 1]]);
    let samples = data["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 100);
    let actual: BTreeSet<Vec<usize>> = samples
        .iter()
        .map(|board| serde_json::from_value(board.clone()).unwrap())
        .collect();
    assert_eq!(actual, expected);
}
