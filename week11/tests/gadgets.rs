use serde_json::Value;
use std::{fs, path::PathBuf, process::Command};

#[test]
fn enumerated_energies_match_independent_truth_tables() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(format!("test-gadgets-{}", std::process::id()));
    let result = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args(["--mode", "gadgets", "--seed", "2026", "--out"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let document: Value = serde_json::from_slice(&fs::read(out.join("run.json")).unwrap()).unwrap();
    let expected: [(&str, &[i64], &[usize]); 7] = [
        ("AND", &[-3, 9, -3, 1, -3, 1, 1, -3], &[0, 2, 4, 7]),
        ("OR", &[-3, 1, 1, -3, 1, -3, 9, -3], &[0, 3, 5, 7]),
        ("NOT", &[1, -1, -1, 1], &[1, 2]),
        ("COPY", &[-1, 1, 1, -1], &[0, 3]),
        ("SET0", &[-1, 1], &[0]),
        ("SET1", &[1, -1], &[1]),
        ("XOR", &[-4, -2, -2, -4, -2, -4, -4, -2], &[0, 3, 5, 6]),
    ];
    let gates = document["gates"].as_object().unwrap();
    assert_eq!(gates.len(), expected.len());
    for (name, energies, valid) in expected {
        let rows = gates[name].as_array().unwrap();
        assert_eq!(rows.len(), energies.len(), "{name}");
        for (i, row) in rows.iter().enumerate() {
            let width = energies.len().ilog2() as usize;
            assert_eq!(row["bits"], format!("{i:0width$b}"), "{name}");
            assert_eq!(row["energy"], energies[i], "{name} row {i}");
            assert_eq!(row["valid"], valid.contains(&i), "{name} row {i}");
        }
    }
    let circuit = document["circuit"].as_array().unwrap();
    assert_eq!(circuit.len(), 16);
    let actual: Vec<_> = circuit
        .iter()
        .map(|row| row["energy"].as_i64().unwrap())
        .collect();
    assert_eq!(
        actual,
        [-1, -5, 9, 9, -1, -5, 1, 1, -1, -5, 1, 1, 3, -1, -3, -3]
    );
    let floor: Vec<_> = circuit
        .iter()
        .filter(|r| r["energy"] == -5)
        .map(|r| r["bits"].as_str().unwrap())
        .collect();
    assert_eq!(floor, ["0001", "0101", "1001"]);
}
