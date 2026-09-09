//! End-to-end checks of the md CLI operations.

use md::cli::RunCfg;
use md::ops::run_sim;
use md::trajectory::read;

#[test]
fn run_writes_a_checkable_trajectory() {
    let path = std::env::temp_dir().join("md_cli_run_test.txt");
    let path = path.to_str().unwrap().to_string();
    let cfg = RunCfg {
        temp: 1.0,
        dt: 0.005,
        steps: 500,
        equil: 100,
        out: path.clone(),
        seed: 1,
    };
    run_sim(&cfg).unwrap();
    let t = read(&path).unwrap();
    assert_eq!(t.meta.n, 100);
    assert_eq!(t.frames.len(), 500);
    // Right after exact rescaling, Verlet drift over 500 steps is tiny:
    // the mean temperature stays close to the target.
    let mean_t: f64 = t
        .frames
        .iter()
        .map(|f| f.iter().map(|a| a[2] * a[2] + a[3] * a[3]).sum::<f64>() / 200.0)
        .sum::<f64>()
        / t.frames.len() as f64;
    assert!((mean_t - 1.0).abs() < 0.05 * 1.0, "mean T {mean_t}");
}
