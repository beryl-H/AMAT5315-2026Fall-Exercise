//! The full default-contract acceptance test (ordinary, non-ignored).
//!
//! Course Requirement: with the contract defaults (N = 100, rho = 0.8,
//! T = 0.5, dt = 0.01, eq 2000, steps 10000, sample 50, seed 2026) the run
//! produces exactly 200 frames and `md check` passes all three bounds:
//! secular drift < 2e-3, |T_speed - 0.5| < 0.05, chi2_22 < 2.

use md::checker::check_artifacts;
use md::io::read_artifacts;
use md::simulate::{SimConfig, run_simulation};

#[test]
fn default_contract_run_passes_the_three_physics_bounds() {
    let config = SimConfig::default();
    let frames = run_simulation(&config);

    // Exactly 200 frames: steps/sample_every [Course Requirement].
    assert_eq!(frames.len(), 10000 / 50);
    // Saved steps are 50, 100, ..., 10000; step 0 never saved.
    for (idx, frame) in frames.iter().enumerate() {
        assert_eq!(frame.step, (idx + 1) * 50);
        assert!((frame.t - frame.step as f64 * config.dt).abs() < 1e-12);
    }

    // Write to a temp dir and run the real checker end to end.
    let out = std::env::temp_dir().join(format!("md-contract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    md::io::write_artifacts(&out, &config.into(), &frames).expect("write artifacts");

    let (run_config, traj) = read_artifacts(&out).expect("read artifacts");
    assert_eq!(run_config.integrator, "velocity-verlet");
    assert_eq!(traj.len(), 200);

    let report = check_artifacts(&out).expect("check must succeed on valid artifacts");
    assert!(report.drift_pass, "drift = {}", report.drift);
    assert!(report.temperature_pass, "T_speed = {}", report.t_speed);
    assert!(report.chi2_pass, "chi2_22 = {}", report.chi2_22);

    let _ = std::fs::remove_dir_all(&out);
}