//! Checker tests: metrics recomputed from raw data, structural validation,
//! and negative controls [Suggestion].

use md::checker::{check_artifacts, run_check};
use md::fluid_potential_energy;
use md::fluid::ForceMethod;
use md::io::{RunConfig, write_artifacts};
use md::simulate::{Frame, SimConfig};
use md::system::Box2;
use md::State;
use std::path::PathBuf;

fn temp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("md-check-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// In-box positions for the synthetic n = 4, rho = 0.8 box
/// (Box2::new(4, 0.8) ~ [2.403, 2.081]) so the box cross-check passes.
fn in_box_positions() -> Vec<[f64; 2]> {
    vec![[0.4, 0.4], [1.2, 0.6], [0.6, 1.4], [1.5, 1.2]]
}

/// Two structurally valid frames (steps 50, 100; t = step*dt) with the same
/// in-box positions and the given velocities; stored energies are recomputed
/// from the raw data so the stored-energy cross-check passes.
fn synthetic_frames(vel: Vec<[f64; 2]>) -> Vec<Frame> {
    let bx = Box2::new(4, 0.8);
    let pos = in_box_positions();
    let state = State {
        positions: pos.clone(),
        velocities: vel.clone(),
    };
    let e_pot = fluid_potential_energy(&state, &bx);
    let e_kin = 0.5 * vel.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum::<f64>();
    (1..=2)
        .map(|s| Frame {
            step: s * 50,
            t: (s * 50) as f64 * 0.01,
            pos: pos.clone(),
            vel: vel.clone(),
            e_pot,
            e_kin,
        })
        .collect()
}

fn write_synthetic(tag: &str, frames: &[Frame], run: Option<RunConfig>) -> PathBuf {
    let dir = temp(tag);
    let run = run.unwrap_or_else(|| {
        RunConfig::from(SimConfig {
            n: 4,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 0,
            steps: 100,
            sample_every: 50,
            seed: 2026,
            force_method: ForceMethod::Naive,
            ramp_to: None,
        })
    });
    write_artifacts(&dir, &run, frames).unwrap();
    dir
}

#[test]
fn malformed_missing_files_fail() {
    assert!(check_artifacts(&temp("none")).is_err());
}

#[test]
fn wrong_integrator_fails() {
    let mut run = RunConfig::from(SimConfig {
        n: 4,
        rho: 0.8,
        temperature: 0.5,
        dt: 0.01,
        eq_steps: 0,
        steps: 100,
        sample_every: 50,
        seed: 2026,
        force_method: ForceMethod::Naive,
        ramp_to: None,
    });
    run.integrator = "euler".into();
    let dir = write_synthetic("integrator", &synthetic_frames(vec![[0.1, 0.1]; 4]), Some(run));
    assert!(check_artifacts(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wrong_frame_count_or_steps_fail() {
    let run = RunConfig::from(SimConfig {
        n: 4,
        rho: 0.8,
        temperature: 0.5,
        dt: 0.01,
        eq_steps: 0,
        steps: 150,
        sample_every: 50,
        seed: 2026,
        force_method: ForceMethod::Naive,
        ramp_to: None,
    });    // 2 frames written but steps says 3 should be saved.
    let dir = write_synthetic("count", &synthetic_frames(vec![[0.1, 0.1]; 4]), Some(run));
    assert!(check_artifacts(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn out_of_box_position_fails() {
    let mut frames = synthetic_frames(vec![[0.1, 0.1]; 4]);
    frames[0].pos[0] = [10.5, 0.5]; // outside [0, Lx)
    let dir = write_synthetic("oob", &frames, None);
    assert!(check_artifacts(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stored_energy_mismatch_fails() {
    let mut frames = synthetic_frames(vec![[0.1, 0.1]; 4]);
    frames[0].e_kin += 1.0; // stored value disagrees with raw velocities
    let dir = write_synthetic("energy", &frames, None);
    assert!(check_artifacts(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn run_check_exit_codes() {
    // Valid structure but wrong temperature (all velocities tiny): the
    // temperature bound must FAIL -> nonzero exit. Malformed -> nonzero.
    let dir = write_synthetic("exit", &synthetic_frames(vec![[0.01, 0.01]; 4]), None);
    assert_ne!(run_check(&dir), 0);
    assert_ne!(run_check(&temp("none2")), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn excessive_drift_fails() {
    // Frame 1 has tiny velocities, frame 2 large ones: the recomputed total
    // energies differ hugely, so the secular-drift bound must FAIL. The
    // structure is otherwise valid (cross-check passes).
    let bx = Box2::new(4, 0.8);
    let pos = in_box_positions();
    let mk = |s: usize, vel: &[[f64; 2]]| {
        let state = State {
            positions: pos.clone(),
            velocities: vel.to_vec(),
        };
        Frame {
            step: s * 50,
            t: (s * 50) as f64 * 0.01,
            pos: pos.clone(),
            vel: vel.to_vec(),
            e_pot: fluid_potential_energy(&state, &bx),
            e_kin: 0.5 * vel.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum::<f64>(),
        }
    };
    let frames = vec![mk(1, &[[0.1, 0.1]; 4]), mk(2, &[[5.0, 5.0]; 4])];
    let dir = write_synthetic("drift", &frames, None);
    let report = check_artifacts(&dir).expect("structure must be valid");
    assert!(!report.drift_pass, "drift {} should FAIL the 2e-3 bound", report.drift);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wrong_temperature_fails() {
    // All velocities tiny -> T_speed far below the 0.5 target.
    let dir = write_synthetic("temp", &synthetic_frames(vec![[0.01, 0.01]; 4]), None);
    let report = check_artifacts(&dir).expect("structure must be valid");
    assert!(
        !report.temperature_pass,
        "T_speed {} should FAIL the 0.05 gate",
        report.t_speed
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn non_maxwell_speed_shape_fails() {
    // All speeds identical -> Maxwell/Rayleigh chi2_22 blows up.
    let dir = write_synthetic("chi2", &synthetic_frames(vec![[0.5, 0.0]; 4]), None);
    let report = check_artifacts(&dir).expect("structure must be valid");
    assert!(!report.chi2_pass, "chi2_22 {} should FAIL the 2 bound", report.chi2_22);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn end_to_end_corrupted_trajectory_fails() {
    // [Suggestion] end-to-end negative control: take a real small run and
    // corrupt one raw velocity, then run the real check path.
    let dir = temp("e2e");
    let config = SimConfig {
        n: 16,
        rho: 0.8,
        temperature: 0.5,
        dt: 0.01,
        eq_steps: 100,
        steps: 100,
        sample_every: 50,
        seed: 2026,
        force_method: ForceMethod::Naive,
        ramp_to: None,
    };
    let frames = md::simulate::run_simulation(&config);
    write_artifacts(&dir, &RunConfig::from(&config), &frames).unwrap();
    // Corrupt: replace a raw velocity component with a non-number.
    let traj = std::fs::read_to_string(dir.join("traj.jsonl")).unwrap();
    let mut lines: Vec<String> = traj.lines().map(str::to_string).collect();
    let mut v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    v["vel"][0][0] = serde_json::json!("not-a-number");
    lines[0] = serde_json::to_string(&v).unwrap();
    std::fs::write(dir.join("traj.jsonl"), lines.join("\n") + "\n").unwrap();
    assert_ne!(run_check(&dir), 0, "corrupted run must FAIL check");
    let _ = std::fs::remove_dir_all(&dir);
}
