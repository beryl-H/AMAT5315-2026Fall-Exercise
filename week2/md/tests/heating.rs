//! Heating ramp acceptance tests: per-step [Suggestion] production
//! thermostat, run.json recording, and no-ramp preservation.

use md::io::{RunConfig, write_artifacts};
use md::simulate::{SimConfig, run_simulation};
use md::thermodynamic_temperature;

const TOL: f64 = 1e-9;

fn heated_config(steps: usize, sample_every: usize) -> SimConfig {
    SimConfig {
        n: 16,
        rho: 0.8,
        temperature: 0.2,
        dt: 0.01,
        eq_steps: 0,
        steps,
        sample_every,
        seed: 2026,
        force_method: md::ForceMethod::Cells,
        ramp_to: Some(1.2),
    }
}

fn ramp_expected(s: usize, s_total: usize) -> f64 {
    0.2 + (1.2 - 0.2) * (s as f64) / (s_total as f64)
}

fn frame_temp(frame: &md::Frame) -> f64 {
    thermodynamic_temperature(&frame.vel)
}

#[test]
fn heated_production_pins_every_saved_frame_and_reaches_endpoint() {
    // S = 100 (divisible by 50) and S = 125 (not divisible by 50): the
    // per-step thermostat must pin every saved frame to T_target(s) and
    // apply exactly T1 at the final step s = S.
    for (steps, sample_every) in [(100usize, 25usize), (125usize, 25usize)] {
        let config = heated_config(steps, sample_every);
        let frames = run_simulation(&config);
        assert_eq!(frames.len(), steps / sample_every);
        for f in &frames {
            let expected = ramp_expected(f.step, steps);
            let t = frame_temp(f);
            assert!((t - expected).abs() < TOL, "step {}: T {t} != {expected}", f.step);
        }
        let last = frames.last().unwrap();
        assert_eq!(last.step, steps);
        assert!((frame_temp(last) - 1.2).abs() < TOL, "endpoint T1 not reached");
    }
}

#[test]
fn no_ramp_leaves_production_thermostat_off() {
    // Without ramp_to the production thermostat stays completely OFF: a
    // frame at a production multiple of 50 is NOT pinned to any target.
    let config = SimConfig {
        n: 16,
        rho: 0.8,
        temperature: 0.5,
        dt: 0.01,
        eq_steps: 0,
        steps: 200,
        sample_every: 25,
        seed: 2026,
        force_method: md::ForceMethod::Cells,
        ramp_to: None,
    };
    let frames = run_simulation(&config);
    let pinned = frames
        .iter()
        .filter(|f| f.step % 50 == 0)
        .all(|f| (frame_temp(f) - 0.5).abs() < 1e-6);
    assert!(!pinned, "production thermostat must stay OFF without --ramp-to");
}

#[test]
fn heating_run_json_records_ramp_to_and_unheated_omits_it() {
    let dir = std::env::temp_dir().join(format!("md-heat-json-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let heated = heated_config(100, 25);
    let frames = run_simulation(&heated);
    write_artifacts(&dir, &RunConfig::from(&heated), &frames).unwrap();
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(v["ramp_to"], 1.2);
    let _ = std::fs::remove_dir_all(&dir);

    let dir2 = std::env::temp_dir().join(format!("md-heat-json2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir2);
    let mut plain = heated_config(100, 25);
    plain.ramp_to = None;
    let frames2 = run_simulation(&plain);
    write_artifacts(&dir2, &RunConfig::from(&plain), &frames2).unwrap();
    let v2: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir2.join("run.json")).unwrap(),
    )
    .unwrap();
    assert!(v2.get("ramp_to").is_none(), "unheated run.json must omit ramp_to");
    let _ = std::fs::remove_dir_all(&dir2);
}
