//! End-to-end checks of the md CLI operations.

use md::cli::{CheckCfg, RunCfg};
use md::ops::run_sim;
use md::trajectory::read;

fn check_cfg(path: &str) -> CheckCfg {
    CheckCfg {
        file: path.into(),
        temp_tol: 0.05,
        drift_tol: 1e-3,
        ks_tol: 0.05,
    }
}

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

#[test]
fn check_passes_on_our_own_run() {
    let path = std::env::temp_dir().join("md_cli_run_test.txt");
    let path = path.to_str().unwrap().to_string();
    let report = md::ops::check(&check_cfg(&path)).unwrap();
    assert!(report.pass, "report {report:?}");
}

#[test]
fn check_rejects_doctored_temperature() {
    // Re-scale all velocities by 1.5: mean temperature rises ~2.25x, out of tolerance.
    let src = std::env::temp_dir().join("md_cli_run_test.txt");
    let mut t = read(src.to_str().unwrap()).unwrap();
    for frame in &mut t.frames {
        for a in frame {
            a[2] *= 1.5;
            a[3] *= 1.5;
        }
    }
    let path = std::env::temp_dir().join("md_cli_doctored.txt");
    md::trajectory::write(path.to_str().unwrap(), &t).unwrap();
    let report = md::ops::check(&check_cfg(path.to_str().unwrap())).unwrap();
    assert!(!report.pass);
}

#[test]
fn energy_of_frames_is_consistent() {
    // The drift metric on an NVE Verlet run stays far below the tolerance.
    let path = std::env::temp_dir().join("md_cli_run_test.txt");
    let report = md::ops::check(&check_cfg(path.to_str().unwrap())).unwrap();
    assert!(report.drift < 1e-3, "drift {}", report.drift);
}

#[test]
fn frames_render_as_ppm() {
    let path = std::env::temp_dir().join("md_cli_run_test.txt");
    let t = read(path.to_str().unwrap()).unwrap();
    let frame = &t.frames[0];
    let positions: Vec<[f64; 2]> = frame.iter().map(|a| [a[0], a[1]]).collect();
    let ppm = md::video::render_frame(t.meta.box_l, &positions, &[], 450);
    // P6 header with the right dimensions, then RGB data.
    const HEADER: &[u8] = b"P6\n450 450 255\n";
    assert_eq!(&ppm[..HEADER.len()], HEADER);
    assert_eq!(ppm.len(), HEADER.len() + 450 * 450 * 3);
    // An atom center lands on a non-white pixel. The implementation draws a
    // 22.5 px margin and a 405 px panel with atoms of radius ~14 px, so any
    // point within ~10 px of the projected center is safely inside the dot.
    let a = frame[0];
    let scale = 405.0 / t.meta.box_l;
    let px = 22.5 + a[0] * scale;
    let py = 22.5 + a[1] * scale;
    let idx = HEADER.len() + (py as usize) * 450 * 3 + (px as usize) * 3;
    assert!(ppm[idx] < 250 || ppm[idx + 1] < 250 || ppm[idx + 2] < 250);
}
