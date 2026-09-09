//! End-to-end checks of the md CLI operations.

use md::cli::{CheckCfg, RunCfg, VideoCfg};
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
        n: 100,
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
    // P6 header with the right dimensions (2 panels wide, 1 tall), then RGB.
    const HEADER: &[u8] = b"P6\n900 450 255\n";
    assert_eq!(&ppm[..HEADER.len()], HEADER);
    assert_eq!(ppm.len(), HEADER.len() + 900 * 450 * 3);
    // An atom center lands on a non-white pixel in the LEFT panel: margin
    // p = 22.5 px, drawing side 405 px, atoms of radius ~14 px.
    let a = frame[0];
    let scale = 405.0 / t.meta.box_l;
    let px = 22.5 + a[0] * scale;
    let py = 22.5 + a[1] * scale;
    let idx = HEADER.len() + (py as usize) * 900 * 3 + (px as usize) * 3;
    assert!(ppm[idx] < 250 || ppm[idx + 1] < 250 || ppm[idx + 2] < 250);
}

#[test]
fn render_wraps_positions_into_the_box() {
    // Shifting every atom by one box length in x and y must not change the
    // frame: rem_euclid(box) makes the display periodic.
    let path = std::env::temp_dir().join("md_cli_run_test.txt");
    let t = read(path.to_str().unwrap()).unwrap();
    let frame = &t.frames[0];
    let box_l = t.meta.box_l;
    let positions: Vec<[f64; 2]> = frame.iter().map(|a| [a[0], a[1]]).collect();
    let shifted: Vec<[f64; 2]> = frame
        .iter()
        .map(|a| [a[0] + box_l, a[1] + 2.0 * box_l])
        .collect();
    let a = md::video::render_frame(box_l, &positions, &[], 300);
    let b = md::video::render_frame(box_l, &shifted, &[], 300);
    assert_eq!(a, b);
}

#[test]
fn rdf_gets_its_own_panel() {
    // A nonempty RDF curve draws blue pixels in the RIGHT half only;
    // the atoms-only LEFT half stays free of blue.
    let box_l = 10.0;
    let positions = vec![[1.0, 5.0], [4.0, 5.0]];
    let curve = vec![(1.0, 2.0), (2.0, 1.5), (3.0, 1.0)];
    let ppm = md::video::render_frame(box_l, &positions, &curve, 300);
    let is_blue = |i: usize| ppm[i] < 100 && ppm[i + 2] > 180;
    let (mut left_blue, mut right_blue) = (0, 0);
    for y in 0..300 {
        for x in 0..300 {
            let o = (y * 600 + x) * 3;
            left_blue += is_blue(o) as usize;
        }
        for x in 300..600 {
            let o = (y * 600 + x) * 3;
            right_blue += is_blue(o) as usize;
        }
    }
    assert_eq!(left_blue, 0, "left panel must not contain the RDF");
    assert!(right_blue > 0, "right panel should show the RDF curve");
}

#[test]
fn make_video_produces_an_mp4() {
    let path = std::env::temp_dir().join("md_cli_run_test.txt");
    let out = std::env::temp_dir().join("md_cli_video_test.mp4");
    let cfg = md::cli::VideoCfg {
        file: path.to_str().unwrap().into(),
        out: out.to_str().unwrap().into(),
        fps: 10.0,
    };
    md::ops::make_video(&cfg).unwrap();
    let meta = std::fs::metadata(&out).unwrap();
    assert!(meta.len() > 1000, "mp4 too small: {}", meta.len());
}
