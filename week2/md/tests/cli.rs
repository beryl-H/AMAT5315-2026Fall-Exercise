//! Integration tests for the `md` binary's file and frame contract.
//!
//! Course Requirement: `md run` writes run.json (required fields,
//! integrator = "velocity-verlet") and traj.jsonl (one object per saved
//! production frame); saved steps follow --sample-every; step 0 is not
//! saved; the generic frame count is steps / sample_every.

use std::path::PathBuf;
use std::process::Command;

fn md_binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_md"))
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("md-cli-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn run_md(args: &[&str]) -> std::process::Output {
    md_binary().args(args).output().expect("failed to run md binary")
}

#[test]
fn run_writes_required_files_and_frames_for_a_small_run() {
    let out = temp_dir("small");
    let output = run_md(&[
        "run", "--n", "16", "--rho", "0.8", "--temperature", "0.5",
        "--dt", "0.01", "--eq-steps", "100", "--steps", "200",
        "--sample-every", "25", "--seed", "2026",
        "--out", out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "md run failed: {}", String::from_utf8_lossy(&output.stderr));

    // run.json required fields [Course Requirement].
    let run_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).expect("run.json missing"),
    )
    .expect("run.json malformed");
    for field in ["n", "rho", "box", "dt", "temperature", "eq_steps",
                  "steps", "sample_every", "seed", "integrator"] {
        assert!(run_json.get(field).is_some(), "run.json missing field {field}");
    }
    assert_eq!(run_json["integrator"], "velocity-verlet");
    assert_eq!(run_json["n"], 16);

    // traj.jsonl frames [Course Requirement].
    let traj = std::fs::read_to_string(out.join("traj.jsonl")).expect("traj.jsonl missing");
    let frames: Vec<serde_json::Value> = traj
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("traj line malformed"))
        .collect();
    // Generic rule: frame count = steps / sample_every (200/25 = 8).
    assert_eq!(frames.len(), 8);
    // Step 0 is never saved; steps follow sample_every.
    for (idx, frame) in frames.iter().enumerate() {
        let expected_step = (idx + 1) * 25;
        assert_eq!(frame["step"].as_u64().unwrap(), expected_step as u64);
        assert!((frame["t"].as_f64().unwrap() - expected_step as f64 * 0.01).abs() < 1e-12);
        assert_eq!(frame["pos"].as_array().unwrap().len(), 16);
        assert_eq!(frame["vel"].as_array().unwrap().len(), 16);
        assert!(frame["E_pot"].is_f64());
        assert!(frame["E_kin"].is_f64());
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn run_rejects_non_square_n_and_non_divisible_steps() {
    // [Suggestion] per design: clear nonzero-exit errors.
    let out = temp_dir("bad");
    let output = run_md(&["run", "--n", "50", "--out", out.to_str().unwrap()]);
    assert!(!output.status.success());
    let output = run_md(&["run", "--n", "16", "--steps", "100", "--sample-every", "30",
                          "--out", out.to_str().unwrap()]);
    assert!(!output.status.success());
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn run_accepts_both_force_methods_and_keeps_run_json_schema() {
    for m in ["naive", "cells"] {
        let out = temp_dir(&format!("force-{m}"));
        let output = run_md(&["run", "--n", "16", "--eq-steps", "0", "--steps", "100",
                              "--sample-every", "50", "--force", m, "--out", out.to_str().unwrap()]);
        assert!(output.status.success(), "{m} run failed: {}", String::from_utf8_lossy(&output.stderr));
        let traj = std::fs::read_to_string(out.join("traj.jsonl")).unwrap();
        assert_eq!(traj.lines().filter(|l| !l.trim().is_empty()).count(), 2);
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
        // run.json schema is unchanged: no force_method key.
        assert!(run.get("force_method").is_none(), "run.json must not contain force_method");
        let _ = std::fs::remove_dir_all(&out);
    }
}

#[test]
fn run_rejects_unknown_force_method() {
    let out = temp_dir("force-bad");
    let output = run_md(&["run", "--n", "16", "--force", "bogus", "--out", out.to_str().unwrap()]);
    assert!(!output.status.success());
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn default_run_uses_cells_force_method() {
    // Final contract: with no --force the run must use Cells. Same seed and
    // parameters make the no-flag trajectory bit-identical to an explicit
    // --force cells run and different from an explicit --force naive run.
    let out_default = temp_dir("flip-default");
    let out_cells = temp_dir("flip-cells");
    let out_naive = temp_dir("flip-naive");
    let common = ["--n", "100", "--eq-steps", "0", "--steps", "100", "--sample-every", "50"];
    for (name, out) in [("default", &out_default), ("cells", &out_cells), ("naive", &out_naive)] {
        let extra: &[&str] = match name {
            "default" => &[],
            "cells" => &["--force", "cells"],
            _ => &["--force", "naive"],
        };
        let mut args: Vec<&str> = vec!["run"];
        args.extend_from_slice(extra);
        args.extend_from_slice(&common);
        args.push("--out");
        args.push(out.to_str().unwrap());
        let output = run_md(&args);
        assert!(output.status.success(), "{name} run failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let def = std::fs::read_to_string(out_default.join("traj.jsonl")).unwrap();
    let cells = std::fs::read_to_string(out_cells.join("traj.jsonl")).unwrap();
    let naive = std::fs::read_to_string(out_naive.join("traj.jsonl")).unwrap();
    assert_eq!(def, cells, "no-flag default must be the Cells path");
    assert_ne!(def, naive, "the Naive path must produce a different trajectory");
    let _ = std::fs::remove_dir_all(&out_default);
    let _ = std::fs::remove_dir_all(&out_cells);
    let _ = std::fs::remove_dir_all(&out_naive);
}

#[test]
fn video_reports_missing_ffmpeg_or_succeeds() {
    // Course Requirement: one binary, md video artifacts --out PATH.
    // If ffmpeg is absent, exit nonzero with a clear message [design].
    let out = temp_dir("video");
    let artifacts = temp_dir("video-src");
    let run = run_md(&["run", "--n", "16", "--eq-steps", "50", "--steps", "100",
                       "--sample-every", "50", "--out", artifacts.to_str().unwrap()]);
    assert!(run.status.success());
    let video = run_md(&["video", artifacts.to_str().unwrap(),
                         "--out", out.join("run.mp4").to_str().unwrap()]);
    if md::video::ffmpeg_available() {
        assert!(video.status.success(), "md video failed: {}", String::from_utf8_lossy(&video.stderr));
        assert!(out.join("run.mp4").exists());
    } else {
        assert!(!video.status.success());
    }
    // --out is required: omitting it must fail to parse.
    let no_out = run_md(&["video", artifacts.to_str().unwrap()]);
    assert!(!no_out.status.success());
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&artifacts);
}

#[test]
fn default_contract_video_encodes_all_frames_under_two_mb() {
    // Course Requirement: the final MP4 is < 2 MB with every one of the 200
    // saved trajectory frames encoded (no subsampling).
    if !md::video::ffmpeg_available() {
        eprintln!("skipping: ffmpeg not installed");
        return;
    }
    let out = temp_dir("video-ctr");
    let artifacts = temp_dir("video-ctr-src");
    let run = run_md(&["run", "--out", artifacts.to_str().unwrap()]);
    assert!(run.status.success(), "contract run failed");
    let traj = std::fs::read_to_string(artifacts.join("traj.jsonl")).unwrap();
    let saved = traj.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(saved, 200, "default contract must save exactly 200 frames");

    let mp4 = out.join("run.mp4");
    let video = run_md(&["video", artifacts.to_str().unwrap(), "--out", mp4.to_str().unwrap()]);
    assert!(video.status.success(), "md video failed: {}", String::from_utf8_lossy(&video.stderr));
    assert!(mp4.exists());
    let size = std::fs::metadata(&mp4).unwrap().len();
    assert!(size > 0, "mp4 must be non-empty");
    assert!(size < 2_000_000, "mp4 is {size} bytes, must be < 2 MB");

    // Count encoded frames with ffprobe (if present): must equal the saved count.
    if let Ok(probe) = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"])
        .arg(&mp4)
        .output()
    {
        if probe.status.success() {
            let text = String::from_utf8_lossy(&probe.stdout);
            let encoded: usize = text.trim().parse().expect("ffprobe frame count");
            assert_eq!(encoded, saved, "must encode every saved frame");
        }
    }
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&artifacts);
}