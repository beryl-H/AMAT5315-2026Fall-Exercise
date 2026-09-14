//! Independent verification of saved artifacts.
//!
//! Physics acceptance metrics are recomputed from the raw positions and
//! velocities in traj.jsonl; the stored E_pot/E_kin are used only for a
//! cross-check, never as inputs to the acceptance bounds.

use crate::io::{read_artifacts, RunConfig};
use crate::metrics::{chi2_22, frame_total_energies, pooled_speeds, recompute_energies, secular_drift, t_speed};
use crate::simulate::Frame;
use crate::system::Box2;
use std::path::Path;

/// Result of the physics checks (structure already validated).
#[derive(Clone, Copy, Debug)]
pub struct CheckReport {
    pub drift: f64,
    pub drift_pass: bool,
    pub t_speed: f64,
    pub temperature_target: f64,
    pub temperature_pass: bool,
    pub chi2_22: f64,
    pub chi2_pass: bool,
    pub max_energy_mismatch: f64,
}

/// A structural or I/O failure (malformed artifacts), distinct from a physics
/// bound not being met (which is reported via CheckReport pass flags).
#[derive(Debug)]
pub struct CheckError(pub String);

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Structural validation [Suggestion implementing the sheet's malformed-file
/// Course Requirement] followed by the physics checks recomputed from raw
/// data.
pub fn check_artifacts(dir: &Path) -> Result<CheckReport, CheckError> {
    let (run, frames): (RunConfig, Vec<Frame>) = read_artifacts(dir).map_err(CheckError)?;

    // ---- structural validation before accepting the run ----
    if run.integrator != "velocity-verlet" {
        return Err(CheckError(format!(
            "integrator must be velocity-verlet, got {}",
            run.integrator
        )));
    }
    if run.sample_every == 0 || run.steps % run.sample_every != 0 {
        return Err(CheckError("steps must be a multiple of sample_every".into()));
    }
    let expected_frames = run.steps / run.sample_every;
    if frames.len() != expected_frames {
        return Err(CheckError(format!(
            "expected {expected_frames} frames, found {}",
            frames.len()
        )));
    }
    let bx = Box2 {
        lx: run.box_dim[0],
        ly: run.box_dim[1],
    };
    let bx_rebuilt = Box2::new(run.n, run.rho);
    // Box cross-check [Suggestion].
    if (bx.lx - bx_rebuilt.lx).abs() > 1e-10 * f64::max(1.0, bx.lx)
        || (bx.ly - bx_rebuilt.ly).abs() > 1e-10 * f64::max(1.0, bx.ly)
    {
        return Err(CheckError("stored box disagrees with n and rho".into()));
    }
    for (idx, frame) in frames.iter().enumerate() {
        let expected_step = (idx + 1) * run.sample_every;
        if frame.step != expected_step {
            return Err(CheckError(format!(
                "frame {idx}: step {} != {}",
                frame.step, expected_step
            )));
        }
        if (frame.t - frame.step as f64 * run.dt).abs() > 1e-10 * f64::max(1.0, frame.t) {
            return Err(CheckError(format!("frame {idx}: t inconsistent with step*dt")));
        }
        if frame.pos.len() != run.n || frame.vel.len() != run.n {
            return Err(CheckError(format!("frame {idx}: wrong array length")));
        }
        for p in &frame.pos {
            if !p[0].is_finite()
                || !p[1].is_finite()
                || !(0.0..bx.lx).contains(&p[0])
                || !(0.0..bx.ly).contains(&p[1])
            {
                return Err(CheckError(format!("frame {idx}: position outside box or non-finite")));
            }
        }
        for v in &frame.vel {
            if !v[0].is_finite() || !v[1].is_finite() {
                return Err(CheckError(format!("frame {idx}: non-finite velocity")));
            }
        }
        if !frame.e_pot.is_finite() || !frame.e_kin.is_finite() {
            return Err(CheckError(format!("frame {idx}: non-finite energy")));
        }
    }

    // ---- physics, recomputed from raw data ----
    let totals = frame_total_energies(&frames, &bx);
    let drift = secular_drift(&totals);
    let speeds = pooled_speeds(&frames);
    let ts = t_speed(&speeds);
    let chi = chi2_22(&speeds, ts);

    // Stored-energy cross-check [Suggestion]; never used in the metrics above.
    let mut max_mismatch: f64 = 0.0;
    for frame in &frames {
        let (re_pot, re_kin) = recompute_energies(frame, &bx);
        for (stored, recomputed) in [(frame.e_pot, re_pot), (frame.e_kin, re_kin)] {
            let rel = (stored - recomputed).abs() / f64::max(1.0, recomputed.abs());
            max_mismatch = max_mismatch.max(rel);
        }
    }
    if max_mismatch > 1e-10 {
        return Err(CheckError(format!(
            "stored energies disagree with recomputed values (max {max_mismatch})"
        )));
    }

    Ok(CheckReport {
        drift,
        drift_pass: drift < 2e-3,
        t_speed: ts,
        temperature_target: run.temperature,
        temperature_pass: (ts - run.temperature).abs() < 0.05,
        chi2_22: chi,
        chi2_pass: chi < 2.0,
        max_energy_mismatch: max_mismatch,
    })
}

/// Print the report and return the process exit code (0 iff all pass).
pub fn run_check(dir: &Path) -> i32 {
    match check_artifacts(dir) {
        Err(e) => {
            eprintln!("check FAILED: {e}");
            1
        }
        Ok(r) => {
            println!(
                "frames cross-check: max energy mismatch = {:.3e}",
                r.max_energy_mismatch
            );
            println!(
                "secular drift   = {:.6e}  (bound 2e-3)  {}",
                r.drift,
                if r.drift_pass { "PASS" } else { "FAIL" }
            );
            println!(
                "T_speed         = {:.6}    (target {:.2}, bound 0.05)  {}",
                r.t_speed,
                r.temperature_target,
                if r.temperature_pass { "PASS" } else { "FAIL" }
            );
            println!(
                "chi2_22         = {:.4}    (bound 2)  {}",
                r.chi2_22,
                if r.chi2_pass { "PASS" } else { "FAIL" }
            );
            if r.drift_pass && r.temperature_pass && r.chi2_pass {
                0
            } else {
                1
            }
        }
    }
}
