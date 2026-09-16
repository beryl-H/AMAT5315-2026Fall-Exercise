//! run.json / traj.jsonl artifact reading and writing.
//!
//! Serialization/deserialization only: no physics is recomputed here.

use crate::simulate::{Frame, SimConfig};
use crate::system::Box2;
use crate::Vec2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;

/// Metadata written to run.json; field order matches the course contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_dim: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    /// Heating target; present in run.json only for heating runs
    /// [Course Requirement: heating runs record ramp_to; Suggestion:
    /// omitted when None so the unheated schema is unchanged].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ramp_to: Option<f64>,
}

impl From<SimConfig> for RunConfig {
    fn from(c: SimConfig) -> Self {
        let bx = Box2::new(c.n, c.rho);
        RunConfig {
            n: c.n,
            rho: c.rho,
            box_dim: [bx.lx, bx.ly],
            dt: c.dt,
            temperature: c.temperature,
            eq_steps: c.eq_steps,
            steps: c.steps,
            sample_every: c.sample_every,
            seed: c.seed,
            integrator: "velocity-verlet".to_string(),
            ramp_to: c.ramp_to,
        }
    }
}

impl From<&SimConfig> for RunConfig {
    fn from(c: &SimConfig) -> Self {
        RunConfig::from(*c)
    }
}

/// Serialized frame schema for traj.jsonl (write side, borrows the data).
#[derive(Serialize)]
struct FrameJson<'a> {
    step: usize,
    t: f64,
    pos: &'a [Vec2],
    vel: &'a [Vec2],
    #[serde(rename = "E_pot")]
    e_pot: f64,
    #[serde(rename = "E_kin")]
    e_kin: f64,
}

/// Owned frame schema for traj.jsonl (read side).
#[derive(Deserialize)]
struct FrameJsonOwned {
    step: usize,
    t: f64,
    pos: Vec<Vec2>,
    vel: Vec<Vec2>,
    #[serde(rename = "E_pot")]
    e_pot: f64,
    #[serde(rename = "E_kin")]
    e_kin: f64,
}

/// Write run.json and traj.jsonl into `dir` (created as needed).
/// run.json is a single JSON document; traj.jsonl is JSON Lines with exactly
/// one frame object per line and no surrounding array.
pub fn write_artifacts(dir: &Path, run: &RunConfig, frames: &[Frame]) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join("run.json"), serde_json::to_string_pretty(run).unwrap())?;
    let mut file = fs::File::create(dir.join("traj.jsonl"))?;
    for f in frames {
        let line = serde_json::to_string(&FrameJson {
            step: f.step,
            t: f.t,
            pos: &f.pos,
            vel: &f.vel,
            e_pot: f.e_pot,
            e_kin: f.e_kin,
        })
        .unwrap();
        writeln!(file, "{line}")?;
    }
    Ok(())
}

/// Read run.json and traj.jsonl from `dir`; Err(msg) on any malformed input.
pub fn read_artifacts(dir: &Path) -> Result<(RunConfig, Vec<Frame>), String> {
    let run_text = fs::read_to_string(dir.join("run.json"))
        .map_err(|e| format!("cannot read run.json: {e}"))?;
    let run: RunConfig = serde_json::from_str(&run_text)
        .map_err(|e| format!("malformed run.json: {e}"))?;
    let traj_text = fs::read_to_string(dir.join("traj.jsonl"))
        .map_err(|e| format!("cannot read traj.jsonl: {e}"))?;
    let mut frames = Vec::new();
    for (i, line) in traj_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: FrameJsonOwned = serde_json::from_str(line)
            .map_err(|e| format!("malformed traj.jsonl line {}: {e}", i + 1))?;
        frames.push(Frame {
            step: f.step,
            t: f.t,
            pos: f.pos,
            vel: f.vel,
            e_pot: f.e_pot,
            e_kin: f.e_kin,
        });
    }
    Ok((run, frames))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate::SimConfig;
    use std::path::PathBuf;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("md-io-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn sample_frames() -> Vec<crate::simulate::Frame> {
        vec![
            crate::simulate::Frame {
                step: 50,
                t: 0.5,
                pos: vec![[0.1, 0.2]],
                vel: vec![[0.3, 0.4]],
                e_pot: -1.5,
                e_kin: 0.25,
            },
            crate::simulate::Frame {
                step: 100,
                t: 1.0,
                pos: vec![[0.5, 0.6]],
                vel: vec![[0.7, 0.8]],
                e_pot: -1.4,
                e_kin: 0.30,
            },
        ]
    }

    #[test]
    fn round_trip_preserves_run_fields_and_frames() {
        let dir = temp("rt");
        let run = RunConfig::from(SimConfig::default());
        write_artifacts(&dir, &run, &sample_frames()).unwrap();
        let (run2, frames2) = read_artifacts(&dir).unwrap();
        assert_eq!(run2.integrator, "velocity-verlet");
        assert_eq!(run2.n, 100);
        assert_eq!(run2.sample_every, 50);
        // serde_json 1.0.151's float parser is not correctly rounded (its
        // default feature set no longer enables float_roundtrip, and ryu is
        // unavailable offline): for the box Ly edge value 10.404478625719541
        // it returns the adjacent f64 (1 ulp lower). So float fields are
        // compared within a tight tolerance rather than with exact equality.
        for axis in 0..2 {
            assert!(
                (run2.box_dim[axis] - run.box_dim[axis]).abs() < 1e-12,
                "box axis {axis}: {} vs {}",
                run2.box_dim[axis],
                run.box_dim[axis]
            );
        }
        assert_eq!(frames2.len(), 2);
        assert_eq!(frames2[0].step, 50);
        assert!((frames2[1].t - 1.0).abs() < 1e-12);
        // Positions are exact for these values (0.1, 0.2 round-trip exactly).
        assert_eq!(frames2[0].pos, vec![[0.1, 0.2]]);
        assert_eq!(frames2[1].pos, vec![[0.5, 0.6]]);
        assert_eq!(frames2[0].vel, vec![[0.3, 0.4]]);
        assert_eq!(frames2[1].vel, vec![[0.7, 0.8]]);
        assert!((frames2[0].e_pot - -1.5).abs() < 1e-12);
        assert!((frames2[0].e_kin - 0.25).abs() < 1e-12);
        assert!((frames2[1].e_pot - -1.4).abs() < 1e-12);
        assert!((frames2[1].e_kin - 0.30).abs() < 1e-12);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_json_keys_are_exactly_the_course_contract() {
        let dir = temp("keys");
        let run = RunConfig::from(SimConfig::default());
        write_artifacts(&dir, &run, &sample_frames()).unwrap();
        let text = std::fs::read_to_string(dir.join("run.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        for key in [
            "n",
            "rho",
            "box",
            "dt",
            "temperature",
            "eq_steps",
            "steps",
            "sample_every",
            "seed",
            "integrator",
        ] {
            assert!(keys.contains(&key), "missing key {key}");
        }
        // traj.jsonl: one JSON object per line with the frame fields.
        let traj = std::fs::read_to_string(dir.join("traj.jsonl")).unwrap();
        let line: serde_json::Value = serde_json::from_str(traj.lines().next().unwrap()).unwrap();
        for key in ["step", "t", "pos", "vel", "E_pot", "E_kin"] {
            assert!(line.get(key).is_some(), "traj frame missing {key}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_errors_on_missing_or_malformed_files() {
        let dir = temp("missing");
        assert!(read_artifacts(&dir).is_err());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("run.json"), "{ not json").unwrap();
        std::fs::write(dir.join("traj.jsonl"), "").unwrap();
        assert!(read_artifacts(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ramp_to_round_trips_and_omits_when_none() {
        let dir = temp("ramp");
        // heated: Some(1.2) serialized and read back
        let mut c = crate::simulate::SimConfig::default();
        c.ramp_to = Some(1.2);
        let run = RunConfig::from(c);
        write_artifacts(&dir, &run, &sample_frames()).unwrap();
        let text = std::fs::read_to_string(dir.join("run.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["ramp_to"], 1.2);
        let (run2, _) = read_artifacts(&dir).unwrap();
        assert_eq!(run2.ramp_to, Some(1.2));
        // unheated: key absent, reads back as None
        let dir2 = temp("ramp2");
        let run3 = RunConfig::from(crate::simulate::SimConfig::default());
        assert_eq!(run3.ramp_to, None);
        write_artifacts(&dir2, &run3, &sample_frames()).unwrap();
        let text2 = std::fs::read_to_string(dir2.join("run.json")).unwrap();
        assert!(!text2.contains("ramp_to"));
        let (run4, _) = read_artifacts(&dir2).unwrap();
        assert_eq!(run4.ramp_to, None);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }
}
