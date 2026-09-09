//! Command-line parsing for the md tool.

pub struct RunCfg {
    pub temp: f64,
    pub dt: f64,
    pub steps: usize,
    pub equil: usize,
    pub out: String,
    pub seed: u64,
}

pub struct CheckCfg {
    pub file: String,
    pub temp_tol: f64,
    pub drift_tol: f64,
    pub ks_tol: f64,
}

pub struct VideoCfg {
    pub file: String,
    pub out: String,
    pub fps: f64,
}

pub enum Command {
    Run(RunCfg),
    Check(CheckCfg),
    Video(VideoCfg),
}

/// Parse subcommand + `--flag value` pairs with spec defaults.
pub fn parse(args: &[String]) -> Result<Command, String> {
    let sub = args.first().ok_or("usage: md <run|check|video> [flags]")?;
    let mut flags: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut positional: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(name) = a.strip_prefix("--") {
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("flag {name} needs a value"))?;
            flags.insert(name.to_string(), value.clone());
            i += 2;
        } else {
            positional.push(a.clone());
            i += 1;
        }
    }
    let num = |name: &str| -> Result<Option<f64>, String> {
        flags
            .get(name)
            .map(|v| v.parse::<f64>().map_err(|_| format!("--{name}: not a number")))
            .transpose()
    };
    let uint = |name: &str| -> Result<Option<usize>, String> {
        flags
            .get(name)
            .map(|v| v.parse::<usize>().map_err(|_| format!("--{name}: not an integer")))
            .transpose()
    };
    match sub.as_str() {
        "run" => {
            reject_unknown(&flags, &["temp", "dt", "steps", "equil", "out", "seed"])?;
            Ok(Command::Run(RunCfg {
            temp: num("temp")?.unwrap_or(1.0),
            dt: num("dt")?.unwrap_or(0.005),
            steps: uint("steps")?.unwrap_or(10000),
            equil: uint("equil")?.unwrap_or(1000),
            out: flags.get("out").cloned().unwrap_or_else(|| "trajectory.txt".into()),
            seed: flags
                .get("seed")
                .map(|v| v.parse::<u64>().map_err(|_| "--seed: not an integer"))
                .transpose()?
                .unwrap_or(1),
            })
        }
        "check" => {
            let file = positional.first().cloned().ok_or("check needs a trajectory file")?;
            reject_unknown(&flags, &["temp-tol", "drift-tol", "ks-tol"])?;
            Ok(Command::Check(CheckCfg {
                file,
                temp_tol: num("temp-tol")?.unwrap_or(0.05),
                drift_tol: num("drift-tol")?.unwrap_or(1e-3),
                ks_tol: num("ks-tol")?.unwrap_or(0.05),
            }))
        }
        "video" => {
            let file = positional.first().cloned().ok_or("video needs a trajectory file")?;
            reject_unknown(&flags, &["out", "fps"])?;
            Ok(Command::Video(VideoCfg {
                file,
                out: flags.get("out").cloned().unwrap_or_else(|| "video.mp4".into()),
                fps: num("fps")?.unwrap_or(30.0),
            }))
        }
        other => Err(format!("unknown command {other:?}; use run, check, or video")),
    }
}

/// Reject any flag not in the allowed set for the subcommand.
fn reject_unknown(
    flags: &std::collections::HashMap<String, String>,
    allowed: &[&str],
) -> Result<(), String> {
    match flags.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(bad) => Err(format!("unknown flag --{bad}")),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse, Command};

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn defaults() {
        let c = parse(&args(&["run"])).unwrap();
        match c {
            Command::Run(cfg) => {
                assert_eq!(cfg.temp, 1.0);
                assert_eq!(cfg.dt, 0.005);
                assert_eq!(cfg.steps, 10000);
                assert_eq!(cfg.equil, 1000);
                assert_eq!(cfg.out, "trajectory.txt");
                assert_eq!(cfg.seed, 1);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn overrides() {
        let c = parse(&args(&[
            "run", "--temp", "0.5", "--dt", "0.01", "--steps", "10", "--equil", "5",
            "--out", "t.txt", "--seed", "9",
        ]))
        .unwrap();
        match c {
            Command::Run(cfg) => {
                assert_eq!(cfg.temp, 0.5);
                assert_eq!(cfg.dt, 0.01);
                assert_eq!(cfg.steps, 10);
                assert_eq!(cfg.equil, 5);
                assert_eq!(cfg.out, "t.txt");
                assert_eq!(cfg.seed, 9);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn check_and_video() {
        let c = parse(&args(&["check", "traj.txt", "--ks-tol", "0.1"])).unwrap();
        match c {
            Command::Check(cfg) => {
                assert_eq!(cfg.file, "traj.txt");
                assert_eq!(cfg.temp_tol, 0.05);
                assert_eq!(cfg.drift_tol, 1e-3);
                assert_eq!(cfg.ks_tol, 0.1);
            }
            _ => panic!("wrong command"),
        }
        let c = parse(&args(&["video", "traj.txt", "--out", "v.mp4", "--fps", "24"])).unwrap();
        match c {
            Command::Video(cfg) => {
                assert_eq!(cfg.file, "traj.txt");
                assert_eq!(cfg.out, "v.mp4");
                assert_eq!(cfg.fps, 24.0);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn errors() {
        assert!(parse(&args(&[])).is_err());
        assert!(parse(&args(&["fly"])).is_err());
        assert!(parse(&args(&["run", "--temp"])).is_err());
        assert!(parse(&args(&["run", "--temp", "hot"])).is_err());
        assert!(parse(&args(&["run", "--mystery", "1"])).is_err());
        assert!(parse(&args(&["check"])).is_err()); // needs a file
    }
}
