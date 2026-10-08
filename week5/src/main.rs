use seismic::{Experiment, Result, npy, simulation};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, path::PathBuf, time::Instant};

fn main() {
    if let Err(e) = run() {
        eprintln!("seismic: {e}");
        std::process::exit(1);
    }
}

fn flags() -> Result<BTreeMap<String, String>> {
    let mut args = env::args().skip(1);
    let mut flags = BTreeMap::new();
    while let Some(key) = args.next() {
        if key == "--enzyme-smoke" {
            flags.insert(key, "true".into());
            continue;
        }
        if key == "--help" {
            println!("seismic --experiment FILE --mode forward --out DIR [--every STRIDE]");
            std::process::exit(0);
        }
        if ![
            "--experiment",
            "--mode",
            "--out",
            "--every",
            "--data",
            "--storage",
            "--checkpoints",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown argument {key}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if flags.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate flag {key}").into());
        }
    }
    Ok(flags)
}

fn run() -> Result<()> {
    let flags = flags()?;
    if flags.contains_key("--enzyme-smoke") {
        println!("{}", json!(seismic::smoke()));
        return Ok(());
    }
    let input = PathBuf::from(
        flags
            .get("--experiment")
            .ok_or("--experiment is required")?,
    );
    let mode = flags.get("--mode").ok_or("--mode is required")?;
    if !["forward", "born", "adjoint"].contains(&mode.as_str()) {
        return Err("mode must be forward, born or adjoint".into());
    }
    let storage = flags.get("--storage").map(String::as_str).unwrap_or("full");
    if !["full", "treeverse"].contains(&storage) {
        return Err("storage must be full or treeverse".into());
    }
    let budget = flags
        .get("--checkpoints")
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    if mode != "adjoint" && (flags.contains_key("--storage") || flags.contains_key("--checkpoints"))
    {
        return Err("storage flags require adjoint mode".into());
    }
    if storage == "treeverse" && budget == 0 {
        return Err("treeverse requires a positive --checkpoints budget".into());
    }
    if storage == "full" && flags.contains_key("--checkpoints") {
        return Err("--checkpoints requires treeverse storage".into());
    }
    let out = PathBuf::from(flags.get("--out").ok_or("--out is required")?);
    let every = flags
        .get("--every")
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    if flags.contains_key("--every") && every == 0 {
        return Err("--every must be positive".into());
    }
    let bytes = fs::read(&input)?;
    let exp: Experiment = serde_json::from_slice(&bytes)?;
    exp.validate()?;
    if mode == "adjoint" && storage == "full" && exp.name == "marmousi" {
        return Err(
            "Marmousi must use treeverse storage; full history would need 4.16 GB per shot".into(),
        );
    }
    let nr = exp.receivers.len();
    let ns = exp.shots.len();
    let weights = if mode == "adjoint" {
        let path = PathBuf::from(flags.get("--data").ok_or("adjoint requires --data")?);
        npy::read_f64(&path, &[ns, exp.steps, nr])?
    } else {
        Vec::new()
    };
    if mode != "adjoint" && flags.contains_key("--data") {
        return Err("--data is only valid in adjoint mode".into());
    }
    if every == 0 && out.join("wavefield.npy").exists() {
        return Err(
            "existing wavefield recording requires --every or a fresh --out directory".into(),
        );
    }
    fs::create_dir_all(&out)?;
    let started = Instant::now();
    let computed = simulation::run(&exp, mode, &weights, every, storage, budget);
    println!("shot\tmode\tl2");
    for (shot, l2) in computed.shot_l2.iter().enumerate() {
        println!("{shot}\t{mode}\t{l2:.15e}");
    }
    let filename = match mode.as_str() {
        "forward" => "traces.npy",
        "born" => "born_data.npy",
        _ => "image.npy",
    };
    let shape = if mode == "adjoint" {
        vec![exp.nz, exp.nx]
    } else {
        vec![ns, exp.steps, nr]
    };
    npy::write_f64(&out.join(filename), &shape, &computed.array)?;
    if every > 0 {
        npy::write_f32(
            &out.join("wavefield.npy"),
            &[computed.recording_steps.len(), exp.nz, exp.nx],
            &computed.frames,
        )?;
    }
    for (shot, actions) in computed.actions.iter().enumerate() {
        // Compact action logs keep each committed file under the 5 MB evidence limit.
        fs::write(
            out.join(format!("actions-{shot}.json")),
            serde_json::to_vec(actions)?,
        )?;
    }
    let mut metadata: Value = serde_json::from_slice(&bytes)?;
    metadata.as_object_mut().unwrap().remove("background");
    metadata.as_object_mut().unwrap().remove("perturbation");
    let mut run = json!({"experiment": metadata, "experiment_file": input,
        "experiment_sha256": format!("{:x}", Sha256::digest(&bytes)),
        "arguments": flags, "precision": "float64", "parameter": "velocity",
        "inner_product": "unweighted Euclidean sums"});
    if every > 0 {
        let times = computed
            .recording_steps
            .iter()
            .map(|&s| s as f64 * exp.dt)
            .collect::<Vec<_>>();
        run["recording"] =
            json!({"every": every, "steps": computed.recording_steps, "times": times});
    }
    fs::write(out.join("run.json"), serde_json::to_vec_pretty(&run)?)?;
    let result = json!({"schema": "week5-born-rtm-v2", "experiment": exp.name,
        "mode": mode, "nx": exp.nx, "nz": exp.nz, "steps": exp.steps, "dt": exp.dt,
        "dx": exp.dx, "shots": ns, "receivers": nr, "implementation": "Rust; isolated Enzyme kernel",
        "seconds": started.elapsed().as_secs_f64(), "statistics": computed.statistics});
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
