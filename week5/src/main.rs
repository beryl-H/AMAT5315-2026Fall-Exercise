use seismic::{Experiment, Result, advance, npy};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, path::PathBuf, time::Instant};

fn main() {
    if let Err(e) = run() { eprintln!("seismic: {e}"); std::process::exit(1); }
}

fn flags() -> Result<BTreeMap<String, String>> {
    let mut args = env::args().skip(1);
    let mut flags = BTreeMap::new();
    while let Some(key) = args.next() {
        if key == "--enzyme-smoke" { flags.insert(key, "true".into()); continue; }
        if key == "--help" { println!("seismic --experiment FILE --mode forward --out DIR [--every STRIDE]"); std::process::exit(0); }
        if !["--experiment", "--mode", "--out", "--every", "--data", "--storage", "--checkpoints"].contains(&key.as_str()) {
            return Err(format!("unknown argument {key}").into());
        }
        let value = args.next().ok_or_else(|| format!("missing value for {key}"))?;
        if flags.insert(key.clone(), value).is_some() { return Err(format!("duplicate flag {key}").into()); }
    }
    Ok(flags)
}

fn run() -> Result<()> {
    let flags = flags()?;
    if flags.contains_key("--enzyme-smoke") { println!("{}", json!(seismic::smoke())); return Ok(()); }
    let input = PathBuf::from(flags.get("--experiment").ok_or("--experiment is required")?);
    let mode = flags.get("--mode").ok_or("--mode is required")?;
    if mode != "forward" { return Err("only forward mode is implemented".into()); }
    let out = PathBuf::from(flags.get("--out").ok_or("--out is required")?);
    let every = flags.get("--every").map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
    if flags.contains_key("--every") && every == 0 { return Err("--every must be positive".into()); }
    let bytes = fs::read(&input)?;
    let exp: Experiment = serde_json::from_slice(&bytes)?;
    exp.validate()?;
    fs::create_dir_all(&out)?;
    let n = exp.cells();
    let nr = exp.receivers.len();
    let ns = exp.shots.len();
    let velocity = exp.velocity();
    let sigma = exp.sigma();
    let mut data = vec![0.0; ns * exp.steps * nr];
    let mut frames = Vec::<f32>::new();
    let mut recording_steps = Vec::<usize>::new();
    let started = Instant::now();
    println!("shot\tmode\tl2");
    for (shot, &position) in exp.shots.iter().enumerate() {
        let footprint = exp.footprint(position);
        let mut source = vec![0.0; n];
        let mut state = vec![0.0; 2 * n];
        let mut next = state.clone();
        for step in 0..exp.steps {
            exp.source(step, &footprint, &mut source);
            advance(&exp, &state, &velocity, &sigma, &source, &mut next);
            std::mem::swap(&mut state, &mut next);
            for (r, &[x, z]) in exp.receivers.iter().enumerate() {
                data[(shot * exp.steps + step) * nr + r] = state[n + z * exp.nx + x];
            }
            if shot == 0 && every > 0 && (step + 1) % every == 0 {
                recording_steps.push(step + 1);
                frames.extend(state[n..].iter().map(|&v| v as f32));
            }
        }
        let slice = &data[shot * exp.steps * nr..(shot + 1) * exp.steps * nr];
        println!("{shot}\t{mode}\t{:.15e}", slice.iter().map(|x| x*x).sum::<f64>().sqrt());
    }
    npy::write_f64(&out.join("traces.npy"), &[ns, exp.steps, nr], &data)?;
    if every > 0 { npy::write_f32(&out.join("wavefield.npy"), &[recording_steps.len(), exp.nz, exp.nx], &frames)?; }
    let mut metadata: Value = serde_json::from_slice(&bytes)?;
    metadata.as_object_mut().unwrap().remove("background");
    metadata.as_object_mut().unwrap().remove("perturbation");
    let mut run = json!({"experiment": metadata, "experiment_file": input,
        "experiment_sha256": format!("{:x}", Sha256::digest(&bytes)),
        "arguments": flags, "precision": "float64", "parameter": "velocity",
        "inner_product": "unweighted Euclidean sums"});
    if every > 0 {
        let times = recording_steps.iter().map(|&s| s as f64 * exp.dt).collect::<Vec<_>>();
        run["recording"] = json!({"every": every, "steps": recording_steps, "times": times});
    }
    fs::write(out.join("run.json"), serde_json::to_vec_pretty(&run)?)?;
    let result = json!({"schema": "week5-born-rtm-v2", "experiment": exp.name,
        "mode": mode, "nx": exp.nx, "nz": exp.nz, "steps": exp.steps, "dt": exp.dt,
        "dx": exp.dx, "shots": ns, "receivers": nr, "implementation": "Rust; isolated Enzyme kernel",
        "seconds": started.elapsed().as_secs_f64(), "statistics": null});
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
