//! Fluid binary (fluid.design.toml): reads the field JSON from stdin,
//! integrates the two-dimensional incompressible vorticity equation
//! omega_t + u . grad(omega) = nu laplacian(omega) with the chosen method,
//! and records snapshots.
//!
//! Usage: field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01
//!            --t-end 1 --every 0.1 --out artifacts/taylor-green

use advdiff::field::{dealias, idft2};
use advdiff::fluid::vorticity_from_velocity;
use advdiff::fluid::{energy, enstrophy, rate, stream_function, velocity};
use advdiff::{Euler, Integrator, Midpoint, RK4};
use clap::Parser;
use num_complex::Complex;
use serde_json::{json, Value};

#[derive(Parser)]
#[command(name = "fluid", about = "2D incompressible vorticity equation solver")]
struct Args {
    /// time integrator: euler, rk2, or rk4
    #[arg(long)]
    method: String,
    /// kinematic viscosity
    #[arg(long)]
    nu: f64,
    /// time step
    #[arg(long)]
    dt: f64,
    /// final integration time
    #[arg(long = "t-end")]
    t_end: f64,
    /// time between snapshots
    #[arg(long)]
    every: f64,
    /// output folder
    #[arg(long)]
    out: String,
}

/// Field JSON piped from `field`: {case, n, seed, k_band, u, v}.
struct FieldInput {
    case: String,
    n: usize,
    seed: i64,
    k_band: Value,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn parse_field_input(s: &str) -> Result<FieldInput, String> {
    let v: Value = serde_json::from_str(s).map_err(|e| format!("invalid field JSON: {e}"))?;
    let n = v["n"]
        .as_u64()
        .ok_or("field JSON: missing integer n")? as usize;
    let len = n * n;
    let u = v["u"]
        .as_array()
        .ok_or("field JSON: missing u array")?
        .iter()
        .map(|x| x.as_f64().ok_or("field JSON: non-finite u entry"))
        .collect::<Result<Vec<_>, _>>()?;
    let v_arr = v["v"]
        .as_array()
        .ok_or("field JSON: missing v array")?
        .iter()
        .map(|x| x.as_f64().ok_or("field JSON: non-finite v entry"))
        .collect::<Result<Vec<_>, _>>()?;
    if u.len() != len || v_arr.len() != len {
        return Err(format!("field JSON: u, v must have n*n = {len} entries"));
    }
    Ok(FieldInput {
        case: v["case"].as_str().unwrap_or("unknown").to_string(),
        n,
        seed: v["seed"].as_i64().unwrap_or(0),
        k_band: v["k_band"].clone(),
        u,
        v: v_arr,
    })
}

/// Snapshot step indices: 0 and every `every_steps` steps up to n_steps.
fn snapshot_schedule(n_steps: usize, every_steps: usize) -> Vec<usize> {
    (0..=n_steps).step_by(every_steps.max(1)).collect()
}

/// Physical u, v, omega from spectral vorticity (pairing A stream function).
fn physical_fields(n: usize, omega_hat: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let psi = stream_function(n, omega_hat);
    let (u_x, u_y) = velocity(n, &psi);
    (idft2(n, &u_x), idft2(n, &u_y), idft2(n, omega_hat))
}

/// One integration step with the chosen method.
fn step(
    method: &str,
    n: usize,
    nu: f64,
    dt: f64,
    omega: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, String> {
    match method {
        "euler" => Ok(Euler.step(|s| rate(n, nu, s), dt, omega)),
        "rk2" => Ok(Midpoint.step(|s| rate(n, nu, s), dt, omega)),
        "rk4" => Ok(RK4.step(|s| rate(n, nu, s), dt, omega)),
        other => Err(format!(
            "unknown method '{other}' (euler, rk2, rk4)"
        )),
    }
}

fn array6(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let args = Args::parse();
    if args.dt <= 0.0 {
        eprintln!("error: dt must be positive");
        std::process::exit(2);
    }
    if args.every <= 0.0 {
        eprintln!("error: every must be positive");
        std::process::exit(2);
    }
    let snap_every = (args.every / args.dt).round() as usize;
    if snap_every == 0 {
        eprintln!("error: every is smaller than dt");
        std::process::exit(2);
    }

    let mut input = String::new();
    use std::io::Read;
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read field from stdin");
    let field = match parse_field_input(&input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let n = field.n;

    // vorticity from the piped velocity, itself under the two-thirds rule
    let omega0 = dealias(n, &vorticity_from_velocity(n, &field.u, &field.v));

    let n_steps = (args.t_end / args.dt).round() as usize;
    let mut omega = omega0.clone();
    let mut snapshots: Vec<(f64, usize, Vec<Complex<f64>>)> = vec![(0.0, 0, omega.clone())];
    let schedule = snapshot_schedule(n_steps, snap_every);
    let mut next_snap = 1; // schedule[0] is step 0, already stored
    let mut stop_line: Option<String> = None;

    let mut stdout_lines = String::from("t\tE\tZ\n");
    for k in 1..=n_steps {
        omega = match step(args.method.as_str(), n, args.nu, args.dt, &omega) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        };
        let t = k as f64 * args.dt;
        let e = energy(n, &omega);
        if !e.is_finite() {
            // the run stops at the first non-finite energy, prints that
            // line last, and exits 1
            let z = enstrophy(n, &omega);
            stop_line = Some(format!("{t}\t{e:.10e}\t{z:.10e}"));
            break;
        }
        if next_snap < schedule.len() && schedule[next_snap] == k {
            snapshots.push((t, k, omega.clone()));
            next_snap += 1;
        }
    }
    for (t, _, omega) in &snapshots {
        let e = energy(n, omega);
        let z = enstrophy(n, omega);
        stdout_lines.push_str(&format!("{t}\t{e:.10e}\t{z:.10e}\n"));
    }
    if let Some(line) = &stop_line {
        stdout_lines.push_str(line);
        stdout_lines.push('\n');
    }
    print!("{stdout_lines}");

    std::fs::create_dir_all(&args.out).expect("create output directory");
    std::fs::write(
        format!("{}/run.json", args.out),
        json!({
            "case": field.case,
            "n": field.n,
            "seed": field.seed,
            "k_band": field.k_band,
            "method": args.method,
            "nu": args.nu,
            "dt": args.dt,
            "t_end": args.t_end,
            "snapshot_every": snap_every,
        })
        .to_string(),
    )
    .expect("write run.json");

    let mut fields = String::new();
    for (t, step, omega) in &snapshots {
        let (u, v, w) = physical_fields(n, omega);
        fields.push_str(&format!(
            "{{\"t\": {:.6}, \"step\": {}, \"u\": {}, \"v\": {}, \"omega\": {}}}\n",
            t,
            step,
            array6(&u),
            array6(&v),
            array6(&w)
        ));
    }
    std::fs::write(format!("{}/fields.jsonl", args.out), fields).expect("write fields.jsonl");

    if stop_line.is_some() {
        std::process::exit(1);
    }
}
