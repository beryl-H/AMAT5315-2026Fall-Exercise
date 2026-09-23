//! Fluid binary: integrates the two-dimensional incompressible vorticity
//! equation from the Taylor-Green vortex with the integrator chosen by
//! --method (fluid.design.toml).
//!
//! Usage: fluid --n 32 --method rk4 --nu 0.001 --dt 0.01 --t-final 2.0
//!            --every 10 --out runs/tg
//! (run from `week4/`).

use advdiff::field::{idft2, taylor_green_spectral};
use advdiff::fluid::{energy, enstrophy, rate};
use advdiff::{advance, Euler, Midpoint, RK4};
use clap::Parser;
use std::f64::consts::TAU;

#[derive(Parser)]
#[command(name = "fluid", about = "2D incompressible vorticity equation solver")]
struct Args {
    /// grid points per axis
    #[arg(long)]
    n: usize,
    /// integrator: euler, midpoint, or rk4
    #[arg(long)]
    method: String,
    /// viscosity
    #[arg(long)]
    nu: f64,
    /// time step
    #[arg(long)]
    dt: f64,
    /// integration time
    #[arg(long = "t-final")]
    t_final: f64,
    /// steps between recorded frames; 0 records none
    #[arg(long)]
    every: usize,
    /// output folder
    #[arg(long)]
    out: String,
}

fn main() {
    let args = Args::parse();
    let n = args.n;

    let omega0 = taylor_green_spectral(n);

    let history = match args.method.as_str() {
        "euler" => advance(&Euler, |s| rate(n, args.nu, s), args.dt, args.t_final, &omega0),
        "midpoint" => advance(&Midpoint, |s| rate(n, args.nu, s), args.dt, args.t_final, &omega0),
        "rk4" => advance(&RK4, |s| rate(n, args.nu, s), args.dt, args.t_final, &omega0),
        other => {
            eprintln!("error: unknown method '{other}' (euler, midpoint, rk4)");
            std::process::exit(2);
        }
    };
    let steps = history.len();
    let (t_final, omega_final) = history.last().expect("non-empty history");

    // exact solution: the Taylor-Green nonlinear term vanishes identically,
    // so omega(x, y, t) = 2 sin(x) sin(y) exp(-2 nu t)
    let exact: Vec<f64> = (0..n * n)
        .map(|q| {
            let (i, j) = (q % n, q / n);
            let x = TAU * i as f64 / n as f64;
            let y = TAU * j as f64 / n as f64;
            2.0 * x.sin() * y.sin() * (-2.0 * args.nu * t_final).exp()
        })
        .collect();
    let u_phys = idft2(n, omega_final);
    let max_err = u_phys
        .iter()
        .zip(&exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let e_final = energy(n, omega_final);
    let z_final = enstrophy(n, omega_final);

    println!(
        "method\tdt\tsteps\tmax_error\tenergy\tenstrophy\n{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}",
        args.method, args.dt, steps, max_err, e_final, z_final
    );

    std::fs::create_dir_all(&args.out).expect("create output directory");
    std::fs::write(
        format!("{}/run.json", args.out),
        format!(
            "{{\"n\": {}, \"method\": \"{}\", \"nu\": {}, \"dt\": {}, \
             \"t_final\": {}, \"every\": {}, \"steps\": {}, \
             \"max_err\": {:.10e}, \"energy\": {:.10e}, \"enstrophy\": {:.10e}}}\n",
            n, args.method, args.nu, args.dt, args.t_final, args.every, steps,
            max_err, e_final, z_final
        ),
    )
    .expect("write run.json");

    let mut series = String::new();
    if args.every > 0 {
        for (idx, (t, omega)) in history.iter().enumerate() {
            if (idx + 1) % args.every != 0 && idx + 1 != steps {
                continue;
            }
            let e = energy(n, omega);
            let z = enstrophy(n, omega);
            let u = idft2(n, omega);
            let err = u
                .iter()
                .enumerate()
                .map(|(q, &v)| {
                    let (i, j) = (q % n, q / n);
                    let x = TAU * i as f64 / n as f64;
                    let y = TAU * j as f64 / n as f64;
                    let ex = 2.0 * x.sin() * y.sin() * (-2.0 * args.nu * t).exp();
                    (v - ex).abs()
                })
                .fold(0.0_f64, f64::max);
            series.push_str(&format!(
                "{{\"t\": {:.10e}, \"energy\": {:.10e}, \"enstrophy\": {:.10e}, \"max_err\": {:.10e}}}\n",
                t, e, z, err
            ));
        }
    }
    std::fs::write(format!("{}/series.jsonl", args.out), series).expect("write series.jsonl");

    let physical_json: Vec<String> = u_phys.iter().map(|v| format!("{v:.10e}")).collect();
    std::fs::write(
        format!("{}/final.json", args.out),
        format!(
            "{{\"t\": {}, \"u\": [{}], \"energy\": {:.10e}, \"enstrophy\": {:.10e}, \
             \"max_err\": {:.10e}}}\n",
            t_final,
            physical_json.join(","),
            e_final,
            z_final,
            max_err
        ),
    )
    .expect("write final.json");
}

