//! Field initial-condition binary (field.design.toml): subcommands
//! `taylor-green` and `random` write one JSON object
//! {case, n, seed, k_band, u, v} to stdout; fluid reads it on stdin.
//!
//! Usage: field taylor-green --n 64 --nu 0.1 --t 1
//!        field random --n 128 --seed 2026 --k-min 2 --k-max 6

use advdiff::field::{dealias_band, idft2, random_vorticity, taylor_green_velocity};
use advdiff::fluid::{stream_function, velocity};
use clap::{Parser, Subcommand};
use serde_json::json;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Taylor-Green velocity field, exact solution of the free decay
    TaylorGreen {
        /// grid points per side
        #[arg(long)]
        n: usize,
        /// time of the exact solution; default 0, the initial field
        #[arg(long)]
        t: Option<f64>,
        /// viscosity of the decay; required if t > 0
        #[arg(long)]
        nu: Option<f64>,
    },
    /// Random vorticity modes of equal amplitude, E(0) = 1/2
    Random {
        /// grid points per side
        #[arg(long)]
        n: usize,
        /// seed of the phases, uniform in [0, 2*pi)
        #[arg(long)]
        seed: u64,
        /// lowest |k| kept, inclusive
        #[arg(long = "k-min")]
        k_min: f64,
        /// highest |k| kept, inclusive
        #[arg(long = "k-max")]
        k_max: f64,
    },
}

/// The `taylor-green` case: velocity on the n*n grid at time t. `nu` is
/// required when t > 0 (the decay needs a viscosity).
fn taylor_green_output(n: usize, t: f64, nu: Option<f64>) -> Result<String, String> {
    if t > 0.0 && nu.is_none() {
        return Err("nu is required when t > 0".to_string());
    }
    let nu = nu.unwrap_or(0.0);
    let (u, v) = taylor_green_velocity(n, t, nu);
    let band = dealias_band(n);
    Ok(json!({
        "case": "taylor-green",
        "n": n,
        "seed": 0,
        "k_band": [0, band],
        "u": u,
        "v": v,
    })
    .to_string())
}

/// The `random` case: vorticity modes of equal amplitude in the annulus
/// k-min <= |k| <= k-max with seeded phases, E(0) = 1/2; the delivered
/// velocity comes from the stream function (pairing A).
fn random_output(n: usize, seed: u64, k_min: f64, k_max: f64) -> Result<String, String> {
    if !(0.0 <= k_min && k_min <= k_max) {
        return Err("k range must satisfy 0 <= k-min <= k-max".to_string());
    }
    let omega_hat = random_vorticity(n, seed, k_min, k_max);
    let psi = stream_function(n, &omega_hat);
    let (u_x, u_y) = velocity(n, &psi);
    Ok(json!({
        "case": "random",
        "n": n,
        "seed": seed,
        "k_band": [k_min, k_max],
        "u": idft2(n, &u_x),
        "v": idft2(n, &u_y),
    })
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::{random_output, taylor_green_output};

    #[test]
    fn taylor_green_output_matches_the_contract() {
        let s = taylor_green_output(8, 0.0, None).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["case"], "taylor-green");
        assert_eq!(v["n"], 8);
        assert_eq!(v["seed"], 0);
        assert_eq!(v["k_band"], serde_json::json!([0, 2]));
        // u = cos(x) sin(y) at grid point (i=1, j=1): x = y = pi/4 -> 0.5
        let u = v["u"].as_array().unwrap();
        assert!((u[1 * 8 + 1].as_f64().unwrap() - 0.5).abs() < 1e-12);
        let vv = v["v"].as_array().unwrap();
        assert!((vv[1 * 8 + 1].as_f64().unwrap() + 0.5).abs() < 1e-12);
    }

    #[test]
    fn nu_is_required_when_t_is_positive() {
        assert!(taylor_green_output(8, 0.0, None).is_ok());
        assert!(taylor_green_output(8, 1.0, None).is_err());
        assert!(taylor_green_output(8, 1.0, Some(0.1)).is_ok());
    }

    #[test]
    fn random_output_delivers_half_unit_energy() {
        let s = random_output(32, 7, 2.0, 6.0).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["case"], "random");
        assert_eq!(v["seed"], 7);
        assert_eq!(v["k_band"], serde_json::json!([2.0, 6.0]));
        let u: Vec<f64> = v["u"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let vv: Vec<f64> = v["v"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let omega = advdiff::fluid::vorticity_from_velocity(32, &u, &vv);
        assert!((advdiff::fluid::energy(32, &omega) - 0.5).abs() < 1e-12);
    }
}

fn main() {
    let cli = Cli::parse();
    let out = match &cli.command {
        Command::TaylorGreen { n, t, nu } => taylor_green_output(*n, t.unwrap_or(0.0), *nu),
        Command::Random { n, seed, k_min, k_max } => {
            random_output(*n, *seed, *k_min, *k_max)
        }
    };
    match out {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
