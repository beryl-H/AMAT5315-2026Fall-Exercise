//! Accuracy comparison on one lap of a periodic Gaussian pulse.
//!
//! Initial condition: u(x,0) = exp(-(x - pi/2)^2 / (2 sigma^2)), sigma = 0.25,
//! periodic on [0, 2pi), n = 64, c = 1, nu = 0.002, integrated to t = 2pi
//! (exactly one lap) with:
//!   1. RK4, dt = 0.02,  Fourier derivatives
//!   2. RK4, dt = 0.02,  centred finite differences
//!   3. forward Euler, dt = 0.005, Fourier derivatives
//! Reference: the exact solution -- the Gaussian advected one full lap back
//! to x = pi/2 and diffused to sigma_t = sqrt(sigma^2 + 2 nu t), periodized.
//!
//! Usage: cargo run --bin run_accuracy -- evidence/line_accuracy.json
//! (run from `week4/`). Prints the maximum error of each run.

use advdiff::{advance, dft, idft, CentredFD, Euler, FourierAdvDiff, RK4};
use std::f64::consts::{PI, TAU};

const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.002;
const SIGMA: f64 = 0.25;
const X0: f64 = PI / 2.0;
const T_FINAL: f64 = TAU;

fn initial(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            (-(x - X0) * (x - X0) / (2.0 * SIGMA * SIGMA)).exp()
        })
        .collect()
}

/// Exact solution at time t: the pulse has advected a full number of laps,
/// its centre is back at X0 (mod 2pi), and diffusion widened it to
/// sigma_t = sqrt(sigma^2 + 2 nu t). Periodized over the box (images are
/// ~e^{-300}, kept for correctness at t = 0 where the test needs them).
fn exact(n: usize, t: f64) -> Vec<f64> {
    let sigma_t = (SIGMA * SIGMA + 2.0 * NU * t).sqrt();
    let amp = SIGMA / sigma_t;
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            let mut acc = 0.0;
            for p in -3i32..=3 {
                let xc = X0 + TAU * p as f64;
                acc += (-(x - xc) * (x - xc) / (2.0 * sigma_t * sigma_t)).exp();
            }
            amp * acc
        })
        .collect()
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::{exact, initial, max_err, N};

    #[test]
    fn exact_at_t_zero_is_the_initial_pulse() {
        // sigma_t = sigma and the pulse is effectively compact: the
        // periodized sum reduces to the single pulse.
        let u0 = initial(N);
        let e0 = exact(N, 0.0);
        for (a, b) in u0.iter().zip(&e0) {
            assert!((a - b).abs() < 1e-9, "{} vs {}", a, b);
        }
    }

    #[test]
    fn max_err_of_identical_profiles_is_zero() {
        let v = vec![0.3_f64; N];
        assert_eq!(max_err(&v, &v), 0.0);
    }
}

// Hand-rolled JSON helpers (duplicated across bins on purpose: bins do not
// share code and no JSON dependency is wanted).
fn json_floats(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.10e}", x)).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "evidence/line_accuracy.json".to_string());

    let xs: Vec<f64> = (0..N).map(|j| TAU * j as f64 / N as f64).collect();
    let u0 = initial(N);
    let u_hat0 = dft(&u0);
    let exact_final = exact(N, T_FINAL);

    let scheme_f = FourierAdvDiff { c: C, nu: NU, n: N };
    let scheme_fd = CentredFD { c: C, nu: NU, n: N };

    let runs: Vec<(&str, f64, Vec<f64>)> = vec![
        (
            "RK4, Fourier derivatives, dt = 0.02",
            0.02,
            idft(
                &advance(&RK4, |s| scheme_f.rate(s), 0.02, T_FINAL, &u_hat0)
                    .last()
                    .expect("non-empty")
                    .1,
            ),
        ),
        (
            "RK4, centred differences, dt = 0.02",
            0.02,
            advance(&RK4, |s| scheme_fd.rate(s), 0.02, T_FINAL, &u0)
                .last()
                .expect("non-empty")
                .1
                .clone(),
        ),
        (
            "forward Euler, Fourier derivatives, dt = 0.005",
            0.005,
            idft(
                &advance(&Euler, |s| scheme_f.rate(s), 0.005, T_FINAL, &u_hat0)
                    .last()
                    .expect("non-empty")
                    .1,
            ),
        ),
    ];

    let mut json = String::from("{\n");
    json.push_str(&format!(
        "\"n\": {}, \"c\": {}, \"nu\": {}, \"sigma\": {}, \"t_final\": {},\n",
        N, C, NU, SIGMA, T_FINAL
    ));
    json.push_str(&format!("\"x\": {},\n", json_floats(&xs)));
    json.push_str(&format!("\"exact\": {},\n", json_floats(&exact_final)));
    json.push_str("\"runs\": [\n");
    for (i, (label, dt, u)) in runs.iter().enumerate() {
        let err = max_err(u, &exact_final);
        println!("{}: max error = {:.6e}", label, err);
        let comma = if i + 1 < runs.len() { "," } else { "" };
        json.push_str(&format!(
            "{{\"label\": \"{}\", \"dt\": {}, \"max_err\": {:.10e}, \"u\": {}}}{}\n",
            label,
            dt,
            err,
            json_floats(u),
            comma
        ));
    }
    json.push_str("]\n}\n");

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).expect("create evidence directory");
    }
    std::fs::write(&out_path, json).expect("write accuracy json");
    println!("wrote {}", out_path);
}
