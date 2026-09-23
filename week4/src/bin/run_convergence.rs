//! Convergence sweep on one lap-free interval: Gaussian pulse
//! (sigma = 0.35, centred at x = pi/2, n = 64, c = 1, nu = 0.05) evolved to
//! t = 1 with Fourier derivatives by forward Euler, explicit midpoint,
//! classical RK4, and equal-weight RK4, each at dt = 0.02, 0.01, 0.005,
//! 0.0025. Reference: the exact solution of the semi-discrete system,
//! u_hat_k(T) = u_hat_k(0) exp(lambda_k T) with the scheme's own Nyquist rule
//! (no travel at k = n/2), so the measured error is purely temporal.
//!
//! Usage: cargo run --bin run_convergence -- evidence/line_convergence.json
//! (run from `week4/`). Prints the max error of every method/dt pair.

use advdiff::{advance, dft, idft, Euler, FourierAdvDiff, Midpoint, RK4, RK4Equal};
use num_complex::Complex;
use std::f64::consts::{PI, TAU};

const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.05;
const SIGMA: f64 = 0.35;
const X0: f64 = PI / 2.0;
const T_FINAL: f64 = 1.0;
const DTS: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];

fn initial(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            (-(x - X0) * (x - X0) / (2.0 * SIGMA * SIGMA)).exp()
        })
        .collect()
}

/// Exact solution operator of the semi-discrete Fourier system after time t,
/// mirroring the scheme's Nyquist rule (no travel at index n/2 for even n).
fn exact_semi(
    scheme: &FourierAdvDiff,
    u_hat0: &[Complex<f64>],
    t: f64,
) -> Vec<Complex<f64>> {
    u_hat0
        .iter()
        .enumerate()
        .map(|(j, &u)| {
            let k = scheme.wavenumber(j);
            let nyquist = scheme.n % 2 == 0 && j == scheme.n / 2;
            let lambda = Complex::new(
                -scheme.nu * k * k,
                if nyquist { 0.0 } else { -scheme.c * k },
            );
            u * (lambda * t).exp()
        })
        .collect()
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::exact_semi;
    use advdiff::{advance, FourierAdvDiff, RK4};
    use num_complex::Complex;

    #[test]
    fn exact_semi_matches_fine_rk4_evolution_of_a_single_mode() {
        let scheme = FourierAdvDiff { c: 1.0, nu: 0.05, n: 16 };
        let mut u_hat0 = vec![Complex::new(0.0, 0.0); 16];
        u_hat0[2] = Complex::new(1.0, -0.5);
        let reference = exact_semi(&scheme, &u_hat0, 1.0);
        // RK4 with a very fine step must reproduce the exponential reference.
        let fine = advance(&RK4, |s| scheme.rate(s), 0.001, 1.0, &u_hat0);
        let final_state = fine.last().expect("non-empty").1.clone();
        assert!(
            (final_state[2] - reference[2]).norm() < 1e-10,
            "got {}, want {}",
            final_state[2],
            reference[2]
        );
    }

    #[test]
    fn exact_semi_damps_nyquist_without_travel() {
        let scheme = FourierAdvDiff { c: 1.0, nu: 0.05, n: 16 };
        let mut u_hat0 = vec![Complex::new(0.0, 0.0); 16];
        u_hat0[8] = Complex::new(1.5, 2.0); // Nyquist index of n = 16
        let reference = exact_semi(&scheme, &u_hat0, 2.0);
        let factor = reference[8] / u_hat0[8];
        assert_eq!(factor.im, 0.0, "Nyquist mode must not travel");
        let expected = (-scheme.nu * 64.0 * 2.0).exp();
        assert!((factor.re - expected).abs() < 1e-12);
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
        .unwrap_or_else(|| "evidence/line_convergence.json".to_string());

    let u0 = initial(N);
    let u_hat0 = dft(&u0);
    let scheme = FourierAdvDiff { c: C, nu: NU, n: N };
    let reference_phys = idft(&exact_semi(&scheme, &u_hat0, T_FINAL));

    let methods: [(&str, &dyn Fn(&[Complex<f64>], f64) -> Vec<Complex<f64>>); 4] = [
        (
            "forward Euler",
            &|s: &[Complex<f64>], dt: f64| {
                advance(&Euler, |y| scheme.rate(y), dt, T_FINAL, s)
                    .last()
                    .expect("non-empty")
                    .1
                    .clone()
            },
        ),
        (
            "explicit midpoint",
            &|s: &[Complex<f64>], dt: f64| {
                advance(&Midpoint, |y| scheme.rate(y), dt, T_FINAL, s)
                    .last()
                    .expect("non-empty")
                    .1
                    .clone()
            },
        ),
        (
            "classical RK4",
            &|s: &[Complex<f64>], dt: f64| {
                advance(&RK4, |y| scheme.rate(y), dt, T_FINAL, s)
                    .last()
                    .expect("non-empty")
                    .1
                    .clone()
            },
        ),
        (
            "equal-weight RK4",
            &|s: &[Complex<f64>], dt: f64| {
                advance(&RK4Equal, |y| scheme.rate(y), dt, T_FINAL, s)
                    .last()
                    .expect("non-empty")
                    .1
                    .clone()
            },
        ),
    ];

    let mut json = String::from("{\n");
    json.push_str(&format!(
        "\"n\": {}, \"c\": {}, \"nu\": {}, \"sigma\": {}, \"t_final\": {},\n",
        N, C, NU, SIGMA, T_FINAL
    ));
    json.push_str(&format!("\"dts\": {},\n", json_floats(&DTS)));
    json.push_str("\"methods\": [\n");
    for (i, (label, run)) in methods.iter().enumerate() {
        let errors: Vec<f64> = DTS
            .iter()
            .map(|&dt| {
                let u_hat_final = run(&u_hat0, dt);
                max_err(&idft(&u_hat_final), &reference_phys)
            })
            .collect();
        for (dt, err) in DTS.iter().zip(&errors) {
            println!("{:20} dt = {:8.4}: max error = {:.6e}", label, dt, err);
        }
        let comma = if i + 1 < methods.len() { "," } else { "" };
        json.push_str(&format!(
            "{{\"label\": \"{}\", \"errors\": {}}}{}\n",
            label,
            json_floats(&errors),
            comma
        ));
    }
    json.push_str("]\n}\n");
    let _ = reference_phys; // already consumed via errors above

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).expect("create evidence directory");
    }
    std::fs::write(&out_path, json).expect("write convergence json");
    println!("wrote {}", out_path);
}
