//! Integrates the Gaussian-pulse initial condition of the advection-diffusion
//! line to t = 6 with the library RK4 and records space-time snapshots.
//!
//! Initial condition: u(x, 0) = exp(-(x - pi/2)^2 / (2 sigma^2)), sigma = 0.35,
//! on the periodic grid x_j = 2 pi j / n with n = 64, nu = 0.05, c = 1.
//! Integration: library RK4 on the Fourier pseudo-spectral rate function,
//! full steps of the requested dt plus one shorter final step to land exactly
//! on t = 6. Snapshots of the physical field are recorded after every step.
//!
//! Usage: cargo run --bin run_gaussian -- evidence/line_evolution.json
//! (run from `week4/`).

use advdiff::{dft, FourierAdvDiff, idft, RK4, Integrator};
use num_complex::Complex;
use std::f64::consts::{PI, TAU};

/// Step the spectral state from t = 0 to t_final with RK4, using full steps
/// of `dt` plus one shorter final step, recording (t, u_hat) after each step.
pub fn advance_spectral(
    scheme: &FourierAdvDiff,
    u_hat0: &[Complex<f64>],
    dt: f64,
    t_final: f64,
) -> Vec<(f64, Vec<Complex<f64>>)> {
    let rk4 = RK4;
    let mut u_hat = u_hat0.to_vec();
    let mut t = 0.0;
    let mut history = Vec::new();
    while t < t_final - 1e-12 {
        let step = dt.min(t_final - t);
        u_hat = rk4.step(|s| scheme.rate(s), step, &u_hat);
        t += step;
        history.push((t, u_hat.clone()));
    }
    history
}

#[cfg(test)]
mod tests {
    use super::advance_spectral;
    use advdiff::FourierAdvDiff;
    use num_complex::Complex;

    fn scheme() -> FourierAdvDiff {
        FourierAdvDiff { c: 1.0, nu: 0.05, n: 16 }
    }

    fn single_mode(n: usize, k: usize) -> Vec<Complex<f64>> {
        let mut u_hat = vec![Complex::new(0.0, 0.0); n];
        u_hat[k] = Complex::new(1.0, 0.0);
        u_hat
    }

    #[test]
    fn runner_lands_exactly_on_t_final_with_full_plus_partial_steps() {
        // dt = 0.3, T = 1: three full steps and one partial step of 0.1.
        // Intermediate times are compared within float epsilon (0.3+0.3+0.3
        // is not bit-exact 0.9); the final time must be exact.
        let s = scheme();
        let history = advance_spectral(&s, &single_mode(16, 2), 0.3, 1.0);
        let times: Vec<f64> = history.iter().map(|(t, _)| *t).collect();
        let expected = [0.3, 0.6, 0.9, 1.0];
        assert_eq!(times.len(), expected.len());
        for (i, &t) in times.iter().enumerate() {
            assert!((t - expected[i]).abs() < 1e-12, "times[{}] = {}", i, t);
        }
        assert_eq!(*times.last().expect("non-empty"), 1.0);
    }

    #[test]
    fn runner_matches_exact_single_mode_evolution_to_t_final() {
        // Exact: u_hat_k(T) = u_hat_k(0) * exp((-i c k - nu k^2) T).
        let s = scheme();
        let k = 2.0_f64;
        let history = advance_spectral(&s, &single_mode(16, 2), 0.3, 1.0);
        let (t, final_state) = history.last().expect("non-empty history");
        assert_eq!(*t, 1.0);
        let lambda = Complex::new(-s.nu * k * k, -s.c * k);
        let expected = (lambda * t).exp();
        // |lambda| dt = 0.603: accumulated RK4 error over the four steps is
        // ~1.7e-3; 1e-2 still separates RK4 from Midpoint (~0.1) and Euler
        // (~0.5) on this data.
        assert!(
            (final_state[2] - expected).norm() < 1e-2,
            "got {}, want {}",
            final_state[2],
            expected
        );
    }
}

// Small hand-rolled JSON helpers (duplicated from the measure_growth bin on
// purpose: bins do not share library code and no JSON dependency is wanted).
fn json_floats(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.10e}", x)).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "evidence/line_evolution.json".to_string());

    let n = 64usize;
    let c = 1.0_f64;
    let nu = 0.05_f64;
    let sigma = 0.35_f64;
    let x0 = PI / 2.0;
    let t_final = 6.0_f64;
    let dts = [0.045_f64, 0.056_f64];

    let xs: Vec<f64> = (0..n).map(|j| TAU * j as f64 / n as f64).collect();

    // Gaussian pulse, effectively compact on the periodic box (the tails at
    // distance ~pi are e^{-40}, so no visible wrap-around seam at t = 0).
    let u0: Vec<f64> = xs
        .iter()
        .map(|&x| (-(x - x0) * (x - x0) / (2.0 * sigma * sigma)).exp())
        .collect();

    let scheme = FourierAdvDiff { c, nu, n };
    let u_hat0 = dft(&u0);

    let mut json = String::from("{\n");
    json.push_str(&format!("\"n\": {}, \"c\": {}, \"nu\": {}, ", n, c, nu));
    json.push_str(&format!("\"sigma\": {}, \"x0\": {}, \"t_final\": {},\n", sigma, x0, t_final));
    json.push_str(&format!("\"x\": {},\n", json_floats(&xs)));
    json.push_str("\"runs\": [\n");
    for (i, &dt) in dts.iter().enumerate() {
        let history = advance_spectral(&scheme, &u_hat0, dt, t_final);
        let times: Vec<f64> = history.iter().map(|(t, _)| *t).collect();
        let rows: Vec<Vec<f64>> = history
            .iter()
            .map(|(_, u_hat)| idft(u_hat))
            .collect();
        let rows_json: Vec<String> = rows.iter().map(|r| json_floats(r)).collect();
        let comma = if i + 1 < dts.len() { "," } else { "" };
        json.push_str(&format!(
            "{{\"dt\": {}, \"steps\": {}, \"times\": {}, \"u\": [{}]}}{}\n",
            dt,
            times.len(),
            json_floats(&times),
            rows_json.join(","),
            comma
        ));
        let (_, last) = history.last().expect("non-empty history");
        let max_abs = idft(last).iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        println!("dt = {}: {} steps, final t = {:.6}, max |u| = {:.6}", dt, times.len(), times[times.len() - 1], max_abs);
    }
    json.push_str("]\n}\n");

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).expect("create evidence directory");
    }
    std::fs::write(&out_path, json).expect("write evolution json");
    println!("wrote {}", out_path);
}
