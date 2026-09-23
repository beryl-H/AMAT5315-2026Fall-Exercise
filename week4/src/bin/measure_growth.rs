//! Measures one-step growth factors |y1| of the three library integrators
//! on a grid of z = lambda*h (here h = 1) by actually advancing y' = lambda*y
//! one step from y0 = 1 with `advdiff`'s `Integrator::step`.
//!
//! Usage: cargo run --bin measure_growth -- evidence/line_growth.json
//! (run from `week4/`).

use advdiff::{Euler, Integrator, Midpoint, RK4};
use num_complex::Complex;

/// Advance y' = lam*y one step of h = 1 from y0 = 1 and return |y1|,
/// the measured growth factor per step of the given library integrator.
fn measured_growth<I: Integrator>(integrator: &I, lam: Complex<f64>) -> f64 {
    let y1 = integrator.step(
        |s: &[Complex<f64>]| s.iter().map(|v| lam * *v).collect(),
        1.0,
        &[Complex::new(1.0, 0.0)],
    );
    y1[0].norm()
}

#[cfg(test)]
mod tests {
    use super::measured_growth;
    use advdiff::{Euler, Midpoint, RK4};
    use num_complex::Complex;

    #[test]
    fn measured_one_step_growth_matches_stability_polynomial_on_real_axis() {
        // z = -1: Euler |1+z| = 0, midpoint |1+z+z^2/2| = 1/2,
        // RK4 |1+z+z^2/2+z^3/6+z^4/24| = |3/8|.
        let z = Complex::new(-1.0, 0.0);
        assert!(measured_growth(&Euler, z) < 1e-14);
        assert!((measured_growth(&Midpoint, z) - 0.5).abs() < 1e-14);
        let rk4_expected = (1.0_f64 - 1.0 + 0.5 - 1.0 / 6.0 + 1.0 / 24.0).abs();
        assert!((measured_growth(&RK4, z) - rk4_expected).abs() < 1e-14);
    }

    #[test]
    fn zero_eigenvalue_gives_unit_growth_for_all_integrators() {
        let z = Complex::new(0.0, 0.0);
        assert!((measured_growth(&Euler, z) - 1.0).abs() < 1e-14);
        assert!((measured_growth(&Midpoint, z) - 1.0).abs() < 1e-14);
        assert!((measured_growth(&RK4, z) - 1.0).abs() < 1e-14);
    }
}

fn json_floats(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.10e}", x)).collect();
    format!("[{}]", parts.join(","))
}

fn json_rows(rows: &[Vec<f64>]) -> String {
    let parts: Vec<String> = rows.iter().map(|r| json_floats(r)).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "evidence/line_growth.json".to_string());

    // Grid of z = lambda*h values covering the stability regions of all
    // three integrators (RK4 reaches Re z ~ -2.785, |Im z| ~ 2.828).
    let nx = 241usize;
    let ny = 241usize;
    let xs: Vec<f64> = (0..nx).map(|i| -3.6 + 5.0 * i as f64 / (nx - 1) as f64).collect();
    let ys: Vec<f64> = (0..ny).map(|j| -3.3 + 6.6 * j as f64 / (ny - 1) as f64).collect();

    let integrators: [(&str, &dyn Fn(Complex<f64>) -> f64); 3] = [
        ("euler", &|z| measured_growth(&Euler, z)),
        ("midpoint", &|z| measured_growth(&Midpoint, z)),
        ("rk4", &|z| measured_growth(&RK4, z)),
    ];

    let mut fields: Vec<(String, Vec<Vec<f64>>)> = Vec::new();
    for (name, growth) in &integrators {
        let rows: Vec<Vec<f64>> = ys.iter().map(|&y| {
            xs.iter()
                .map(|&x| growth(Complex::new(x, y)))
                .collect()
        }).collect();
        fields.push((name.to_string(), rows));
    }

    let mut json = String::from("{\n");
    json.push_str(&format!("\"x\": {},\n", json_floats(&xs)));
    json.push_str(&format!("\"y\": {},\n", json_floats(&ys)));
    json.push_str("\"growth\": {\n");
    for (i, (name, rows)) in fields.iter().enumerate() {
        let comma = if i + 1 < fields.len() { "," } else { "" };
        json.push_str(&format!("\"{}\": {}{}\n", name, json_rows(rows), comma));
    }
    json.push_str("}\n}\n");

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).expect("create evidence directory");
    }
    std::fs::write(&out_path, json).expect("write growth json");
    println!("wrote {}", out_path);
}
