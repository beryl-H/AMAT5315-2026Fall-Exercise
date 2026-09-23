//! Field verification binary: linear 2D advection-diffusion of a single
//! Hermitian Fourier mode (field.design.toml).
//!
//! The initial condition is the Hermitian mode pair +-(kx, ky) with amplitude
//! 1 -- the physical field 2 cos(kx x + ky y) -- and the exact solution is the
//! same pair scaled by exp((-i (cx kx + cy ky) - nu (kx^2 + ky^2)) t), with
//! the per-axis no-travel rule of the linear rate (moot in band). The max
//! abs error is measured on the spectral state.
//!
//! Usage: field --n 32 --kx 1 --ky 2 --cx 1 --cy 1 --method rk4 --nu 0.05
//!            --t-final 1.0 --dt 0.01 --out runs/field-single-mode
//! (run from `week4/`).

use advdiff::field::{
    advdiff_exponential, advdiff_rate, dealias_band, hermitian_mode, idft2,
};
use advdiff::{advance, Euler, Midpoint, RK4};
use clap::Parser;
use num_complex::Complex;

/// Exact solution of the linear advection-diffusion equation for the
/// Hermitian single mode at time t.
fn exact_solution(
    n: usize,
    kx: usize,
    ky: usize,
    cx: f64,
    cy: f64,
    nu: f64,
    t: f64,
) -> Vec<Complex<f64>> {
    // the principal mode travels with exp(lambda t); its Hermitian partner
    // (wavenumbers -kx, -ky) travels with the conjugate exp(conj(lambda) t)
    let factor = advdiff_exponential(n, kx, ky, cx, cy, nu, t);
    let mut u = hermitian_mode(n, kx, ky, Complex::new(1.0, 0.0));
    let (mx, my) = ((n - kx) % n, (n - ky) % n);
    u[ky * n + kx] *= factor;
    if (mx, my) != (kx, ky) {
        u[my * n + mx] *= factor.conj();
    }
    u
}

#[cfg(test)]
mod tests {
    use super::exact_solution;
    use advdiff::field::{advdiff_exponential, hermitian_mode};
    use num_complex::Complex;

    #[test]
    fn exact_solution_is_the_initial_mode_times_the_exponential() {
        let n = 32;
        let u0 = exact_solution(n, 1, 2, 1.0, 1.0, 0.05, 0.0);
        // at t = 0 the exact solution is the initial Hermitian mode
        assert_eq!(u0, hermitian_mode(n, 1, 2, Complex::new(1.0, 0.0)));
        // at t = 1 the principal mode carries the exponential factor and the
        // Hermitian partner (wavenumbers -kx, -ky) carries its conjugate
        let u1 = exact_solution(n, 1, 2, 1.0, 1.0, 0.05, 1.0);
        let factor = advdiff_exponential(n, 1, 2, 1.0, 1.0, 0.05, 1.0);
        let q = 2 * n + 1; // [ky*n + kx]
        assert!((u1[q] - u0[q] * factor).norm() < 1e-14);
        let p = (n - 2) * n + (n - 1);
        assert!(
            (u1[p] - u0[p] * factor.conj()).norm() < 1e-14,
            "partner: got {}, want {}",
            u1[p],
            u0[p] * factor.conj()
        );
    }
}

#[derive(Parser)]
#[command(name = "field", about = "single-mode linear 2D advection-diffusion verification")]
struct Args {
    /// grid points per axis
    #[arg(long)]
    n: usize,
    /// x wavenumber of the initial condition (two-thirds band)
    #[arg(long)]
    kx: usize,
    /// y wavenumber of the initial condition (two-thirds band)
    #[arg(long)]
    ky: usize,
    /// advection velocity in x
    #[arg(long)]
    cx: f64,
    /// advection velocity in y
    #[arg(long)]
    cy: f64,
    /// integrator: euler, midpoint, or rk4
    #[arg(long)]
    method: String,
    /// viscosity
    #[arg(long)]
    nu: f64,
    /// integration time
    #[arg(long = "t-final")]
    t_final: f64,
    /// time step
    #[arg(long)]
    dt: f64,
    /// output folder
    #[arg(long)]
    out: String,
}

// Hand-rolled JSON helpers (duplicated across bins on purpose: bins do not
// share code and no JSON dependency is wanted).
fn max_spectral_err(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn main() {
    let args = Args::parse();
    let band = dealias_band(args.n);
    if args.kx > band || args.ky > band {
        eprintln!(
            "error: kx, ky must satisfy the two-thirds rule (|k| <= {band})"
        );
        std::process::exit(2);
    }

    let u_hat0 = hermitian_mode(args.n, args.kx, args.ky, Complex::new(1.0, 0.0));
    let reference = exact_solution(args.n, args.kx, args.ky, args.cx, args.cy, args.nu, args.t_final);

    let history = match args.method.as_str() {
        "euler" => advance(&Euler, |s| advdiff_rate(args.n, args.cx, args.cy, args.nu, s), args.dt, args.t_final, &u_hat0),
        "midpoint" => advance(&Midpoint, |s| advdiff_rate(args.n, args.cx, args.cy, args.nu, s), args.dt, args.t_final, &u_hat0),
        "rk4" => advance(&RK4, |s| advdiff_rate(args.n, args.cx, args.cy, args.nu, s), args.dt, args.t_final, &u_hat0),
        other => {
            eprintln!("error: unknown method '{other}' (euler, midpoint, rk4)");
            std::process::exit(2);
        }
    };
    let (t_final, final_state) = history.last().expect("non-empty history");
    let steps = history.len();
    let err = max_spectral_err(&final_state, &reference);

    println!(
        "method\tdt\tsteps\tmax_error\n{}\t{}\t{}\t{:.6e}",
        args.method, args.dt, steps, err
    );

    std::fs::create_dir_all(&args.out).expect("create output directory");
    std::fs::write(
        format!("{}/run.json", args.out),
        format!(
            "{{\"n\": {}, \"kx\": {}, \"ky\": {}, \"cx\": {}, \"cy\": {}, \
             \"method\": \"{}\", \"nu\": {}, \"t_final\": {}, \"dt\": {}, \
             \"steps\": {}, \"max_err\": {:.10e}}}\n",
            args.n, args.kx, args.ky, args.cx, args.cy, args.method, args.nu,
            args.t_final, args.dt, steps, err
        ),
    )
    .expect("write run.json");

    let physical = idft2(args.n, &final_state);
    let physical_json: Vec<String> = physical.iter().map(|v| format!("{v:.10e}")).collect();
    std::fs::write(
        format!("{}/final.json", args.out),
        format!(
            "{{\"t\": {}, \"u\": [{}], \"max_err\": {:.10e}}}\n",
            t_final,
            physical_json.join(","),
            err
        ),
    )
    .expect("write final.json");
}
