//! Derivative verification on the periodic [0, 2pi)^2 grid for
//! g(x, y) = sin(3x) cos(2y).
//!
//! Fourier pseudospectral derivatives (advdiff library, n = 32, in-band mode
//! so they are exact to round-off) against the analytic derivatives, and
//! second-order centred finite differences with periodic wrapping on
//! n = 32 and n = 64. All fields are computed in full f64 precision; the
//! table prints maximum absolute errors and the finite-difference
//! refinement ratio (approximately 4 for second-order stencils).
//!
//! Usage: cargo run --release --bin derivatives_check
//! (run from `week4/`).

use advdiff::field::{dft2, deriv_x, deriv_y, idft2, laplacian};
use std::f64::consts::TAU;

fn g(x: f64, y: f64) -> f64 {
    (3.0 * x).sin() * (2.0 * y).cos()
}
fn gx(x: f64, y: f64) -> f64 {
    3.0 * (3.0 * x).cos() * (2.0 * y).cos()
}
fn gxx(x: f64, y: f64) -> f64 {
    -9.0 * (3.0 * x).sin() * (2.0 * y).cos()
}
fn gxy(x: f64, y: f64) -> f64 {
    -6.0 * (3.0 * x).cos() * (2.0 * y).sin()
}
fn glap(x: f64, y: f64) -> f64 {
    -13.0 * (3.0 * x).sin() * (2.0 * y).cos()
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max)
}

/// Analytic derivatives sampled on the n*n grid, in the order
/// (d/dx, d2/dx2, d2/dxdy, laplacian).
fn analytic(n: usize) -> [Vec<f64>; 4] {
    let mut a = [
        vec![0.0; n * n],
        vec![0.0; n * n],
        vec![0.0; n * n],
        vec![0.0; n * n],
    ];
    for j in 0..n {
        for i in 0..n {
            let (x, y) = (TAU * i as f64 / n as f64, TAU * j as f64 / n as f64);
            a[0][j * n + i] = gx(x, y);
            a[1][j * n + i] = gxx(x, y);
            a[2][j * n + i] = gxy(x, y);
            a[3][j * n + i] = glap(x, y);
        }
    }
    a
}

/// Fourier pseudospectral max abs errors via the library's routines:
/// first derivatives from deriv_x / deriv_y, d2/dx2 and d2/dxdy by
/// composition, laplacian from its own operator.
fn fourier_errors(n: usize) -> [f64; 4] {
    let samples: Vec<f64> = (0..n * n)
        .map(|q| {
            let (i, j) = (q % n, q / n);
            g(TAU * i as f64 / n as f64, TAU * j as f64 / n as f64)
        })
        .collect();
    let g_hat = dft2(n, &samples);
    let dx = idft2(n, &deriv_x(n, &g_hat));
    let dxx = idft2(n, &deriv_x(n, &deriv_x(n, &g_hat)));
    let dxy = idft2(n, &deriv_y(n, &deriv_x(n, &g_hat)));
    let lap = idft2(n, &laplacian(n, &g_hat));
    let a = analytic(n);
    [
        max_err(&dx, &a[0]),
        max_err(&dxx, &a[1]),
        max_err(&dxy, &a[2]),
        max_err(&lap, &a[3]),
    ]
}

/// Second-order centred finite differences with periodic wrapping.
fn fd_errors(n: usize) -> [f64; 4] {
    let h = TAU / n as f64;
    let at = |i: usize, j: usize| {
        g(
            TAU * (i % n) as f64 / n as f64,
            TAU * (j % n) as f64 / n as f64,
        )
    };
    // first derivatives, centred
    let mut gx = vec![0.0; n * n];
    let mut gy = vec![0.0; n * n];
    let mut gxx = vec![0.0; n * n];
    let mut gyy = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            gx[j * n + i] = (at(i + 1, j) - at(i + n - 1, j)) / (2.0 * h);
            gy[j * n + i] = (at(i, j + 1) - at(i, j + n - 1)) / (2.0 * h);
            gxx[j * n + i] = (at(i + 1, j) - 2.0 * at(i, j) + at(i + n - 1, j)) / (h * h);
            gyy[j * n + i] = (at(i, j + 1) - 2.0 * at(i, j) + at(i, j + n - 1)) / (h * h);
        }
    }
    // mixed d2/dxdy: second-order centred dy followed by centred dx
    let mut gyx_stage = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            gyx_stage[j * n + i] = (at(i, j + 1) - at(i, j + n - 1)) / (2.0 * h);
        }
    }
    let mut gxy = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            gxy[j * n + i] = (gyx_stage[j * n + (i + 1) % n]
                - gyx_stage[j * n + (i + n - 1) % n])
                / (2.0 * h);
        }
    }
    let lap: Vec<f64> = (0..n * n).map(|q| gxx[q] + gyy[q]).collect();
    let a = analytic(n);
    [
        max_err(&gx, &a[0]),
        max_err(&gxx, &a[1]),
        max_err(&gxy, &a[2]),
        max_err(&lap, &a[3]),
    ]
}

fn main() {
    let fourier = fourier_errors(32);
    let fd32 = fd_errors(32);
    let fd64 = fd_errors(64);
    let names = ["d/dx", "d2/dx2", "d2/dxdy", "laplacian"];
    println!(
        "{:<10} {:>15} {:>15} {:>15} {:>14}",
        "derivative", "Fourier n=32", "FD n=32", "FD n=64", "FD ratio"
    );
    for k in 0..4 {
        println!(
            "{:<10} {:>15.6e} {:>15.6e} {:>15.6e} {:>14.6}",
            names[k], fourier[k], fd32[k], fd64[k], fd32[k] / fd64[k]
        );
    }
}
