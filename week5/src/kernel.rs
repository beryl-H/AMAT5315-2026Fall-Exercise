#![no_std]
#![feature(autodiff)]
use core::autodiff::{autodiff_forward, autodiff_reverse};

// Only this static library receives -Zautodiff=Enable. No allocation or std.
#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 {
    x * x * x
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
    let (y, dy) = cube_forward(x, 1.0);
    let (_, adj) = cube_reverse(x, 1.0);
    unsafe {
        *out = y;
        *out.add(1) = dy;
        *out.add(2) = adj;
    }
}

fn step(
    state: &[f64], velocity: &[f64], sigma: &[f64], source: &[f64],
    dt: f64, dx: f64, nx: usize, nz: usize, out: &mut [f64],
) {
    let n = nx * nz;
    out.fill(0.0);
    for z in 1..nz - 1 {
        for x in 1..nx - 1 {
            let i = z * nx + x;
            let current = state[n + i];
            let lap = (state[n + i - 1] + state[n + i + 1]
                + state[n + i - nx] + state[n + i + nx] - 4.0 * current) / (dx * dx);
            out[i] = current;
            out[n + i] = (2.0 * current - (1.0 - sigma[i] * dt) * state[i]
                + dt * dt * (velocity[i] * velocity[i] * lap + source[i]))
                / (1.0 + sigma[i] * dt);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn seismic_step(
    state: *const f64, velocity: *const f64, sigma: *const f64, source: *const f64,
    dt: f64, dx: f64, nx: usize, nz: usize, out: *mut f64,
) {
    let n = nx * nz;
    unsafe {
        step(core::slice::from_raw_parts(state, 2 * n),
             core::slice::from_raw_parts(velocity, n),
             core::slice::from_raw_parts(sigma, n),
             core::slice::from_raw_parts(source, n), dt, dx, nx, nz,
             core::slice::from_raw_parts_mut(out, 2 * n));
    }
}

unsafe extern "C" { fn abort() -> !; }
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { unsafe { abort() } }
