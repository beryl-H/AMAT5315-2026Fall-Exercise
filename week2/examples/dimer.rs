//! Plot the dimer experiment's relative energy error against time, using the
//! md crate's own `dimer`, `run`, `Euler`, and `VelocityVerlet`.
//!
//! Left panel: both integrators, 500 steps at dt = 0.01.
//! Right panel: velocity-Verlet alone for 5000 steps, error scaled by 1000.
//!
//! Run from the crate root `week2/`:
//!     cargo run --example dimer
//!     rsvg-convert -o dimer.png target/dimer.svg

use std::fmt::Write as _;

use md::{dimer, run, Euler, VelocityVerlet};

const DT: f64 = 0.01;
const RED: &str = "#dc1e3c";
const BLUE: &str = "#0d6efd";
const AXIS: &str = "#555555";
const GRID: &str = "#dddddd";

fn main() {
    // The three runs, all from the same dimer() initial state.
    let mut s = dimer();
    let euler = run(&Euler, &mut s, DT, 500);
    let mut s = dimer();
    let verlet = run(&VelocityVerlet, &mut s, DT, 500);
    let mut s = dimer();
    let verlet_long = run(&VelocityVerlet, &mut s, DT, 5000);

    let emax = euler.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    let vmax = verlet.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    let vmax_long = verlet_long.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    println!(
        "Euler 500 max {emax:.3e}; Verlet 500 max {vmax:.3e}; Verlet 5000 max {vmax_long:.3e}"
    );

    let scaled: Vec<f64> = verlet_long.iter().map(|e| e * 1000.0).collect();

    let mut svg = String::from(
        "<svg xmlns='http://www.w3.org/2000/svg' width='1100' height='520' viewBox='0 0 1100 520'>",
    );
    svg.push_str("<rect x='0' y='0' width='1100' height='520' fill='#ffffff'/>");

    // Panel geometry: (x, y, width, height) of each plotting box.
    let (ax, ay, aw, ah) = (70.0, 60.0, 470.0, 360.0);
    let (bx, by, bw, bh) = (590.0, 60.0, 470.0, 360.0);

    // Left panel: t in [0, 5], y range covers Euler (pumping) and Verlet (flat).
    let xr = (0.0, 5.0);
    let ymin = euler
        .iter()
        .chain(&verlet)
        .fold(0.0_f64, |m, e| m.min(*e))
        .min(0.0);
    let ymax = euler
        .iter()
        .chain(&verlet)
        .fold(0.0_f64, |m, e| m.max(*e))
        .max(0.0);
    let left = yrange(ymin, ymax);
    panel(
        &mut svg,
        "500 steps at dt = 0.01",
        (ax, ay, aw, ah),
        xr,
        left,
        "Euler",
        RED,
        &euler,
        "velocity-Verlet",
        BLUE,
        &verlet,
    );

    // Right panel: Verlet alone, 5000 steps, error scaled by 1000.
    let xr2 = (0.0, 50.0);
    let ymin2 = scaled.iter().fold(0.0_f64, |m, e| m.min(*e)).min(0.0);
    let ymax2 = scaled.iter().fold(0.0_f64, |m, e| m.max(*e)).max(0.0);
    let right = yrange(ymin2, ymax2);
    panel(
        &mut svg,
        "velocity-Verlet, 5000 steps (error x1000)",
        (bx, by, bw, bh),
        xr2,
        right,
        "velocity-Verlet",
        BLUE,
        &scaled,
        "",
        GRID,
        &[],
    );

    svg.push_str("</svg>");
    std::fs::write("target/dimer.svg", svg).expect("write target/dimer.svg");
    println!("wrote target/dimer.svg");
}

/// Nice y-axis range [lo, hi] with lo <= 0 <= hi, padded and rounded to a
/// step of 1, 2, or 5 times a power of ten.
fn yrange(min: f64, max: f64) -> (f64, f64, f64) {
    let pad = (max - min) * 0.05;
    let (lo, hi) = (min - pad, max + pad);
    let raw = (hi - lo) / 6.0;
    let k = 10.0_f64.powf(raw.log10().floor());
    let step = if raw >= 5.0 * k {
        5.0 * k
    } else if raw >= 2.0 * k {
        2.0 * k
    } else {
        k
    };
    (lo.floor() / step * step, hi.ceil() / step * step, step)
}

/// Draw one panel: frame, y = 0 axis, ticks, optional legend, and two series.
fn panel(
    svg: &mut String,
    title: &str,
    (x0, y0, w, h): (f64, f64, f64, f64),
    xr: (f64, f64),
    (lo, hi, step): (f64, f64, f64),
    name1: &str,
    color1: &str,
    data1: &[f64],
    name2: &str,
    color2: &str,
    data2: &[f64],
) {
    // Title above the panel, then the frame.
    let _ = write!(svg, "<text x='{x0}' y='{}' font-size='20' fill='#222222'>{title}</text>", y0 - 18.0);

    // Frame and axes.
    let _ = write!(svg, "<rect x='{x0}' y='{y0}' width='{w}' height='{h}' fill='none' stroke='{AXIS}' stroke-width='1.2'/>");
    let x_axis_y = y_of(0.0, y0, h, lo, hi);
    let _ = write!(svg, "<line x1='{x0}' y1='{x_axis_y}' x2='{}' y2='{x_axis_y}' stroke='{AXIS}' stroke-width='1'/>", x0 + w);

    // Horizontal grid lines and y tick labels at multiples of step.
    let mut t = (lo / step).ceil() * step;
    while t <= hi {
        let y = y_of(t, y0, h, lo, hi);
        let _ = write!(svg, "<line x1='{x0}' y1='{y}' x2='{}' y2='{y}' stroke='{GRID}' stroke-width='0.8'/>", x0 + w);
        let _ = write!(
            svg,
            "<text x='{}' y='{}' font-size='15' text-anchor='end' fill='{AXIS}'>{:.3}</text>",
            x0 - 8.0,
            y + 5.0,
            t
        );
        t += step;
    }

    // x ticks at nice times, labelled under the axis.
    let span = xr.1 - xr.0;
    let xt = if span > 30.0 { 10.0 } else { 1.0 };
    let mut tt = xt;
    while tt <= xr.1 {
        let x = x_of(tt, x0, w, xr);
        let _ = write!(svg, "<line x1='{x}' y1='{x_axis_y}' x2='{x}' y2='{}' stroke='{GRID}' stroke-width='0.8'/>", y0 + h);
        let _ = write!(
            svg,
            "<text x='{x}' y='{}' font-size='15' text-anchor='middle' fill='{AXIS}'>{:.0}</text>",
            y0 + h + 22.0,
            tt
        );
        tt += xt;
    }

    // t axis label and y axis label.
    let _ = write!(svg, "<text x='{}' y='{}' font-size='16' text-anchor='middle' fill='{AXIS}'>t</text>", x0 + w / 2.0, y0 + h + 46.0);
    let _ = write!(svg, "<text x='{}' y='{}' font-size='16' text-anchor='middle' fill='{AXIS}'>relative energy error</text>", x0 + w / 2.0, y0 - 42.0);

    // Series as polylines.
    let mut points1 = String::new();
    for (i, e) in data1.iter().enumerate() {
        let t = (i as f64 + 1.0) * DT;
        let _ = write!(points1, " {:.2},{:.2}", x_of(t, x0, w, xr), y_of(*e, y0, h, lo, hi));
    }
    let _ = write!(svg, "<polyline points='{}' fill='none' stroke='{color1}' stroke-width='1.6'/>", points1.trim_start());
    if !data2.is_empty() {
        let mut points2 = String::new();
        for (i, e) in data2.iter().enumerate() {
            let t = (i as f64 + 1.0) * DT;
            let _ = write!(points2, " {:.2},{:.2}", x_of(t, x0, w, xr), y_of(*e, y0, h, lo, hi));
        }
        let _ = write!(svg, "<polyline points='{}' fill='none' stroke='{color2}' stroke-width='1.6'/>", points2.trim_start());
    }

    // Legend (two lines when a second series is present).
    if !name1.is_empty() {
        let (lx, ly0) = (x0 + 12.0, y0 + 24.0);
        let lines: [(&str, &str); 2] = [(name1, color1), (name2, color2)];
        for (k, (name, color)) in lines.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let ly = ly0 + k as f64 * 20.0;
            let _ = write!(svg, "<line x1='{lx}' y1='{ly}' x2='{}' y2='{ly}' stroke='{color}' stroke-width='2.4'/>", lx + 34.0);
            let _ = write!(svg, "<text x='{}' y='{}' font-size='16' fill='#222222'>{name}</text>", lx + 42.0, ly + 5.0);
        }
    }
}

fn x_of(t: f64, x0: f64, w: f64, xr: (f64, f64)) -> f64 {
    x0 + (t - xr.0) / (xr.1 - xr.0) * w
}

fn y_of(v: f64, y0: f64, h: f64, lo: f64, hi: f64) -> f64 {
    y0 + h - (v - lo) / (hi - lo) * h
}
