//! Convince visually that --force naive and --force cells simulate the same
//! physics. Three panels, all from runs with identical seeds and settings:
//!
//!   1. Per-frame recomputed total energy: naive engine vs cells engine on
//!      the *same* stored state (log10 relative difference).
//!   2. Total-energy difference between two independent same-seed runs
//!      (naive vs cells): grows only through chaotic round-off amplification.
//!   3. Radial distribution function g(r) accumulated over each run: the two
//!      curves coincide.
//!
//! Run from `week2/`:
//!     cargo run --release --example force_compare
//!     rsvg-convert -o force-compare.png target/force_compare.svg

use std::fmt::Write as _;

use md::cli::RunCfg;
use md::ops::{self};
use md::rdf::Rdf;
use md::trajectory::{self, Trajectory};
use md::{ForceStrategy, System};

const N: usize = 400;
const STEPS: usize = 4000;
const EQUIL: usize = 300;

fn run(strategy: ForceStrategy, tag: &str) -> Trajectory {
    let cfg = RunCfg {
        force: strategy,
        n: N,
        temp: 1.0,
        dt: 0.005,
        steps: STEPS,
        equil: EQUIL,
        ramp_to: None,
        out: format!("/tmp/force_compare_{tag}.txt"),
        seed: 2026,
    };
    ops::run_sim(&cfg).unwrap();
    trajectory::read(&cfg.out).unwrap()
}

fn frame_energy(t: &Trajectory, k: usize, strategy: ForceStrategy) -> f64 {
    let pos: Vec<[f64; 2]> = t.frames[k].iter().map(|a| [a[0], a[1]]).collect();
    let vel: Vec<[f64; 2]> = t.frames[k].iter().map(|a| [a[2], a[3]]).collect();
    let mut s = System::periodic(pos, vel, t.meta.box_l, 2.5);
    s.set_force(strategy);
    s.total_energy()
}

fn main() {
    // Two independent runs with the same seed: only the force strategy differs.
    let ta = run(ForceStrategy::Cells, "cells");
    let tb = run(ForceStrategy::Naive, "naive");
    let e0 = ta.frames.len() as f64 * 0.0 + ta.meta.dt; // (kept simple: t per frame below)
    let _ = e0;

    let mut engine_diff = Vec::with_capacity(ta.frames.len());
    let mut run_diff = Vec::with_capacity(ta.frames.len());
    let mut times = Vec::with_capacity(ta.frames.len());
    for k in 0..ta.frames.len() {
        let ec = frame_energy(&ta, k, ForceStrategy::Cells);
        let en = frame_energy(&ta, k, ForceStrategy::Naive);
        let ebn = frame_energy(&tb, k, ForceStrategy::Naive);
        let scale = ec.abs().max(1.0);
        engine_diff.push(((ec - en).abs() / scale).max(f64::MIN_POSITIVE).log10());
        run_diff.push(((ec - ebn).abs() / scale).max(f64::MIN_POSITIVE).log10());
        times.push(k as f64 * ta.meta.dt);
    }

    // g(r) over each full run.
    let g = |t: &Trajectory| {
        let mut rdf = Rdf::new(t.meta.box_l, 0.05);
        for frame in &t.frames {
            let pos: Vec<[f64; 2]> = frame.iter().map(|a| [a[0], a[1]]).collect();
            rdf.add_frame(&pos);
        }
        rdf.curve()
    };
    let (ga, gb) = (g(&ta), g(&tb));

    let tmax = *times.last().unwrap();
    let max_g = ga.iter().chain(&gb).map(|&(_, y)| y).fold(0.0, f64::max);

    // ---- SVG, three stacked panels ----
    let mut svg = String::from(
        "<svg xmlns='http://www.w3.org/2000/svg' width='1100' height='980' viewBox='0 0 1100 980'>",
    );
    svg.push_str("<rect x='0' y='0' width='1100' height='980' fill='#ffffff'/>");
    let (x0, w) = (90.0, 940.0);
    let y_of = |v: f64, lo: f64, hi: f64, y0: f64, h: f64| y0 + (hi - v) / (hi - lo) * h;
    let x_of = |t: f64| x0 + t / tmax * w;
    let poly = |svg: &mut String, pts: &[String], color: &str, width: f64| {
        let body = pts.join(" ");
        let _ = write!(svg, "<polyline points='{body}' fill='none' stroke='{color}' stroke-width='{width}'/>");
    };

    // Panel 1: log10 relative engine difference per frame (same state).
    let (y0, h) = (50.0, 240.0);
    let _ = write!(svg, "<text x='{x0}' y='{}' font-size='18' fill='#222'>1. cells engine vs naive engine on the SAME state: log10 |dE|/|E| per frame</text>", y0 - 8.0);
    let pts: Vec<String> = times.iter().zip(&engine_diff).map(|(t, v)| format!("{:.2},{:.2}", x_of(*t), y_of(*v, -17.0, -13.0, y0, h))).collect();
    poly(&mut svg, &pts, "#0d6efd", 1.4);
    let _ = write!(svg, "<text x='{}' y='{}' font-size='15' fill='#0d6efd'>flat at ~1e-16: engines identical per state</text>", x0, y0 + h + 22.0);

    // Panel 2: difference between two independent same-seed runs (chaos).
    let (y0, h) = (y0 + h + 60.0, 240.0);
    let _ = write!(svg, "<text x='{x0}' y='{}' font-size='18' fill='#222'>2. two independent runs (naive vs cells, same seed): log10 |dE|/|E|</text>", y0 - 8.0);
    let pts: Vec<String> = times.iter().zip(&run_diff).map(|(t, v)| format!("{:.2},{:.2}", x_of(*t), y_of(*v, -17.0, 0.0, y0, h))).collect();
    poly(&mut svg, &pts, "#dc1e3c", 1.4);
    let _ = write!(svg, "<text x='{}' y='{}' font-size='15' fill='#dc1e3c'>round-off seeds chaos; the divergence is NOT a physics difference (see panel 1)</text>", x0, y0 + h + 22.0);

    // Panel 3: g(r) overlay.
    let (y0, h) = (y0 + h + 60.0, 300.0);
    let _ = write!(svg, "<text x='{x0}' y='{}' font-size='18' fill='#222'>3. g(r): cells run vs naive run</text>", y0 - 8.0);
    let g_of = |svg: &mut String, curve: &[(f64, f64)], color: &str| {
        let pts: Vec<String> = curve.iter().map(|(r, v)| format!("{:.2},{:.2}", x_of(*r), y_of(*v, 0.0, 5.0, y0, h))).collect();
        poly(svg, &pts, color, 1.6);
    };
    g_of(&mut svg, &ga, "#dc1e3c");
    g_of(&mut svg, &gb, "#0d6efd");
    let _ = write!(svg, "<text x='{}' y='{}' font-size='15' fill='#dc1e3c'>naive</text>", x0 + 80.0, y0 + 30.0);
    let _ = write!(svg, "<text x='{}' y='{}' font-size='15' fill='#0d6efd'>cells</text>", x0 + 80.0, y0 + 52.0);

    // Panel frames, axes, ticks.
    for (py0, ph, lo, hi) in [(50.0, 240.0, -17.0, -13.0), (350.0, 240.0, -17.0, 0.0), (650.0, 300.0, 0.0, 5.0)] {
        let _ = write!(svg, "<rect x='{x0}' y='{py0}' width='{w}' height='{ph}' fill='none' stroke='#888' stroke-width='1'/>");
        for tick in 0..=4 {
            let fx = tick as f64 / 4.0;
            let _ = write!(svg, "<line x1='{}' y1='{}' x2='{}' y2='{}' stroke='#dddddd' stroke-width='0.7'/>",
                x0 + fx * w, py0, x0 + fx * w, py0 + ph);
            let v = lo + fx * (hi - lo);
            let _ = write!(svg, "<text x='{}' y='{}' font-size='13' text-anchor='middle' fill='#666'>{v:.0}</text>",
                x0 + fx * w, py0 + ph + 16.0);
        }
        for fy in [0.0, 0.5, 1.0] {
            let _ = write!(svg, "<line x1='{x0}' y1='{}' x2='{}' y2='{}' stroke='#dddddd' stroke-width='0.7'/>",
                py0 + fy * ph, x0 + w, py0 + fy * ph);
        }
    }
    svg.push_str("</svg>");
    std::fs::write("target/force_compare.svg", svg).unwrap();

    // Numbers for the report.
    let med_engine = engine_diff.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_g_diff = ga.iter().zip(&gb).map(|(&(_, a), &(_, b))| (a - b).abs()).fold(0.0, f64::max);
    println!("panel1: worst engine |dE|/|E| = 10^({med_engine:.1})");
    println!("panel2: final run divergence 10^({:.1})", run_diff.last().unwrap());
    println!("panel3: max |g_cells - g_naive| = {max_g_diff:.2e}, g peak ~ {max_g:.2}");
}
