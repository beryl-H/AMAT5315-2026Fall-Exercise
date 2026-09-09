//! Plot the Lennard-Jones pair field around one atom, from the crate's own
//! `energy` and `force` functions: energy as background colours (white = 0,
//! blue = attraction, red = repulsion, clipped at +-1), force on a neighbour
//! as arrows that point radially outward inside r0 and inward outside.
//!
//! Run from the crate root `week2/`:
//!     cargo run --example field
//!     rsvg-convert -o field.png target/field.svg

use std::fmt::Write as _;

const WHITE: [u8; 3] = [255, 255, 255];
const BLUE: [u8; 3] = [13, 110, 253];
const RED: [u8; 3] = [220, 30, 60];
const INK: &str = "#333333";

fn shade(u: f64) -> String {
    let u = u.clamp(-1.0, 1.0);
    let (a, b) = if u < 0.0 { (WHITE, BLUE) } else { (WHITE, RED) };
    let t = u.abs();
    let c: [u8; 3] =
        std::array::from_fn(|k| (a[k] as f64 + (b[k] as f64 - a[k] as f64) * t).round() as u8);
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn main() {
    let mut svg = String::from(
        "<svg xmlns='http://www.w3.org/2000/svg' width='720' height='720' viewBox='-3.9 -3.9 7.8 7.8'>",
    );
    svg.push_str("<rect x='-3.9' y='-3.9' width='7.8' height='7.8' fill='#ffffff'/>");

    // Pair energy as a 60x60 heatmap on [-3, 3]^2, clipped to [-1, 1].
    let n = 60;
    let cell = 6.0 / n as f64;
    for i in 0..n {
        for j in 0..n {
            let x = -3.0 + (i as f64 + 0.5) * cell;
            let y = -3.0 + (j as f64 + 0.5) * cell;
            let r = (x * x + y * y).sqrt();
            // Saturated repulsive core; otherwise energy as-is.
            let u = if r < 0.8 {
                1.0
            } else {
                md::energy(r).clamp(-1.0, 1.0)
            };
            let c = shade(u);
            let _ = write!(
                svg,
                "<rect x='{:.3}' y='{:.3}' width='{:.3}' height='{:.3}' fill='{}' stroke='{}' stroke-width='0.004'/>",
                x - cell / 2.0,
                y - cell / 2.0,
                cell,
                cell,
                c,
                c
            );
        }
    }

    // Force on atom j as arrows on a 12x12 grid; length compressed, sign kept.
    let r0 = 2.0_f64.powf(1.0 / 6.0);
    for i in 0..12 {
        for j in 0..12 {
            let x = -2.75 + i as f64 * 0.5;
            let y = -2.75 + j as f64 * 0.5;
            let r = (x * x + y * y).sqrt();
            if r <= 0.95 {
                continue; // inside the central atom
            }
            let f = md::force(r);
            let (ux, uy) = (f.signum() * x / r, f.signum() * y / r);
            let len = 0.32 * (f.abs().min(3.0) / 3.0).sqrt();
            let (hx, hy) = (ux * len / 2.0, uy * len / 2.0);
            let (px, py) = (x - hx, y - hy); // shaft start
            let (tx, ty) = (x + hx, y + hy); // arrow tip
            let (nx, ny) = (-uy, ux); // perpendicular to the shaft
            let (back, wing) = (0.30 * len, 0.16 * len);
            let _ = write!(
                svg,
                "<line x1='{:.3}' y1='{:.3}' x2='{:.3}' y2='{:.3}' stroke='{}' stroke-width='0.014'/>",
                px, py, tx, ty, INK
            );
            let _ = write!(
                svg,
                "<polygon points='{:.3},{:.3} {:.3},{:.3} {:.3},{:.3}' fill='{}'/>",
                tx,
                ty,
                tx - ux * back + nx * wing,
                ty - uy * back + ny * wing,
                tx - ux * back - nx * wing,
                ty - uy * back - ny * wing,
                INK
            );
        }
    }

    // Equilibrium circle, central atom, and frame.
    let _ = write!(
        svg,
        "<circle cx='0' cy='0' r='{:.3}' fill='none' stroke='{}' stroke-width='0.02' stroke-dasharray='0.2 0.15'/>",
        r0, INK
    );
    svg.push_str("<circle cx='0' cy='0' r='0.12' fill='#000000'/>");
    svg.push_str(
        "<rect x='-3' y='-3' width='6' height='6' fill='none' stroke='#999999' stroke-width='0.02'/>",
    );
    svg.push_str(
        "<text x='-3' y='-3.18' font-size='0.19' fill='#555555'>dashed circle: r = r0 ~ 1.12</text>",
    );

    // Tick marks and labels on both axes.
    for v in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let _ = write!(
            svg,
            "<line x1='{:.1}' y1='3' x2='{:.1}' y2='3.07' stroke='#999999' stroke-width='0.02'/>",
            v, v
        );
        let _ = write!(
            svg,
            "<text x='{:.1}' y='3.3' font-size='0.18' text-anchor='middle' fill='#777777'>{:.0}</text>",
            v, v
        );
        let _ = write!(
            svg,
            "<line x1='-3' y1='{:.1}' x2='-3.07' y2='{:.1}' stroke='#999999' stroke-width='0.02'/>",
            v, v
        );
        let _ = write!(
            svg,
            "<text x='-3.2' y='{:.1}' font-size='0.18' text-anchor='end' dominant-baseline='middle' fill='#777777'>{:.0}</text>",
            v, v
        );
    }

    svg.push_str("</svg>");
    std::fs::write("target/field.svg", svg).expect("write target/field.svg");
    println!("wrote target/field.svg");
}
