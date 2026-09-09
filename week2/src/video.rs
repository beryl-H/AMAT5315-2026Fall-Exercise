//! PPM frame rendering (pixels, no fonts) and ffmpeg muxing.

/// One rendered frame as PPM (P6) bytes, width 2s by height s.
/// Left panel: the periodic box with atoms (positions wrapped into [0, box)).
/// Right panel: the RDF curve so far on its own axes, r in [0, box/2],
/// g in [0, 5].
pub fn render_frame(
    box_l: f64,
    positions: &[[f64; 2]],
    rdf_curve: &[(f64, f64)],
    s: u32,
) -> Vec<u8> {
    let s = s as usize;
    let (w, h) = (2 * s, s);
    let mut px = vec![255u8; w * h * 3];
    let set = |px: &mut [u8], x: f64, y: f64, c: [u8; 3]| {
        let (xi, yi) = (x as isize, y as isize);
        if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
            let o = ((yi as usize) * w + xi as usize) * 3;
            px[o] = c[0];
            px[o + 1] = c[1];
            px[o + 2] = c[2];
        }
    };
    let p = s as f64 * 0.05; // inner margin of each square panel
    let side = s as f64 - 2.0 * p; // drawing side of one panel
    let scale = side / box_l;
    let circle = |px: &mut [u8], cx: f64, cy: f64, rad: f64, c: [u8; 3]| {
        let r2 = rad * rad;
        for dy in -(rad as isize + 1)..=(rad as isize + 1) {
            for dx in -(rad as isize + 1)..=(rad as isize + 1) {
                if (dx as f64).powi(2) + (dy as f64).powi(2) <= r2 {
                    set(px, cx + dx as f64, cy + dy as f64, c);
                }
            }
        }
    };
    // Left panel: atoms at their positions wrapped into the periodic box.
    let wrap = |v: f64| v.rem_euclid(box_l);
    for pos in positions {
        circle(
            &mut px,
            p + wrap(pos[0]) * scale,
            p + wrap(pos[1]) * scale,
            0.4 * scale,
            [40, 40, 40],
        );
    }
    // Right panel: RDF curve in blue on light axes, on its own square.
    let x0 = s as f64 + p;
    let g_ceil = 5.0;
    let g_scale = side / g_ceil;
    let r_max = box_l / 2.0;
    let to_px = |r: f64, g: f64| (x0 + r / r_max * side, p + side - g * g_scale);
    // Axes (bottom and left of the right panel).
    for i in 0..(side as usize) {
        set(&mut px, x0 + i as f64, p + side, [180, 180, 180]);
        set(&mut px, x0, p + i as f64, [180, 180, 180]);
    }
    let line = |px: &mut [u8], a: (f64, f64), b: (f64, f64), c: [u8; 3]| {
        let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) as usize).max(1);
        for k in 0..=steps {
            let f = k as f64 / steps as f64;
            set(px, a.0 + f * (b.0 - a.0), a.1 + f * (b.1 - a.1), c);
        }
    };
    for wnd in rdf_curve.windows(2) {
        line(
            &mut px,
            to_px(wnd[0].0, wnd[0].1),
            to_px(wnd[1].0, wnd[1].1),
            [13, 110, 253],
        );
    }

    let mut out = Vec::with_capacity(w * h * 3 + 32);
    out.extend_from_slice(format!("P6\n{w} {h} 255\n").as_bytes());
    out.extend_from_slice(&px);
    out
}

/// Mux rendered PPM frames into an MP4 via ffmpeg (input %06d.ppm in dir).
pub fn mux(ppm_dir: &str, out: &str, fps: f64) -> std::io::Result<()> {
    let status = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg(format!("{fps}"))
        .arg("-i")
        .arg(format!("{ppm_dir}/%06d.ppm"))
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(out)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other("ffmpeg failed"))
    }
}
