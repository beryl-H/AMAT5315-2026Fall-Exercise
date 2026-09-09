//! PPM frame rendering (pixels, no fonts) and ffmpeg muxing.

/// One rendered frame as PPM (P6) bytes. Left panel: the box with atoms.
/// Right panel: the RDF curve so far, fixed axes r in [0, box/2], g in [0, 5].
pub fn render_frame(
    box_l: f64,
    positions: &[[f64; 2]],
    rdf_curve: &[(f64, f64)],
    size: u32,
) -> Vec<u8> {
    let size = size as usize;
    let mut px = vec![255u8; size * size * 3];
    let set = |px: &mut [u8], x: f64, y: f64, c: [u8; 3]| {
        let (xi, yi) = (x as isize, y as isize);
        if xi >= 0 && yi >= 0 && (xi as usize) < size && (yi as usize) < size {
            let o = (yi * size + xi) * 3;
            px[o] = c[0];
            px[o + 1] = c[1];
            px[o + 2] = c[2];
        }
    };
    let panel = size as f64 * 0.05; // margin
    let side = size as f64 - 2.0 * panel;
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
    // Atoms: dark circles of radius 0.4 sigma.
    for p in positions {
        circle(
            &mut px,
            panel + p[0] * scale,
            panel + p[1] * scale,
            0.4 * scale,
            [40, 40, 40],
        );
    }
    // Right panel: RDF curve in blue on light axes.
    let rx0 = size as f64 - panel - side;
    let g_ceil = 5.0;
    let g_scale = side / g_ceil;
    let r_max = box_l / 2.0;
    let to_px = |r: f64, g: f64| (rx0 + r / r_max * side, panel + side - g * g_scale);
    // Axes.
    for i in 0..(side as usize) {
        set(&mut px, rx0 + i as f64, panel + side, [180, 180, 180]);
        set(&mut px, rx0, panel + i as f64, [180, 180, 180]);
    }
    let line = |px: &mut [u8], p: (f64, f64), q: (f64, f64), c: [u8; 3]| {
        let steps = ((q.0 - p.0).abs().max((q.1 - p.1).abs()) as usize).max(1);
        for k in 0..=steps {
            let f = k as f64 / steps as f64;
            set(px, p.0 + f * (q.0 - p.0), p.1 + f * (q.1 - p.1), c);
        }
    };
    for w in rdf_curve.windows(2) {
        line(
            &mut px,
            to_px(w[0].0, w[0].1),
            to_px(w[1].0, w[1].1),
            [13, 110, 253],
        );
    }

    let mut out = Vec::with_capacity(size * size * 3 + 32);
    out.extend_from_slice(format!("P6\n{size} {size} 255\n").as_bytes());
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
