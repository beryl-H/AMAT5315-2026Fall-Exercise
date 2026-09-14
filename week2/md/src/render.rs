//! Minimal RGBA rasterizer for the video panels (no image crate).

use crate::simulate::Frame;
use crate::system::Box2;

/// A simple RGBA canvas.
#[derive(Clone, Debug)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Canvas {
    /// New canvas starting opaque black (alpha 255, so video frames are not
    /// transparent).
    pub fn new(width: usize, height: usize) -> Canvas {
        let mut rgba = vec![0u8; width * height * 4];
        for px in rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&[0, 0, 0, 255]);
        }
        Canvas { width, height, rgba }
    }

    /// Fill the whole canvas with one color.
    pub fn fill(&mut self, [r, g, b, a]: [u8; 4]) {
        for px in self.rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&[r, g, b, a]);
        }
    }

    /// Set one pixel (out-of-bounds is ignored).
    pub fn set(&mut self, x: usize, y: usize, color: [u8; 4]) {
        if x < self.width && y < self.height {
            let i = (y * self.width + x) * 4;
            self.rgba[i..i + 4].copy_from_slice(&color);
        }
    }

    /// Filled disc via integer radius scan (small radii only).
    pub fn filled_disc(&mut self, cx: usize, cy: usize, radius: usize, color: [u8; 4]) {
        let r = radius as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let x = cx as isize + dx;
                    let y = cy as isize + dy;
                    if x >= 0 && y >= 0 {
                        self.set(x as usize, y as usize, color);
                    }
                }
            }
        }
    }

    /// Bresenham-ish line by linear interpolation.
    pub fn line(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, color: [u8; 4]) {
        let steps = usize::max(x1.abs_diff(x0), y1.abs_diff(y0)).max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let x = (x0 as f64 + t * (x1 as f64 - x0 as f64)) as usize;
            let y = (y0 as f64 + t * (y1 as f64 - y0 as f64)) as usize;
            self.set(x, y, color);
        }
    }

    pub fn into_pixels(self) -> Vec<u8> {
        self.rgba
    }
}

/// Render one trajectory frame: left panel = periodic box + particle discs,
/// right panel = the static whole-trajectory g(r) polyline with a g = 1
/// reference. Resolution/margins are [Suggestion] visualization choices.
pub fn render_frame(
    frame: &Frame,
    bx: &Box2,
    g: &[(f64, f64)],
    (w, h): (usize, usize),
) -> Vec<u8> {
    let mut canvas = Canvas::new(w, h);
    let panel_w = w / 2;
    let margin = 8usize; // [Suggestion] layout

    // Left panel: periodic box border + particles.
    let box_w = panel_w - 2 * margin;
    let box_h = h - 2 * margin;
    let sx = box_w as f64 / bx.lx;
    let sy = box_h as f64 / bx.ly;
    let (x0, y0) = (margin, margin);
    let (x1, y1) = (panel_w - margin, h - margin);
    canvas.line(x0, y0, x1, y0, [255, 255, 255, 255]);
    canvas.line(x0, y1, x1, y1, [255, 255, 255, 255]);
    canvas.line(x0, y0, x0, y1, [255, 255, 255, 255]);
    canvas.line(x1, y0, x1, y1, [255, 255, 255, 255]);
    for p in &frame.pos {
        let px = margin + (p[0] * sx) as usize;
        let py = margin + (p[1] * sy) as usize;
        canvas.filled_disc(px, py, 3, [255, 80, 80, 255]); // [Suggestion] radius/color
    }

    // Right panel: g(r) polyline + g = 1 reference line.
    let gx0 = panel_w + margin;
    let gy0 = h - margin;
    let gx1 = w - margin;
    let gy1 = margin;
    let g_w = gx1 - gx0;
    let g_h = gy0 - gy1;
    let g_max = g.iter().map(|(_, v)| *v).fold(2.0, f64::max).max(2.0);
    let r_max = g.last().map(|(r, _)| *r).unwrap_or(1.0).max(1e-9);
    canvas.line(gx0, gy0, gx1, gy0, [255, 255, 255, 255]);
    canvas.line(gx0, gy0, gx0, gy1, [255, 255, 255, 255]);
    let ref_y = gy0 - ((1.0 / g_max) * g_h as f64) as usize;
    canvas.line(gx0, ref_y, gx1, ref_y, [120, 120, 120, 255]);
    for pair in g.windows(2) {
        let (ra, ga) = pair[0];
        let (rb, gb) = pair[1];
        let xa = gx0 + (ra / r_max * g_w as f64) as usize;
        let ya = gy0 - (ga / g_max * g_h as f64) as usize;
        let xb = gx0 + (rb / r_max * g_w as f64) as usize;
        let yb = gy0 - (gb / g_max * g_h as f64) as usize;
        canvas.line(xa, ya, xb, yb, [0, 200, 255, 255]);
    }

    canvas.into_pixels()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_starts_opaque_black_and_supports_disc() {
        let mut c = Canvas::new(16, 16);
        assert_eq!(c.rgba.len(), 16 * 16 * 4);
        assert!(c.rgba.chunks(4).all(|px| px[3] == 255), "canvas must start opaque");
        c.filled_disc(8, 8, 3, [255, 0, 0, 255]);
        assert_eq!(c.rgba[(8 * 16 + 8) * 4], 255);
        // A pixel outside the disc radius is untouched (black).
        assert_eq!(c.rgba[(0 * 16 + 0) * 4], 0);
    }

    #[test]
    fn line_draws_straight_horizontal() {
        let mut c = Canvas::new(16, 4);
        c.line(2, 2, 13, 2, [255; 4]);
        assert_eq!(c.rgba[(2 * 16 + 2) * 4], 255);
        assert_eq!(c.rgba[(2 * 16 + 13) * 4], 255);
    }

    #[test]
    fn render_frame_produces_full_size_rgba() {
        let frame = crate::simulate::Frame {
            step: 50,
            t: 0.5,
            pos: vec![[0.0, 0.0], [2.0, 2.0], [4.0, 1.0], [1.0, 4.0]],
            vel: vec![[0.0; 2]; 4],
            e_pot: 0.0,
            e_kin: 0.0,
        };
        let bx = crate::system::Box2 { lx: 5.0, ly: 5.0 };
        let g = vec![(0.5, 1.0); 10];
        let pixels = render_frame(&frame, &bx, &g, (320, 160));
        assert_eq!(pixels.len(), 320 * 160 * 4);
    }

    #[test]
    fn render_frame_is_deterministic() {
        let frame = crate::simulate::Frame {
            step: 50,
            t: 0.5,
            pos: vec![[0.3, 1.1], [2.0, 2.0], [4.4, 0.5]],
            vel: vec![[0.1, 0.2]; 3],
            e_pot: 0.0,
            e_kin: 0.0,
        };
        let bx = crate::system::Box2 { lx: 5.0, ly: 5.0 };
        let g = vec![(0.25, 1.1), (0.75, 1.0), (1.25, 0.95)];
        let a = render_frame(&frame, &bx, &g, (64, 32));
        let b = render_frame(&frame, &bx, &g, (64, 32));
        assert_eq!(a, b, "rendering must be deterministic");
    }

    #[test]
    fn particle_coordinates_map_inside_left_panel() {
        // One atom at the box centre: it must draw a disc at the centre of
        // the left (box) panel, i.e. around pixel ((panel_w + margin)/2, h/2).
        let frame = crate::simulate::Frame {
            step: 50,
            t: 0.5,
            pos: vec![[2.5, 2.5]],
            vel: vec![[0.0, 0.0]],
            e_pot: 0.0,
            e_kin: 0.0,
        };
        let bx = crate::system::Box2 { lx: 5.0, ly: 5.0 };
        let g = vec![(0.5, 1.0); 10];
        let (w, h) = (160usize, 80usize);
        let pixels = render_frame(&frame, &bx, &g, (w, h));
        // Centre of the left panel (w/4, h/2).
        let cx = w / 4;
        let cy = h / 2;
        assert_eq!(pixels[(cy * w + cx) * 4], 255, "particle red channel at panel centre");
        assert_eq!(pixels[(cy * w + cx) * 4 + 3], 255, "particle opaque");
    }
}
