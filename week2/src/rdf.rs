//! 2D radial distribution function g(r) with minimum-image distances.

/// Histogram accumulator for g(r) over many frames of one periodic box.
pub struct Rdf {
    box_l: f64,
    dr: f64,
    bins: Vec<f64>,
    frames: usize,
    n: usize,
}

impl Rdf {
    pub fn new(box_l: f64, dr: f64) -> Self {
        let bins = (box_l / 2.0 / dr).ceil() as usize;
        Rdf { box_l, dr, bins: vec![0.0; bins], frames: 0, n: 0 }
    }

    /// Add one frame's pair distances (minimum image) to the histogram.
    pub fn add_frame(&mut self, positions: &[[f64; 2]]) {
        let l = self.box_l;
        self.n = positions.len();
        self.frames += 1;
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let mut d = [
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                ];
                for k in 0..2 {
                    if d[k] > 0.5 * l {
                        d[k] -= l;
                    } else if d[k] < -0.5 * l {
                        d[k] += l;
                    }
                }
                let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
                let k = (r / self.dr) as usize;
                if k < self.bins.len() {
                    self.bins[k] += 1.0;
                }
            }
        }
    }

    /// The averaged, normalized g(r): (r_mid, g) for bins with nonzero counts.
    pub fn curve(&self) -> Vec<(f64, f64)> {
        let rho = self.n as f64 / (self.box_l * self.box_l);
        self.bins
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0.0)
            .map(|(k, c)| {
                let r0 = k as f64 * self.dr;
                let ring = std::f64::consts::PI * ((r0 + self.dr).powi(2) - r0.powi(2));
                let expected = self.frames as f64 * self.n as f64 * rho * ring / 2.0;
                (r0 + self.dr / 2.0, c / expected)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Rdf;

    #[test]
    fn single_known_pair() {
        // Two atoms 3.0 apart in a box of 10: exactly one pair in the bin
        // [3.0, 3.5). g = 1 / (N rho ring_area / 2) with the ring area
        // pi (3.5^2 - 3^2).
        let (positions, l) = (vec![[1.0, 5.0], [4.0, 5.0]], 10.0);
        let mut rdf = Rdf::new(l, 0.5);
        rdf.add_frame(&positions);
        let curve = rdf.curve();
        let n = 2.0;
        let rho = n / (l * l);
        let ring = std::f64::consts::PI * (3.5_f64.powi(2) - 3.0_f64.powi(2));
        let expected = 1.0 / (n * rho * ring / 2.0);
        let hit: Vec<(f64, f64)> = curve.iter().copied().filter(|(_, g)| *g > 0.0).collect();
        assert_eq!(hit.len(), 1);
        let (r, g) = hit[0];
        assert!((r - 3.25).abs() < 0.25, "bin center {r}");
        assert!((g - expected).abs() < 1e-9, "g {g} vs {expected}");
    }

    #[test]
    fn frames_average() {
        // Averaging two identical frames gives the same curve as one frame.
        let positions = vec![[1.0, 5.0], [4.0, 5.0]];
        let mut one = Rdf::new(10.0, 0.5);
        one.add_frame(&positions);
        let mut two = Rdf::new(10.0, 0.5);
        two.add_frame(&positions);
        two.add_frame(&positions);
        assert_eq!(one.curve(), two.curve());
    }
}
