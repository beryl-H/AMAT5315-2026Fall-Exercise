//! Deterministic seeded RNG: xorshift64* with Box-Muller Gaussians.

pub struct Rng(u64);

impl Rng {
    /// New generator from a seed; seed 0 is mapped away (xorshift needs nonzero).
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.uniform(), b.uniform());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let da: Vec<f64> = (0..10).map(|_| a.uniform()).collect();
        let db: Vec<f64> = (0..10).map(|_| b.uniform()).collect();
        assert_ne!(da, db);
    }

    #[test]
    fn uniform_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn gaussian_moments() {
        let mut r = Rng::new(3);
        let n = 20_000;
        let (mut sum, mut sum2) = (0.0, 0.0);
        for _ in 0..n {
            let g = r.gaussian();
            sum += g;
            sum2 += g * g;
        }
        let mean = sum / n as f64;
        let var = sum2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((0.93..1.07).contains(&var), "var {var}");
    }
}
