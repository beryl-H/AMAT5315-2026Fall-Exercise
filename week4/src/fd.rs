//! Centred finite differences for u_t + c u_x = nu u_xx on the periodic
//! grid x_j = 2 pi j / n, j = 0..n-1, h = 2 pi / n.

use std::f64::consts::TAU;

pub struct CentredFD {
    pub c: f64,
    pub nu: f64,
    pub n: usize,
}

impl CentredFD {
    fn h(&self) -> f64 {
        TAU / self.n as f64
    }

    /// Periodic centred first difference (u_{j+1} - u_{j-1}) / (2h).
    pub fn derivative(&self, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n, "state length must equal n");
        let h = self.h();
        (0..self.n)
            .map(|j| {
                let up = u[(j + 1) % self.n];
                let um = u[(j + self.n - 1) % self.n];
                (up - um) / (2.0 * h)
            })
            .collect()
    }

    /// Rate of change: -c u_x + nu u_xx with centred periodic differences.
    pub fn rate(&self, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n, "state length must equal n");
        let h = self.h();
        (0..self.n)
            .map(|j| {
                let up = u[(j + 1) % self.n];
                let um = u[(j + self.n - 1) % self.n];
                -self.c * (up - um) / (2.0 * h)
                    + self.nu * (up - 2.0 * u[j] + um) / (h * h)
            })
            .collect()
    }
}
