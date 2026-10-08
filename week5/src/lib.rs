use serde::Deserialize;

pub mod npy;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Deserialize)]
pub struct Experiment {
    pub name: String,
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub shots: Vec<[f64; 2]>,
    pub receivers: Vec<[usize; 2]>,
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub background: Vec<Vec<f64>>,
    pub perturbation: Vec<Vec<f64>>,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
}

impl Experiment {
    pub fn validate(&self) -> Result<()> {
        if self.nx < 3 || self.nz < 3 || self.steps == 0 {
            return Err("grid must be at least 3x3 with a positive number of steps".into());
        }
        for (name, v) in [("dx", self.dx), ("dt", self.dt),
                         ("source_frequency", self.source_frequency),
                         ("length_unit_m", self.length_unit_m), ("time_unit_s", self.time_unit_s)] {
            if !v.is_finite() || v <= 0.0 { return Err(format!("{name} must be finite and positive").into()); }
        }
        if !self.source_peak_time.is_finite() || !self.source_amplitude.is_finite()
            || !self.sponge_strength.is_finite() || self.sponge_strength < 0.0 {
            return Err("source and nonnegative sponge parameters must be finite".into());
        }
        for field in [&self.background, &self.perturbation] {
            if field.len() != self.nz || field.iter().any(|r| r.len() != self.nx) {
                return Err("model arrays must have shape [nz,nx]".into());
            }
            if field.iter().flatten().any(|v| !v.is_finite()) { return Err("model arrays must be finite".into()); }
        }
        if self.background.iter().flatten().any(|&c| c <= 0.0) { return Err("velocity must be positive".into()); }
        let max_c = self.background.iter().flatten().copied().fold(0.0, f64::max);
        if self.dt * max_c / self.dx > 1.0 / 2.0_f64.sqrt() {
            return Err("CFL stability condition dt*max(c)/dx <= 1/sqrt(2) violated".into());
        }
        if self.shots.is_empty() || self.receivers.is_empty() { return Err("survey must have shots and receivers".into()); }
        if self.shots.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()
            || p[0] < 0.0 || p[1] < 0.0 || p[0] > (self.nx - 1) as f64 || p[1] > (self.nz - 1) as f64)
            || self.receivers.iter().any(|p| p[0] >= self.nx || p[1] >= self.nz) {
            return Err("source or receiver outside grid".into());
        }
        Ok(())
    }

    pub fn cells(&self) -> usize { self.nx * self.nz }
    pub fn velocity(&self) -> Vec<f64> { self.background.iter().flatten().copied().collect() }
    pub fn perturbation(&self) -> Vec<f64> { self.perturbation.iter().flatten().copied().collect() }
    pub fn sigma(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.cells()];
        if self.sponge_width == 0 { return out; }
        for z in 0..self.nz {
            for x in 0..self.nx {
                let distance = x.min(self.nx - 1 - x).min(z).min(self.nz - 1 - z) as f64;
                let ramp = (1.0 - distance / self.sponge_width as f64).max(0.0);
                out[z * self.nx + x] = self.sponge_strength * ramp * ramp;
            }
        }
        out
    }
    pub fn footprint(&self, shot: [f64; 2]) -> Vec<f64> {
        let mut out = vec![0.0; self.cells()];
        for z in 0..self.nz {
            for x in 0..self.nx {
                let a = x as f64 - shot[0];
                let b = z as f64 - shot[1];
                out[z * self.nx + x] = (-(a * a + b * b) / 2.0).exp();
            }
        }
        out
    }
    pub fn source(&self, step: usize, footprint: &[f64], out: &mut [f64]) {
        let theta = std::f64::consts::PI * self.source_frequency * (step as f64 * self.dt - self.source_peak_time);
        let pulse = self.source_amplitude * (1.0 - 2.0 * theta * theta) * (-theta * theta).exp();
        for (q, &v) in out.iter_mut().zip(footprint) { *q = pulse * v; }
    }
}

unsafe extern "C" {
    fn enzyme_cube(x: f64, out: *mut f64);
    fn seismic_step(state: *const f64, velocity: *const f64, sigma: *const f64,
                    source: *const f64, dt: f64, dx: f64, nx: usize, nz: usize, out: *mut f64);
}

pub fn smoke() -> [f64; 3] {
    let mut out = [0.0; 3];
    unsafe { enzyme_cube(2.0, out.as_mut_ptr()); }
    out
}

pub fn advance(exp: &Experiment, state: &[f64], velocity: &[f64], sigma: &[f64], source: &[f64], out: &mut [f64]) {
    let n = exp.cells();
    assert_eq!(state.len(), 2 * n); assert_eq!(out.len(), 2 * n);
    assert_eq!(velocity.len(), n); assert_eq!(sigma.len(), n); assert_eq!(source.len(), n);
    // Disjoint slices are enforced by Rust's immutable/mutable borrows.
    unsafe { seismic_step(state.as_ptr(), velocity.as_ptr(), sigma.as_ptr(), source.as_ptr(),
                          exp.dt, exp.dx, exp.nx, exp.nz, out.as_mut_ptr()); }
}
