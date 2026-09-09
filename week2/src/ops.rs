//! The three operations behind the md subcommands.

use crate::cli::RunCfg;
use crate::rng::Rng;
use crate::trajectory::{Meta, Trajectory};
use crate::{triangular_lattice, Integrator, System, VelocityVerlet};

/// Number of atoms: fixed by spec (10x10 lattice).
const N_SIDE: usize = 10;
/// Shifted-force cutoff.
const RC: f64 = 2.5;

/// Maxwell-Boltzmann velocities at temperature `t`, rescaled to exactly `t`.
fn maxwell_velocities(n: usize, t: f64, rng: &mut Rng) -> Vec<[f64; 2]> {
    let mut v: Vec<[f64; 2]> = (0..n)
        .map(|_| [rng.gaussian() * t.sqrt(), rng.gaussian() * t.sqrt()])
        .collect();
    rescale(&mut v, t);
    v
}

/// Rescale velocities so the instantaneous temperature hits exactly `t`
/// (2D: T = sum |v|^2 / (2N)).
fn rescale(v: &mut [[f64; 2]], t: f64) {
    let k: f64 = v.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum();
    let factor = (2.0 * t * v.len() as f64 / k).sqrt();
    for a in v {
        a[0] *= factor;
        a[1] *= factor;
    }
}

pub fn run_sim(cfg: &RunCfg) -> Result<String, String> {
    let (positions, box_l) = triangular_lattice(N_SIDE, 0.8);
    let mut rng = Rng::new(cfg.seed);
    let velocities = maxwell_velocities(positions.len(), cfg.temp, &mut rng);
    let mut system = System::periodic(positions, velocities, box_l, RC);
    let integrator = VelocityVerlet;

    // Equilibration: velocity rescaling every step at the target temperature.
    for _ in 0..cfg.equil {
        integrator.step(&mut system, cfg.dt);
        rescale(&mut system.velocities, cfg.temp);
    }

    // Recording: pure NVE, one frame per step.
    let mut frames = Vec::with_capacity(cfg.steps);
    for _ in 0..cfg.steps {
        integrator.step(&mut system, cfg.dt);
        frames.push(
            system
                .positions
                .iter()
                .zip(&system.velocities)
                .map(|(x, v)| [x[0], x[1], v[0], v[1]])
                .collect(),
        );
    }
    let t = Trajectory {
        meta: Meta {
            n: system.n_atoms(),
            box_l,
            temp: cfg.temp,
            dt: cfg.dt,
        },
        frames,
    };
    crate::trajectory::write(&cfg.out, &t).map_err(|e| e.to_string())?;
    Ok(format!(
        "wrote {} frames (N = {}, box = {box_l:.4}) to {}",
        t.frames.len(),
        t.meta.n,
        cfg.out
    ))
}

use crate::cli::CheckCfg;
use crate::trajectory;

/// Outcome of the three physics checks on one trajectory.
pub struct Report {
    pub mean_temp: f64,
    pub drift: f64,
    pub ks: f64,
    pub pass: bool,
    pub target_temp: f64,
}

impl std::fmt::Debug for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Report {{ mean_temp: {:.3}, drift: {:.2e}, ks: {:.4}, pass: {} }}",
            self.mean_temp, self.drift, self.ks, self.pass
        )
    }
}

/// Total energy of one saved frame (shifted-force pairs + kinetic).
fn frame_energy(meta: &trajectory::Meta, frame: &[[f64; 4]]) -> f64 {
    let positions: Vec<[f64; 2]> = frame.iter().map(|a| [a[0], a[1]]).collect();
    let velocities: Vec<[f64; 2]> = frame.iter().map(|a| [a[2], a[3]]).collect();
    System::periodic(positions, velocities, meta.box_l, RC).total_energy()
}

/// Kolmogorov-Smirnov statistic of speeds vs the 2D Maxwell-Boltzmann CDF
/// F(v) = 1 - exp(-v^2 / (2 T)).
fn ks_statistic(speeds: &mut [f64], temp: f64) -> f64 {
    speeds.sort_by(|a, b| a.total_cmp(b));
    let n = speeds.len() as f64;
    speeds
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let cdf = 1.0 - (-v * v / (2.0 * temp)).exp();
            ((i as f64 + 1.0) / n - cdf).abs().max(cdf - i as f64 / n)
        })
        .fold(0.0_f64, f64::max)
}

pub fn check(cfg: &CheckCfg) -> Result<Report, String> {
    let t = trajectory::read(&cfg.file)?;
    if t.frames.is_empty() {
        return Err("trajectory has no frames".into());
    }
    let n = t.meta.n as f64;

    // Temperature per frame from raw velocities.
    let temps: Vec<f64> = t
        .frames
        .iter()
        .map(|f| f.iter().map(|a| a[2] * a[2] + a[3] * a[3]).sum::<f64>() / (2.0 * n))
        .collect();
    let mean_temp = temps.iter().sum::<f64>() / temps.len() as f64;

    // Energy drift: least-squares slope of E(t), scaled by duration and |E0|.
    let energies: Vec<f64> = t.frames.iter().map(|f| frame_energy(&t.meta, f)).collect();
    let e0 = energies[0];
    let duration = (t.frames.len() - 1) as f64 * t.meta.dt;
    let tm = duration / 2.0;
    let em = energies.iter().sum::<f64>() / energies.len() as f64;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (k, e) in energies.iter().enumerate() {
        let tk = k as f64 * t.meta.dt;
        cov += (tk - tm) * (e - em);
        var += (tk - tm) * (tk - tm);
    }
    let slope = cov / var;
    let drift = (slope * duration / e0.abs()).abs();

    // Speed distribution vs 2D Maxwell-Boltzmann at the mean temperature.
    let mut speeds: Vec<f64> = t
        .frames
        .iter()
        .flat_map(|f| f.iter().map(|a| (a[2] * a[2] + a[3] * a[3]).sqrt()))
        .collect();
    let ks = ks_statistic(&mut speeds, mean_temp);

    let pass = (mean_temp - t.meta.temp).abs() <= cfg.temp_tol * t.meta.temp
        && drift < cfg.drift_tol
        && ks < cfg.ks_tol;
    Ok(Report {
        mean_temp,
        drift,
        ks,
        pass,
        target_temp: t.meta.temp,
    })
}

/// The exact check output lines printed by `md check`.
pub fn format_report(r: &Report, cfg: &CheckCfg) -> String {
    let temp_ok = (r.mean_temp - r.target_temp).abs() <= cfg.temp_tol * r.target_temp;
    let mut s = String::new();
    s.push_str(&format!(
        "temperature: {:.3} (target {:.3}, tol {:.2}) ... {}\n",
        r.mean_temp,
        r.target_temp,
        cfg.temp_tol,
        if temp_ok { "PASS" } else { "FAIL" }
    ));
    s.push_str(&format!(
        "energy drift: {:.3e} (tol {:.1e}) ... {}\n",
        r.drift,
        cfg.drift_tol,
        if r.drift < cfg.drift_tol { "PASS" } else { "FAIL" }
    ));
    s.push_str(&format!(
        "speed distribution KS: {:.4} (tol {:.3}) ... {}\n",
        r.ks,
        cfg.ks_tol,
        if r.ks < cfg.ks_tol { "PASS" } else { "FAIL" }
    ));
    s.push_str(&format!("verdict: {}\n", if r.pass { "PASS" } else { "FAIL" }));
    s
}
