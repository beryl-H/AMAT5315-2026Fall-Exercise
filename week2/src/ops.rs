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
    let mut integrator = VelocityVerlet;

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
