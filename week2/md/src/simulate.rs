//! Equilibration + production driver for the periodic fluid.

use crate::fluid::{fluid_accelerations, fluid_potential_energy};
use crate::integrator::{Integrator, VelocityVerlet};
use crate::state::State;
use crate::system::{Box2, lattice_state, wrap};
use crate::thermostat::{gaussian_velocities, remove_com_velocity, rescale_to};
use crate::Vec2;

/// Number of equilibration steps between thermostat events [Course Req.].
pub const THERMOSTAT_INTERVAL: usize = 50;

/// Run parameters; Default is the course contract run.
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
        }
    }
}

/// One saved production frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    pub e_pot: f64,
    pub e_kin: f64,
}

/// Schedule B thermostat event count (doc/test helper): the initial rescale
/// plus one rescale after every 50th equilibration step. Production never
/// rescales.
pub fn thermostat_events(eq_steps: usize) -> usize {
    1 + eq_steps / THERMOSTAT_INTERVAL
}

/// Run equilibration (thermostat every 50 steps, Schedule B) then
/// thermostat-free production, saving every `sample_every` steps
/// (step 0 never saved).
pub fn run_simulation(config: &SimConfig) -> Vec<Frame> {
    let bx = Box2::new(config.n, config.rho);
    let mut state = lattice_state(config.n, config.rho);
    state.velocities = gaussian_velocities(config.n, config.temperature, config.seed);
    remove_com_velocity(&mut state.velocities);
    rescale_to(&mut state.velocities, config.temperature);

    let accelerations = |s: &State| fluid_accelerations(s, &bx);
    let mut verlet = VelocityVerlet::default();
    verlet.initialize(&state, &accelerations);

    // Equilibration: rescale after each step s with s % 50 == 0.
    for step in 1..=config.eq_steps {
        verlet.step(&mut state, config.dt, &accelerations);
        wrap_state(&mut state, &bx);
        if step % THERMOSTAT_INTERVAL == 0 {
            rescale_to(&mut state.velocities, config.temperature);
        }
    }

    // Production: thermostat OFF, time starts at zero.
    let mut frames = Vec::with_capacity(config.steps / config.sample_every);
    for step in 1..=config.steps {
        verlet.step(&mut state, config.dt, &accelerations);
        wrap_state(&mut state, &bx);
        if step % config.sample_every == 0 {
            frames.push(Frame {
                step,
                t: step as f64 * config.dt,
                pos: state.positions.clone(),
                vel: state.velocities.clone(),
                e_pot: fluid_potential_energy(&state, &bx),
                e_kin: crate::kinetic_energy(&state),
            });
        }
    }
    frames
}

fn wrap_state(state: &mut State, bx: &Box2) {
    for p in state.positions.iter_mut() {
        p[0] = wrap(p[0], bx.lx);
        p[1] = wrap(p[1], bx.ly);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> SimConfig {
        SimConfig {
            n: 16,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 100,
            steps: 200,
            sample_every: 25,
            seed: 2026,
        }
    }

    #[test]
    fn defaults_are_the_course_contract_values() {
        let c = SimConfig::default();
        assert_eq!(c.n, 100);
        assert_eq!(c.rho, 0.8);
        assert_eq!(c.temperature, 0.5);
        assert_eq!(c.dt, 0.01);
        assert_eq!(c.eq_steps, 2000);
        assert_eq!(c.steps, 10000);
        assert_eq!(c.sample_every, 50);
        assert_eq!(c.seed, 2026);
    }

    #[test]
    fn frames_follow_sample_every_and_skip_step_zero() {
        // Course Requirement: frame count = steps/sample_every, saved steps
        // are sample_every, 2*sample_every, ..., steps; production step 0 is
        // never saved; t = step * dt.
        let frames = run_simulation(&small_config());
        assert_eq!(frames.len(), 8);
        for (idx, f) in frames.iter().enumerate() {
            assert_eq!(f.step, (idx + 1) * 25);
            assert!((f.t - f.step as f64 * 0.01).abs() < 1e-12);
        }
        // No production step-0 frame.
        assert!(frames.iter().all(|f| f.step > 0));
    }

    #[test]
    fn positions_stay_wrapped_and_energies_finite() {
        let frames = run_simulation(&small_config());
        let bx = Box2::new(16, 0.8);
        for f in &frames {
            for p in &f.pos {
                assert!((0.0..bx.lx).contains(&p[0]), "x = {}", p[0]);
                assert!((0.0..bx.ly).contains(&p[1]), "y = {}", p[1]);
            }
            assert!(f.e_pot.is_finite());
            assert!(f.e_kin.is_finite());
        }
    }

    #[test]
    fn thermostat_event_count_is_schedule_b() {
        // Course Requirement (Schedule B): initial rescale (event 1) plus one
        // after every 50th equilibration step; none during production.
        assert_eq!(THERMOSTAT_INTERVAL, 50);
        assert_eq!(thermostat_events(2000), 1 + 2000 / 50);
        assert_eq!(thermostat_events(100), 1 + 100 / 50);
        assert_eq!(thermostat_events(0), 1);
    }

    #[test]
    fn com_stays_zero_through_production() {
        // COM is removed exactly once before equilibration and never re-added;
        // velocity-Verlet with antisymmetric pair forces conserves momentum.
        let frames = run_simulation(&small_config());
        for f in &frames {
            let mx = f.vel.iter().map(|v| v[0]).sum::<f64>() / f.vel.len() as f64;
            let my = f.vel.iter().map(|v| v[1]).sum::<f64>() / f.vel.len() as f64;
            assert!(mx.abs() < 1e-10 && my.abs() < 1e-10, "COM ({mx},{my})");
        }
    }

    #[test]
    fn no_thermostat_rescaling_during_production() {
        // If the thermostat were still active every 50 production steps, the
        // saved frames at production steps 50, 100, 150, 200 would have their
        // thermodynamic temperature pinned exactly to config.temperature
        // (rescale sets it to ~1e-16 of target). With the thermostat OFF in
        // production, those temperatures drift away from target.
        let config = small_config();
        let frames = run_simulation(&config);
        let pinned = frames
            .iter()
            .filter(|f| f.step % THERMOSTAT_INTERVAL == 0)
            .all(|f| {
                let t = crate::thermostat::thermodynamic_temperature(&f.vel);
                (t - config.temperature).abs() < 1e-6
            });
        assert!(!pinned, "temperature stayed pinned to target in production");
    }
}
