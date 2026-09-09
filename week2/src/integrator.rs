//! Update rules that advance a `System` by one timestep.

use crate::System;

/// An update rule: advances owned particle state by one step of length `dt`.
pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

/// Forward Euler: x_{n+1} = x_n + v_n * dt, v_{n+1} = v_n + a_n * dt,
/// using the accelerations at the current positions.
pub struct Euler;

impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        system.refresh_accelerations(); // a_n at x_n
        let a = system.accelerations.clone(); // snapshot: borrows must end first
        for (x, v) in system.positions.iter_mut().zip(&system.velocities) {
            x[0] += dt * v[0];
            x[1] += dt * v[1];
        }
        for (v, a) in system.velocities.iter_mut().zip(&a) {
            v[0] += dt * a[0];
            v[1] += dt * a[1];
        }
    }
}

/// Velocity-Verlet, one force evaluation per step. The cache holds a_n at x_n
/// (set by `System::new` and refreshed at the end of every step).
pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        let half = dt / 2.0;
        // v_{n+1/2} = v_n + (dt/2) * a_n
        for (v, a) in system.velocities.iter_mut().zip(&system.accelerations) {
            v[0] += half * a[0];
            v[1] += half * a[1];
        }
        // x_{n+1} = x_n + dt * v_{n+1/2}
        for (x, v) in system.positions.iter_mut().zip(&system.velocities) {
            x[0] += dt * v[0];
            x[1] += dt * v[1];
        }
        // a_{n+1} at the new positions, stored for the next step
        system.refresh_accelerations();
        // v_{n+1} = v_{n+1/2} + (dt/2) * a_{n+1}
        for (v, a) in system.velocities.iter_mut().zip(&system.accelerations) {
            v[0] += half * a[0];
            v[1] += half * a[1];
        }
    }
}
