//! Shared integration interface and the two Week 2 integrators.

use crate::state::{State, Vec2};

/// Common interface for advancing a molecular-dynamics state by one step.
pub trait Integrator {
    /// Optional one-time setup before the first step.
    ///
    /// The default does nothing. Velocity-Verlet overrides this to cache the
    /// initial acceleration.
    fn initialize(&mut self, _state: &State, _acceleration: &dyn Fn(&State) -> Vec<Vec2>) {}

    /// Advance `state` by exactly one step of size `dt`.
    fn step(&mut self, state: &mut State, dt: f64, acceleration: &dyn Fn(&State) -> Vec<Vec2>);
}

/// Forward Euler: both updates use values from the old state.
#[derive(Clone, Debug, Default)]
pub struct Euler;

impl Integrator for Euler {
    fn step(&mut self, state: &mut State, dt: f64, acceleration: &dyn Fn(&State) -> Vec<Vec2>) {
        // Cache the old velocities so the position update cannot accidentally
        // use the updated velocity (which would be semi-implicit Euler).
        let v_n = state.velocities.clone();
        let a_n = acceleration(state);

        // x_{n+1} = x_n + v_n * dt
        for (i, position) in state.positions.iter_mut().enumerate() {
            position[0] += dt * v_n[i][0];
            position[1] += dt * v_n[i][1];
        }

        // v_{n+1} = v_n + a_n * dt
        for (velocity, acceleration_i) in state.velocities.iter_mut().zip(a_n.iter()) {
            velocity[0] += dt * acceleration_i[0];
            velocity[1] += dt * acceleration_i[1];
        }
    }
}

/// Velocity-Verlet with the initial acceleration cached before the first step.
#[derive(Clone, Debug, Default)]
pub struct VelocityVerlet {
    previous_acceleration: Option<Vec<Vec2>>,
}

impl Integrator for VelocityVerlet {
    fn initialize(&mut self, state: &State, acceleration: &dyn Fn(&State) -> Vec<Vec2>) {
        self.previous_acceleration = Some(acceleration(state));
    }

    fn step(&mut self, state: &mut State, dt: f64, acceleration: &dyn Fn(&State) -> Vec<Vec2>) {
        let a_n = self
            .previous_acceleration
            .take()
            .expect("VelocityVerlet::initialize must be called before step");
        let half_dt = 0.5 * dt;

        // v_half = v_n + 0.5 * dt * a_n
        let mut v_half = state.velocities.clone();
        for (velocity, a) in v_half.iter_mut().zip(a_n.iter()) {
            velocity[0] += half_dt * a[0];
            velocity[1] += half_dt * a[1];
        }

        // x_{n+1} = x_n + dt * v_half
        for (i, position) in state.positions.iter_mut().enumerate() {
            position[0] += dt * v_half[i][0];
            position[1] += dt * v_half[i][1];
        }

        // a_{n+1} = acceleration(x_{n+1}); one new evaluation per step.
        let a_next = acceleration(state);

        // v_{n+1} = v_half + 0.5 * dt * a_{n+1}
        for (velocity, a) in v_half.iter_mut().zip(a_next.iter()) {
            velocity[0] += half_dt * a[0];
            velocity[1] += half_dt * a[1];
        }

        state.velocities = v_half;
        self.previous_acceleration = Some(a_next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_updates_positions_with_old_velocities() {
        let mut state = State {
            positions: vec![[1.0, 0.0], [0.0, 0.0]],
            velocities: vec![[1.0, 0.0], [0.0, 0.0]],
        };
        let constant_acceleration = |_: &State| vec![[2.0, 0.0], [2.0, 0.0]];
        let mut euler = Euler::default();

        euler.step(&mut state, 1.0, &constant_acceleration);

        // Old-velocity update: x1 = 1 + 1 = 2, not 3.
        assert_eq!(state.positions[0], [2.0, 0.0]);
        assert_eq!(state.velocities[0], [3.0, 0.0]);
    }

    #[test]
    fn velocity_verlet_uses_half_velocity_and_caches_next_acceleration() {
        let mut state = State {
            positions: vec![[1.0, 0.0], [0.0, 0.0]],
            velocities: vec![[1.0, 0.0], [0.0, 0.0]],
        };
        let mut verlet = VelocityVerlet::default();
        let first_acceleration = |_: &State| vec![[2.0, 0.0], [2.0, 0.0]];

        verlet.initialize(&state, &first_acceleration);
        assert!(verlet.previous_acceleration.is_some());

        let second_acceleration = |_: &State| vec![[4.0, 0.0], [4.0, 0.0]];
        verlet.step(&mut state, 1.0, &second_acceleration);

        // v_half = [2, 0], x1 = 1 + 2 = 3, v1 = 2 + 2 = 4.
        assert_eq!(state.positions[0], [3.0, 0.0]);
        assert_eq!(state.velocities[0], [4.0, 0.0]);
        assert!(verlet.previous_acceleration.is_some());
    }
}
