pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// Lennard-Jones pair energy in reduced units (epsilon = sigma = 1):
/// `4 * (r^-12 - r^-6)`.
pub fn lj_energy(r: f64) -> f64 {
    let inv_r6 = r.powi(-6);
    4.0 * inv_r6 * (inv_r6 - 1.0)
}

/// Lennard-Jones pair force in reduced units:
/// `-d lj_energy / dr = 24 / r * (2 * r^-12 - r^-6)`.
pub fn lj_force(r: f64) -> f64 {
    let inv_r6 = r.powi(-6);
    let inv_r12 = inv_r6 * inv_r6;
    24.0 / r * (2.0 * inv_r12 - inv_r6)
}

mod state;
pub use state::{State, Vec2, two_atom_initial_state};

mod experiment;
pub use experiment::{
    kinetic_energy, lj_accelerations, potential_energy, relative_energy_error, run_experiment,
    total_energy, ExperimentResult,
};

mod integrator;
pub use integrator::{Euler, Integrator, VelocityVerlet};

pub mod cli;
pub use cli::{Cli, Command, RunArgs};

pub mod system;
pub use system::{Box2, lattice_constants, lattice_state, minimum_image, side, wrap};

pub mod pair;
pub use pair::{RC, shifted_energy, shifted_force};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_says_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}
