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

pub mod fluid;
pub use fluid::{fluid_accelerations, fluid_potential_energy};

mod thermostat;
pub use thermostat::{gaussian_velocities, remove_com_velocity, rescale_to, thermodynamic_temperature};

pub mod simulate;
pub use simulate::{Frame, SimConfig, THERMOSTAT_INTERVAL, run_simulation, thermostat_events};

pub mod io;
pub use io::{RunConfig, read_artifacts, write_artifacts};

mod metrics;
pub use metrics::{bin_edges, chi2_22, frame_total_energies, pooled_speeds, radial_distribution, secular_drift, t_speed};

pub mod checker;
pub use checker::{CheckError, CheckReport, check_artifacts, run_check};

mod render;
pub use render::{Canvas, render_frame};

pub mod video;
pub use video::{encode_video, ffmpeg_available};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_says_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}
