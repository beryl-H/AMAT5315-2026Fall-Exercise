//! Week 3: sampling the two-dimensional Ising model along a temperature
//! ramp. Updates: random-site Metropolis sweeps and single-cluster (Wolff)
//! flips.

pub mod cli;
pub mod io;
pub mod lattice;
pub mod metropolis;
pub mod ramp;
pub mod wolff;

pub use cli::{Args, temperature_grid, validate};
pub use io::{RunConfig, write_artifacts};
pub use lattice::Lattice;
pub use metropolis::{SweepStats, should_accept, sweep};
pub use ramp::{SeriesRow, SpinFrame, TemperatureResult, Update, run_ramp};
pub use wolff::{WolffStats, add_probability, cluster_flip};
