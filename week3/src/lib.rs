//! Week 3: sampling the two-dimensional Ising model along a temperature
//! ramp. Part 1 implements the random-site Metropolis update.

pub mod cli;
pub mod io;
pub mod lattice;
pub mod metropolis;
pub mod ramp;

pub use cli::{Args, temperature_grid, validate};
pub use io::{RunConfig, write_artifacts};
pub use lattice::Lattice;
pub use metropolis::{SweepStats, should_accept, sweep};
pub use ramp::{SeriesRow, SpinFrame, TemperatureResult, run_ramp};
