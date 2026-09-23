//! Week 4: one `Integrator` trait (forward Euler, explicit midpoint,
//! classical RK4) and two rate functions for the periodic
//! advection-diffusion equation u_t + c u_x = nu u_xx.

pub mod integrators;
pub mod scalar;

pub use integrators::{Euler, Integrator, Midpoint};
pub use scalar::Scalar;
