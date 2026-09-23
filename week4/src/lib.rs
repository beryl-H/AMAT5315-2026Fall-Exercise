//! Week 4: one `Integrator` trait (forward Euler, explicit midpoint,
//! classical RK4) and two rate functions for the periodic
//! advection-diffusion equation u_t + c u_x = nu u_xx.

pub mod fd;
pub mod field;
pub mod fourier;
pub mod integrators;
pub mod scalar;
pub mod stepper;

pub use fd::CentredFD;
pub use field::{
    advdiff_exponential, advdiff_rate, dealias, dealias_band, dft2, deriv_x, deriv_y,
    hermitian_mode, idft2, is_nyquist, laplacian, taylor_green_spectral, wavenumber,
};
pub use fourier::{dft, idft, FourierAdvDiff};
pub use integrators::{Euler, Integrator, Midpoint, RK4, RK4Equal};
pub use scalar::Scalar;
pub use stepper::advance;
