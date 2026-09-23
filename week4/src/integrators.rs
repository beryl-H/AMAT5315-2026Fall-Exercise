//! One `Integrator` trait, three explicit one-step methods.
//!
//! Integrators only add/subtract vectors element-wise and scale them by
//! real `f64` coefficients (`y * a`); they never multiply two state
//! elements together.

use crate::scalar::Scalar;

/// Advance a state vector one step of size `dt` under rate function `f`.
pub trait Integrator {
    fn step<F, S>(&self, f: F, dt: f64, y: &[S]) -> Vec<S>
    where
        F: Fn(&[S]) -> Vec<S>,
        S: Scalar;
}

/// Forward Euler: y + dt * f(y).
pub struct Euler;

impl Integrator for Euler {
    fn step<F, S>(&self, f: F, dt: f64, y: &[S]) -> Vec<S>
    where
        F: Fn(&[S]) -> Vec<S>,
        S: Scalar,
    {
        let k1 = f(y);
        y.iter().zip(k1).map(|(yi, k1i)| *yi + k1i * dt).collect()
    }
}
