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

/// RK4 stage structure (nodes 0, 1/2, 1/2, 1) with equal weights 1/4.
/// Second order: sum b_i = 1 and sum b_i c_i = 1/2 hold, but
/// sum b_i c_i^2 = 3/8 != 1/3, so the third-order condition fails.
pub struct RK4Equal;

impl Integrator for RK4Equal {
    fn step<F, S>(&self, f: F, dt: f64, y: &[S]) -> Vec<S>
    where
        F: Fn(&[S]) -> Vec<S>,
        S: Scalar,
    {
        let k1 = f(y);
        let y2: Vec<S> = y
            .iter()
            .zip(&k1)
            .map(|(yi, k1i)| *yi + *k1i * (dt / 2.0))
            .collect();
        let k2 = f(&y2);
        let y3: Vec<S> = y
            .iter()
            .zip(&k2)
            .map(|(yi, k2i)| *yi + *k2i * (dt / 2.0))
            .collect();
        let k3 = f(&y3);
        let y4: Vec<S> = y.iter().zip(&k3).map(|(yi, k3i)| *yi + *k3i * dt).collect();
        let k4 = f(&y4);
        y.iter()
            .zip(k1)
            .zip(k2)
            .zip(k3)
            .zip(k4)
            .map(|((((yi, k1i), k2i), k3i), k4i)| {
                *yi + (k1i + k2i + k3i + k4i) * (dt / 4.0)
            })
            .collect()
    }
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
/// Explicit midpoint (RK2): k1 = f(y); y + dt * f(y + dt/2 k1).
pub struct Midpoint;

impl Integrator for Midpoint {
    fn step<F, S>(&self, f: F, dt: f64, y: &[S]) -> Vec<S>
    where
        F: Fn(&[S]) -> Vec<S>,
        S: Scalar,
    {
        let k1 = f(y);
        let y_mid: Vec<S> = y
            .iter()
            .zip(k1)
            .map(|(yi, k1i)| *yi + k1i * (dt / 2.0))
            .collect();
        let k2 = f(&y_mid);
        y.iter().zip(k2).map(|(yi, k2i)| *yi + k2i * dt).collect()
    }
}
/// Classical fourth-order Runge-Kutta: weights 1/6, 1/3, 1/3, 1/6.
pub struct RK4;

impl Integrator for RK4 {
    fn step<F, S>(&self, f: F, dt: f64, y: &[S]) -> Vec<S>
    where
        F: Fn(&[S]) -> Vec<S>,
        S: Scalar,
    {
        let k1 = f(y);
        let y2: Vec<S> = y
            .iter()
            .zip(&k1)
            .map(|(yi, k1i)| *yi + *k1i * (dt / 2.0))
            .collect();
        let k2 = f(&y2);
        let y3: Vec<S> = y
            .iter()
            .zip(&k2)
            .map(|(yi, k2i)| *yi + *k2i * (dt / 2.0))
            .collect();
        let k3 = f(&y3);
        let y4: Vec<S> = y.iter().zip(&k3).map(|(yi, k3i)| *yi + *k3i * dt).collect();
        let k4 = f(&y4);
        y.iter()
            .zip(k1)
            .zip(k2)
            .zip(k3)
            .zip(k4)
            .map(|((((yi, k1i), k2i), k3i), k4i)| {
                *yi + (k1i + k2i * 2.0 + k3i * 2.0 + k4i) * (dt / 6.0)
            })
            .collect()
    }
}
