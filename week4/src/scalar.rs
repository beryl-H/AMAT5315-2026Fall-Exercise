//! Arithmetic abstraction so one integrator code path serves real and
//! complex state vectors.
//!
//! Scaling convention (spec): the only arithmetic integrators perform is
//! element-wise add/subtract and scaling by a real `f64` coefficient
//! written `y * a`; complex multiplication lives in rate functions only.

use std::ops::{Add, Mul, Sub};

pub trait Scalar: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f64, Output = Self> {
    fn zero() -> Self;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl Scalar for num_complex::Complex<f64> {
    fn zero() -> Self {
        num_complex::Complex::new(0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Scalar;
    use num_complex::Complex;

    /// Scale-and-add exactly as the integrators use the trait.
    fn combine<S: Scalar>(a: S, b: S, w: f64) -> S {
        a + b * w
    }

    #[test]
    fn scaling_works_for_real_and_complex() {
        assert_eq!(combine(1.0_f64, 2.0_f64, 0.5), 2.0);
        let c = combine(Complex::new(1.0, -1.0), Complex::new(4.0, 2.0), 0.5);
        assert_eq!(c, Complex::new(3.0, 0.0));
    }

    #[test]
    fn zero_is_zero_for_both() {
        assert_eq!(<f64 as Scalar>::zero(), 0.0);
        assert_eq!(<Complex<f64> as Scalar>::zero(), Complex::new(0.0, 0.0));
    }
}
