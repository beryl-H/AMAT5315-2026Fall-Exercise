# Week 4 Integrators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust library `advdiff` with one `Integrator` trait (forward Euler, explicit midpoint, classical RK4) and two rate functions (Fourier pseudo-spectral and centred finite differences) for the periodic advection–diffusion equation, tested against exact single-wave solutions.

**Architecture:** A small `Scalar` arithmetic trait lets one generic `step` method serve real (`f64`) and complex (`Complex<f64>`) state vectors. Rate functions are plain structs exposing `rate` (and bare `derivative`); the Fourier scheme zeroes the first-derivative coefficient of the Nyquist index. Integration tests compare one `step` against closed-form single-wave references, so all measured error is temporal.

**Tech Stack:** Rust 2021 (rustc 1.75), `num-complex = "0.4"`, built-in `cargo test`.

**Spec:** `week4/SPEC.md` (design contract; read together with this plan).

## Global Constraints

- Crate root is `week4/`; package name `advdiff`, edition 2021; toolchain rustc/cargo 1.75.
- Only external dependency: `num-complex = "0.4"`.
- Integrators perform only element-wise add/subtract and real scaling written `y * a` (`Mul<f64, Output = Self>`); no state element multiplication inside integrators (spec: Scaling convention).
- Spectral state has length `n`, indices `0..n-1`; for even `n` the Nyquist index is `j = n/2`, wavenumber magnitude `n/2`, first-derivative coefficient zero (spec: Fourier state indexing).
- Verification command: `cargo test` run from `week4/`; output must be green and warning-free.
- History order is strict for every task: SPEC → failing test (RED commit) → implementation → passing test (GREEN commit). No implementation is written before its test has been watched failing. In Rust a compile failure on an unresolved (not-yet-existing) API is the expected RED for a missing feature.
- Commit style follows repo history: `test: ... (red)` then `feat: ... (green)`.
- Any run/comparison/plot script goes in `week4/scripts/` and is run from there.

---

### Task 1: Crate scaffold and `Scalar` trait

**Files:**
- Create: `week4/Cargo.toml`
- Create: `week4/src/lib.rs`
- Create: `week4/src/scalar.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `advdiff::scalar::Scalar` with bounds `Copy + Add<Output> + Sub<Output> + Mul<f64, Output>` and `fn zero() -> Self`, implemented for `f64` and `num_complex::Complex<f64>`; `advdiff::{Scalar}` re-export.

- [ ] **Step 1: Write the failing test (unit tests inside the module)**

Create `week4/src/scalar.rs` containing only the test module (the trait does not exist yet, so this fails to compile — that is the RED):

```rust
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
```

Create `week4/src/lib.rs` (module declared but the file must compile only after Step 3 — the RED run in Step 2 is expected to fail on the missing trait):

```rust
//! Week 4: one `Integrator` trait (forward Euler, explicit midpoint,
//! classical RK4) and two rate functions for the periodic
//! advection-diffusion equation u_t + c u_x = nu u_xx.

pub mod scalar;

pub use scalar::Scalar;
```

Create `week4/Cargo.toml`:

```toml
[package]
name = "advdiff"
version = "0.1.0"
edition = "2021"

[dependencies]
num-complex = "0.4"
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test` from `week4/`.
Expected: compile error — `Scalar` not found / not defined (feature missing, not a typo).

- [ ] **Step 3: Write minimal implementation**

Prepend to `week4/src/scalar.rs`, above the test module:

```rust
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
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `cargo test` from `week4/`.
Expected: 2 tests pass, no warnings.

- [ ] **Step 5: Commit**

```bash
git add week4/Cargo.toml week4/src/lib.rs week4/src/scalar.rs
git commit -m "feat: scaffold advdiff crate with Scalar arithmetic trait (green)"
```

### Task 2: `Integrator` trait and forward Euler

**Files:**
- Create: `week4/src/integrators.rs`
- Modify: `week4/src/lib.rs` (declare + re-export module)
- Test: `week4/tests/contract.rs` (new)

**Interfaces:**
- Consumes: `advdiff::Scalar`.
- Produces: `trait Integrator { fn step<F, S>(&self, f: F, dt: f64, y: &[S]) -> Vec<S> where F: Fn(&[S]) -> Vec<S>, S: Scalar; }`; struct `Euler` implementing it; re-exports `advdiff::{Euler, Integrator}`.

- [ ] **Step 1: Write the failing test**

Create `week4/tests/contract.rs`:

```rust
//! Contract tests for the Week 4 integrators and rate functions.

use advdiff::{Euler, Integrator};
use num_complex::Complex;

/// dy/dt = -y, exact one-step polynomials are known in closed form.
fn decay(y: &[f64]) -> Vec<f64> {
    y.iter().map(|v| -v).collect()
}

#[test]
fn euler_one_step_matches_exact_polynomial() {
    let dt = 0.1;
    let y = Euler.step(decay, dt, &[1.0]);
    let expected = 1.0 - dt;
    assert!((y[0] - expected).abs() < 1e-12, "got {}, want {}", y[0], expected);
}

#[test]
fn euler_works_for_complex_state() {
    let y = Euler.step(
        |s: &[Complex<f64>]| s.iter().map(|v| -*v).collect(),
        0.1,
        &[Complex::new(1.0, 2.0)],
    );
    assert!((y[0] - Complex::new(0.9, 1.8)).norm() < 1e-12);
}
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test` from `week4/`.
Expected: compile error — `Euler` / `Integrator` unresolved imports.

- [ ] **Step 3: Commit the failing test**

```bash
git add week4/tests/contract.rs
git commit -m "test: add failing Integrator trait and forward Euler tests (red)"
```

- [ ] **Step 4: Write minimal implementation**

Create `week4/src/integrators.rs`:

```rust
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
```

Modify `week4/src/lib.rs`:

```rust
pub mod integrators;
pub mod scalar;

pub use integrators::{Euler, Integrator};
pub use scalar::Scalar;
```

- [ ] **Step 5: Run tests to verify GREEN**

Run: `cargo test` from `week4/`.
Expected: all tests pass (scalar 2 + contract 2), no warnings.

- [ ] **Step 6: Commit**

```bash
git add week4/src/integrators.rs week4/src/lib.rs
git commit -m "feat: implement Integrator trait and forward Euler (green)"
```

### Task 3: Explicit midpoint (RK2)

**Files:**
- Modify: `week4/src/integrators.rs`
- Modify: `week4/src/lib.rs` (re-export)
- Test: `week4/tests/contract.rs`

**Interfaces:**
- Consumes: `Integrator`, `Scalar`.
- Produces: struct `Midpoint` implementing `Integrator`; re-export `advdiff::Midpoint`.

- [ ] **Step 1: Write the failing test**

Append to `week4/tests/contract.rs`:

```rust
use advdiff::Midpoint;

#[test]
fn midpoint_one_step_matches_exact_polynomial() {
    let dt = 0.1;
    let y = Midpoint.step(decay, dt, &[1.0]);
    let expected = 1.0 - dt + dt * dt / 2.0;
    assert!((y[0] - expected).abs() < 1e-12, "got {}, want {}", y[0], expected);
}
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test` from `week4/`.
Expected: compile error — `Midpoint` unresolved.

- [ ] **Step 3: Commit the failing test**

```bash
git add week4/tests/contract.rs
git commit -m "test: add failing explicit midpoint tests (red)"
```

- [ ] **Step 4: Write minimal implementation**

Append to `week4/src/integrators.rs`:

```rust
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
```

In `week4/src/lib.rs` change the integrators re-export line to:

```rust
pub use integrators::{Euler, Integrator, Midpoint};
```

- [ ] **Step 5: Run tests to verify GREEN**

Run: `cargo test` from `week4/`.
Expected: all tests pass, no warnings.

- [ ] **Step 6: Commit**

```bash
git add week4/src/integrators.rs week4/src/lib.rs
git commit -m "feat: implement explicit midpoint integrator (green)"
```

### Task 4: Classical RK4

**Files:**
- Modify: `week4/src/integrators.rs`
- Modify: `week4/src/lib.rs` (re-export)
- Test: `week4/tests/contract.rs`

**Interfaces:**
- Consumes: `Integrator`, `Scalar`.
- Produces: struct `RK4` implementing `Integrator`; re-export `advdiff::RK4`.

- [ ] **Step 1: Write the failing test**

Append to `week4/tests/contract.rs`:

```rust
use advdiff::RK4;

#[test]
fn rk4_one_step_matches_exact_polynomial() {
    let dt = 0.1;
    let y = RK4.step(decay, dt, &[1.0]);
    let expected = 1.0 - dt + dt * dt / 2.0 - dt.powi(3) / 6.0 + dt.powi(4) / 24.0;
    assert!((y[0] - expected).abs() < 1e-12, "got {}, want {}", y[0], expected);
}
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test` from `week4/`.
Expected: compile error — `RK4` unresolved.

- [ ] **Step 3: Commit the failing test**

```bash
git add week4/tests/contract.rs
git commit -m "test: add failing classical RK4 tests (red)"
```

- [ ] **Step 4: Write minimal implementation**

Append to `week4/src/integrators.rs`:

```rust
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
```

In `week4/src/lib.rs` change the integrators re-export line to:

```rust
pub use integrators::{Euler, Integrator, Midpoint, RK4};
```

- [ ] **Step 5: Run tests to verify GREEN**

Run: `cargo test` from `week4/`.
Expected: all tests pass, no warnings.

- [ ] **Step 6: Commit**

```bash
git add week4/src/integrators.rs week4/src/lib.rs
git commit -m "feat: implement classical fourth-order Runge-Kutta (green)"
```

### Task 5: Fourier rate function with Nyquist rule, plus DFT helpers

**Files:**
- Create: `week4/src/fourier.rs`
- Modify: `week4/src/lib.rs`
- Test: `week4/tests/contract.rs`

**Interfaces:**
- Consumes: `num_complex::Complex`.
- Produces:
  - `struct FourierAdvDiff { pub c: f64, pub nu: f64, pub n: usize }`
  - `fn wavenumber(&self, j: usize) -> f64` (index -> wavenumber, Nyquist magnitude `n/2`)
  - `fn derivative(&self, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>>` (Nyquist coefficient zero)
  - `fn rate(&self, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>>` (`-nu (n/2)^2 u_hat` at Nyquist)
  - free functions `fn dft(u: &[f64]) -> Vec<Complex<f64>>`, `fn idft(u_hat: &[Complex<f64>]) -> Vec<f64>` (conventions `u(x) = sum_k u_hat[k] e^{ikx}`, `u_hat[k] = (1/n) sum_j u_j e^{-ikx}`)
  - re-exports `advdiff::{FourierAdvDiff, dft, idft}`.

- [ ] **Step 1: Write the failing test**

Append to `week4/tests/contract.rs`:

```rust
use advdiff::{dft, FourierAdvDiff, idft};

fn fourier_scheme() -> FourierAdvDiff {
    FourierAdvDiff { c: 1.0, nu: 0.1, n: 32 }
}

#[test]
fn fourier_derivative_of_constant_field_is_zero() {
    let scheme = fourier_scheme();
    // Constant field: only the k = 0 mode is excited; i * 0 = 0.
    let mut u_hat = vec![Complex::new(0.0, 0.0); scheme.n];
    u_hat[0] = Complex::new(3.7, 0.0);
    assert!(scheme.derivative(&u_hat).iter().all(|&d| d == Complex::new(0.0, 0.0)));
}

#[test]
fn fourier_derivative_of_single_mode_is_ik() {
    let scheme = fourier_scheme();
    let mut u_hat = vec![Complex::new(0.0, 0.0); scheme.n];
    u_hat[2] = Complex::new(1.0, 0.0);
    let d = scheme.derivative(&u_hat);
    // d/dx e^{i k x} = i k e^{i k x}, k = 2
    assert!((d[2] - Complex::new(0.0, 2.0)).norm() < 1e-14);
    assert!(d.iter().enumerate().all(|(j, &v)| j == 2 || v.norm() < 1e-14));
}

#[test]
fn fourier_rate_of_single_mode_is_neg_ick_neg_nu_k2() {
    let scheme = fourier_scheme();
    let mut u_hat = vec![Complex::new(0.0, 0.0); scheme.n];
    u_hat[2] = Complex::new(1.0, 0.0);
    let r = scheme.rate(&u_hat);
    // (-i c k - nu k^2) u = (-2i - 0.4) at k = 2
    assert!((r[2] - Complex::new(-0.4, -2.0)).norm() < 1e-14);
}

#[test]
fn nyquist_mode_does_not_travel_only_diffuses() {
    let scheme = fourier_scheme(); // n = 32, Nyquist index 16
    let nyq = scheme.n / 2;
    let mut u_hat = vec![Complex::new(0.0, 0.0); scheme.n];
    u_hat[nyq] = Complex::new(1.5, -0.5);
    let r = scheme.rate(&u_hat);
    // No advective (imaginary) part: exactly -nu (n/2)^2 * u_hat.
    assert_eq!(r[nyq].im, 0.0, "Nyquist mode must not travel");
    let expected_re = -scheme.nu * (nyq as f64) * (nyq as f64) * u_hat[nyq].re;
    assert!((r[nyq].re - expected_re).abs() < 1e-12);
}

#[test]
fn dft_idft_round_trip_recovers_sine() {
    let n = 32usize;
    let k0 = 2.0;
    let u: Vec<f64> = (0..n)
        .map(|j| {
            let x = std::f64::consts::TAU * j as f64 / n as f64;
            (k0 * x).sin()
        })
        .collect();
    let u_hat = dft(&u);
    // sin(k0 x) = (e^{i k0 x} - e^{-i k0 x}) / (2i): u_hat[k0] = -i/2, u_hat[n - k0] = i/2
    assert!((u_hat[2] - Complex::new(0.0, -0.5)).norm() < 1e-12);
    assert!((u_hat[30] - Complex::new(0.0, 0.5)).norm() < 1e-12);
    let back = idft(&u_hat);
    for (a, b) in back.iter().zip(&u) {
        assert!((a - b).abs() < 1e-12, "round trip mismatch {} vs {}", a, b);
    }
}
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test` from `week4/`.
Expected: compile error — `FourierAdvDiff`, `dft`, `idft` unresolved.

- [ ] **Step 3: Commit the failing test**

```bash
git add week4/tests/contract.rs
git commit -m "test: add failing Fourier rate function and Nyquist rule tests (red)"
```

- [ ] **Step 4: Write minimal implementation**

Create `week4/src/fourier.rs`:

```rust
//! Fourier pseudo-spectral rate function for u_t + c u_x = nu u_xx on the
//! periodic grid x_j = 2 pi j / n, j = 0..n-1.
//!
//! Spectral state has length n, indices 0..n-1. Wavenumbers: k_j = j for
//! 0 <= j < n/2; for even n the Nyquist index j = n/2 has wavenumber
//! magnitude n/2 with first-derivative coefficient zero (it does not
//! travel); k_j = j - n for n/2 < j <= n-1. For odd n there is no Nyquist
//! index and the split is at (n-1)/2.

use num_complex::Complex;
use std::f64::consts::TAU;

pub struct FourierAdvDiff {
    pub c: f64,
    pub nu: f64,
    pub n: usize,
}

impl FourierAdvDiff {
    /// Wavenumber of spectral index `j`. For even `n` the Nyquist index
    /// `j = n/2` returns magnitude `n/2` (used only for the diffusion term).
    pub fn wavenumber(&self, j: usize) -> f64 {
        if j <= self.n / 2 {
            j as f64
        } else {
            j as f64 - self.n as f64
        }
    }

    fn is_nyquist(&self, j: usize) -> bool {
        self.n % 2 == 0 && j == self.n / 2
    }

    /// First spectral derivative d/dx. The Nyquist index contributes zero.
    pub fn derivative(&self, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
        assert_eq!(u_hat.len(), self.n, "state length must equal n");
        u_hat.iter()
            .enumerate()
            .map(|(j, uj)| {
                let coeff = if self.is_nyquist(j) {
                    Complex::new(0.0, 0.0)
                } else {
                    Complex::new(0.0, 1.0) * self.wavenumber(j)
                };
                coeff * uj
            })
            .collect()
    }

    /// Rate of change du_hat/dt = (-i c k - nu k^2) u_hat, with the Nyquist
    /// index advecting nothing: rate = -nu (n/2)^2 u_hat there.
    pub fn rate(&self, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>> {
        assert_eq!(u_hat.len(), self.n, "state length must equal n");
        u_hat.iter()
            .enumerate()
            .map(|(j, uj)| {
                let k = self.wavenumber(j);
                let adv = if self.is_nyquist(j) {
                    Complex::new(0.0, 0.0)
                } else {
                    Complex::new(0.0, -self.c * k)
                };
                let diff = Complex::new(-self.nu * k * k, 0.0);
                (adv + diff) * uj
            })
            .collect()
    }
}

/// Discrete Fourier transform, convention u(x) = sum_k u_hat[k] e^{ikx}:
/// u_hat[k] = (1/n) sum_j u_j e^{-i k x_j}.
pub fn dft(u: &[f64]) -> Vec<Complex<f64>> {
    let n = u.len();
    (0..n)
        .map(|k| {
            let mut acc = Complex::new(0.0, 0.0);
            for (j, &uj) in u.iter().enumerate() {
                let x = TAU * j as f64 / n as f64;
                acc += uj * Complex::new(0.0, -(k as f64) * x).exp();
            }
            acc / n as f64
        })
        .collect()
}

/// Inverse discrete Fourier transform back to physical space (real part,
/// exact for input spectra coming from `dft` of a real field).
pub fn idft(u_hat: &[Complex<f64>]) -> Vec<f64> {
    let n = u_hat.len();
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            let mut acc = Complex::new(0.0, 0.0);
            for (k, &uk) in u_hat.iter().enumerate() {
                acc += uk * Complex::new(0.0, (k as f64) * x).exp();
            }
            acc.re
        })
        .collect()
}
```

Modify `week4/src/lib.rs`:

```rust
pub mod fourier;
pub mod integrators;
pub mod scalar;

pub use fourier::{dft, idft, FourierAdvDiff};
pub use integrators::{Euler, Integrator, Midpoint, RK4};
pub use scalar::Scalar;
```

- [ ] **Step 5: Run tests to verify GREEN**

Run: `cargo test` from `week4/`.
Expected: all tests pass, no warnings.

- [ ] **Step 6: Commit**

```bash
git add week4/src/fourier.rs week4/src/lib.rs
git commit -m "feat: implement Fourier advection-diffusion rate with zeroed Nyquist derivative (green)"
```

### Task 6: Remaining tests — FD unit tests, exact single-wave acceptance, convergence order (all RED)

**Files:**
- Test: `week4/tests/contract.rs`

**Interfaces:**
- Consumes: `Euler`, `Midpoint`, `RK4`, `Integrator`, `FourierAdvDiff`, `dft`, `idft` (already green from Tasks 1–5) and `advdiff::CentredFD`, which does not exist yet — the unresolved import is what makes every new test in this commit fail.
- Produces: the complete contract test suite; nothing new in `src/`.

- [ ] **Step 1: Write the failing tests**

Append to `week4/tests/contract.rs` — first the centred-difference unit tests:

```rust
use advdiff::CentredFD;
use std::f64::consts::TAU;

fn fd_scheme() -> CentredFD {
    CentredFD { c: 1.0, nu: 0.1, n: 32 }
}

/// Grid values of cos(k0 x), k0 = 2.
fn cos_wave(n: usize, k0: f64) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            (k0 * x).cos()
        })
        .collect()
}

#[test]
fn fd_derivative_of_constant_field_is_zero() {
    let scheme = fd_scheme();
    let u = vec![3.7; scheme.n];
    assert!(scheme.derivative(&u).iter().all(|&d| d == 0.0));
    assert!(scheme.rate(&u).iter().all(|&r| r.abs() < 1e-15));
}

#[test]
fn fd_derivative_of_single_mode_matches_modified_wavenumber() {
    let scheme = fd_scheme();
    let h = TAU / scheme.n as f64;
    let k0 = 2.0;
    let u = cos_wave(scheme.n, k0);
    let d = scheme.derivative(&u);
    // D cos(k0 x_j) = -(sin(k0 h)/h) sin(k0 x_j) exactly for centred differences.
    for (j, &dj) in d.iter().enumerate() {
        let x = TAU * j as f64 / scheme.n as f64;
        let expected = -(k0 * h).sin() / h * (k0 * x).sin();
        assert!((dj - expected).abs() < 1e-12, "j = {}: {} vs {}", j, dj, expected);
    }
}

#[test]
fn fd_rate_matches_modified_wavenumber_eigenvalues() {
    let scheme = fd_scheme();
    let h = TAU / scheme.n as f64;
    let k0 = 2.0;
    let amp = 1.3;
    let mut u = cos_wave(scheme.n, k0);
    for v in u.iter_mut() {
        *v *= amp;
    }
    let r = scheme.rate(&u);
    // Advective eigenvalue -i c sin(k0 h)/h, diffusive -4 nu sin^2(k0 h/2)/h^2.
    let gamma = -4.0 * scheme.nu * (0.5 * k0 * h).sin().powi(2) / (h * h);
    let omega = scheme.c * (k0 * h).sin() / h;
    for (j, &rj) in r.iter().enumerate() {
        let x = TAU * j as f64 / scheme.n as f64;
        let expected = amp * (gamma * (k0 * x).cos() + omega * (k0 * x).sin());
        assert!((rj - expected).abs() < 1e-12, "j = {}: {} vs {}", j, rj, expected);
    }
}
```

Then the exact single-wave acceptance tests and the convergence-order test:

```rust
/// Max abs difference between two vectors.
fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Max abs difference between two complex vectors.
fn max_err_c(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
}

/// Initial spectrum of sin(k0 x): u_hat[k0] = -i/2, u_hat[n - k0] = i/2.
fn sine_spectrum(n: usize, k0: usize) -> Vec<Complex<f64>> {
    let mut u_hat = vec![Complex::new(0.0, 0.0); n];
    u_hat[k0] = Complex::new(0.0, -0.5);
    u_hat[n - k0] = Complex::new(0.0, 0.5);
    u_hat
}

/// Exact one-step reference for the Fourier scheme: each mode grows by
/// exp((-i c k - nu k^2) dt) (single wave, spatially exact).
fn fourier_reference(scheme: &FourierAdvDiff, u_hat: &[Complex<f64>], dt: f64) -> Vec<Complex<f64>> {
    u_hat.iter()
        .enumerate()
        .map(|(j, &u)| {
            let k = scheme.wavenumber(j);
            let lambda = Complex::new(-scheme.nu * k * k, -scheme.c * k);
            u * (lambda * dt).exp()
        })
        .collect()
}

/// Grid values of sin(k0 x).
fn sine_wave(n: usize, k0: f64) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let x = TAU * j as f64 / n as f64;
            (k0 * x).sin()
        })
        .collect()
}

/// Exact one-step reference for the centred-difference scheme: the
/// semi-discrete single wave travels with modified speed c sin(k0 h)/h and
/// decays at rate 4 nu sin^2(k0 h/2)/h^2.
fn fd_reference(scheme: &CentredFD, dt: f64, k0: f64) -> Vec<f64> {
    let h = TAU / scheme.n as f64;
    let damp = (-4.0 * scheme.nu * (0.5 * k0 * h).sin().powi(2) * dt / (h * h)).exp();
    let speed = scheme.c * (k0 * h).sin() / h;
    (0..scheme.n)
        .map(|j| {
            let x = TAU * j as f64 / scheme.n as f64;
            damp * (k0 * x - speed * dt).sin()
        })
        .collect()
}

fn step_fourier<I: Integrator>(int: &I, dt: f64) -> f64 {
    let scheme = fourier_scheme();
    let u_hat = sine_spectrum(scheme.n, 2);
    let y1 = int.step(|s| scheme.rate(s), dt, &u_hat);
    max_err_c(&y1, &fourier_reference(&scheme, &u_hat, dt))
}

fn step_fd<I: Integrator>(int: &I, dt: f64) -> f64 {
    let scheme = fd_scheme();
    let u = sine_wave(scheme.n, 2.0);
    let y1 = int.step(|s| scheme.rate(s), dt, &u);
    max_err(&y1, &fd_reference(&scheme, dt, 2.0))
}

#[test]
fn one_step_fourier_euler_within_bound() {
    assert!(step_fourier(&Euler, 0.05) < 1e-2);
}

#[test]
fn one_step_fourier_midpoint_within_bound() {
    assert!(step_fourier(&Midpoint, 0.05) < 1e-3);
}

#[test]
fn one_step_fourier_rk4_within_bound() {
    assert!(step_fourier(&RK4, 0.05) < 1e-6);
}

#[test]
fn one_step_fd_euler_within_bound() {
    assert!(step_fd(&Euler, 0.05) < 1e-2);
}

#[test]
fn one_step_fd_midpoint_within_bound() {
    assert!(step_fd(&Midpoint, 0.05) < 1e-3);
}

#[test]
fn one_step_fd_rk4_within_bound() {
    assert!(step_fd(&RK4, 0.05) < 1e-6);
}

/// One-step error must shrink at the method's order when dt halves.
#[test]
fn convergence_order_on_fourier_scheme() {
    let dt = 0.05;
    let e_euler = step_fourier(&Euler, dt) / step_fourier(&Euler, dt / 2.0);
    let e_mid = step_fourier(&Midpoint, dt) / step_fourier(&Midpoint, dt / 2.0);
    let e_rk4 = step_fourier(&RK4, dt) / step_fourier(&RK4, dt / 2.0);
    assert!(e_euler > 1.7 && e_euler < 2.3, "Euler ratio {}", e_euler);
    assert!(e_mid > 3.5 && e_mid < 4.5, "Midpoint ratio {}", e_mid);
    assert!(e_rk4 > 14.0 && e_rk4 < 18.0, "RK4 ratio {}", e_rk4);
}
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test` from `week4/`.
Expected: compile error — `CentredFD` unresolved (feature missing). Every test in this commit depends on the not-yet-existing type, so the whole suite cannot compile.

- [ ] **Step 3: Commit the failing tests**

```bash
git add week4/tests/contract.rs
git commit -m "test: add failing centred-difference, single-wave acceptance, and convergence-order tests (red)"
```

### Task 7: Centred finite-difference rate function (GREEN for the whole suite)

**Files:**
- Create: `week4/src/fd.rs`
- Modify: `week4/src/lib.rs`

**Interfaces:**
- Consumes: nothing beyond `std`.
- Produces:
  - `struct CentredFD { pub c: f64, pub nu: f64, pub n: usize }`
  - `fn derivative(&self, u: &[f64]) -> Vec<f64>` (periodic centred difference `(u_{j+1} - u_{j-1}) / (2h)`)
  - `fn rate(&self, u: &[f64]) -> Vec<f64>` (`-c * derivative + nu * (u_{j+1} - 2 u_j + u_{j-1}) / h^2`)
  - re-export `advdiff::CentredFD`.

- [ ] **Step 1: Write the implementation**

Create `week4/src/fd.rs`:

```rust
//! Centred finite differences for u_t + c u_x = nu u_xx on the periodic
//! grid x_j = 2 pi j / n, j = 0..n-1, h = 2 pi / n.

use std::f64::consts::TAU;

pub struct CentredFD {
    pub c: f64,
    pub nu: f64,
    pub n: usize,
}

impl CentredFD {
    fn h(&self) -> f64 {
        TAU / self.n as f64
    }

    /// Periodic centred first difference (u_{j+1} - u_{j-1}) / (2h).
    pub fn derivative(&self, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n, "state length must equal n");
        let h = self.h();
        (0..self.n)
            .map(|j| {
                let up = u[(j + 1) % self.n];
                let um = u[(j + self.n - 1) % self.n];
                (up - um) / (2.0 * h)
            })
            .collect()
    }

    /// Rate of change: -c u_x + nu u_xx with centred periodic differences.
    pub fn rate(&self, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n, "state length must equal n");
        let h = self.h();
        (0..self.n)
            .map(|j| {
                let up = u[(j + 1) % self.n];
                let um = u[(j + self.n - 1) % self.n];
                -self.c * (up - um) / (2.0 * h)
                    + self.nu * (up - 2.0 * u[j] + um) / (h * h)
            })
            .collect()
    }
}
```

Modify `week4/src/lib.rs`:

```rust
pub mod fd;
pub mod fourier;
pub mod integrators;
pub mod scalar;

pub use fd::CentredFD;
pub use fourier::{dft, idft, FourierAdvDiff};
pub use integrators::{Euler, Integrator, Midpoint, RK4};
pub use scalar::Scalar;
```

- [ ] **Step 2: Run tests to verify GREEN**

Run: `cargo test` from `week4/`.
Expected: all tests pass — the Task 6 RED tests turn green here, including the six one-step accuracy cases and the convergence-order test. No warnings.

- [ ] **Step 3: Commit**

```bash
git add week4/src/fd.rs week4/src/lib.rs
git commit -m "feat: implement centred finite-difference rate function (green)"
```

### Task 8: Negative controls (sabotage) and final verification

**Files:**
- Modify temporarily: `week4/src/integrators.rs`, `week4/src/fourier.rs` (reverted before commit)

**Interfaces:**
- Consumes: everything.
- Produces: evidence (in this session's transcript) that the test suite catches the two specified defects; final state green.

- [ ] **Step 1: Sabotage Midpoint to be Euler**

In `week4/src/integrators.rs`, temporarily replace the body of `impl Integrator for Midpoint::step` with a copy of the `Euler` body (`let k1 = f(y); y.iter().zip(k1).map(|(yi, k1i)| *yi + k1i * dt).collect()`).

- [ ] **Step 2: Confirm the sabotage is caught**

Run: `cargo test` from `week4/`.
Expected failures: `midpoint_one_step_matches_exact_polynomial`, `one_step_fourier_midpoint_within_bound`, `one_step_fd_midpoint_within_bound`, `convergence_order_on_fourier_scheme` (Midpoint ratio ~2).

- [ ] **Step 3: Sabotage the Nyquist rule**

Revert Step 1 (`git checkout -- week4/src/integrators.rs`). Then in `week4/src/fourier.rs`, temporarily make `is_nyquist` return `false` always.

- [ ] **Step 4: Confirm the Nyquist sabotage is caught**

Run: `cargo test` from `week4/`.
Expected failure: `nyquist_mode_does_not_travel_only_diffuses` (imaginary part non-zero).

- [ ] **Step 5: Revert and verify final green**

```bash
git checkout -- week4/src/fourier.rs
cargo test
```

Run from `week4/`. Expected: every test passes, no warnings. Working tree clean.

- [ ] **Step 6: No commit**

The sabotage is a temporary negative control; after `git checkout --` the tree is identical to the Task 7 commit. Confirm with `git status --short` (empty).
