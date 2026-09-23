# Week 4 — One `Integrator` trait, three Runge–Kutta methods, two rate functions

Crate: `advdiff` (library, edition 2021, Rust 1.75). Dependency: `num-complex = "0.4"`.

## Problem

Solve $u_t + c\,u_x = \nu\,u_{xx}$ on a periodic $[0, 2\pi)$ grid of $n$ points,
$x_j = 2\pi j/n$, $h = 2\pi/n$, by advancing the method-of-lines ODE system
$\dot u = f(u)$ one step at a time with three explicit integrators.

## API

### Scalar abstraction (`src/scalar.rs`)

```rust
pub trait Scalar: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f64, Output = Self> {
    fn zero() -> Self;
}
```

Implemented for `f64` and `num_complex::Complex<f64>`. All integrator arithmetic
goes through this trait, so one trait serves both real and complex state vectors.

### Integrator trait (`src/integrators.rs`)

```rust
pub trait Integrator {
    fn step<F, S>(&self, f: F, dt: f64, y: &[S]) -> Vec<S>
    where
        F: Fn(&[S]) -> Vec<S>,
        S: Scalar;
}
```

Three unit structs implement it:

- `Euler` — forward Euler: `y + dt * f(y)`
- `Midpoint` — explicit midpoint (RK2): `k1 = f(y)`, `y + dt * f(y + dt/2 * k1)`
- `RK4` — classical: `k1..k4` with weights `1/6, 1/3, 1/3, 1/6`

No other stepping helper; callers loop `step` as needed.

### Fourier rate function (`src/fourier.rs`)

```rust
pub struct FourierAdvDiff { pub c: f64, pub nu: f64, pub n: usize }
impl FourierAdvDiff {
    pub fn rate(&self, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>>;
    pub fn derivative(&self, u_hat: &[Complex<f64>]) -> Vec<Complex<f64>>; // d/dx only
}
```

State: spectral coefficients `u_hat[k]`, `k = 0..n`. Wavenumbers
`k_j = j` for `j < n/2`, `k_j = j - n` for `j > n/2`, and the Nyquist mode
`j = n/2` has wavenumber treated as `k = n/2` **with first-derivative
coefficient forced to zero** — the Nyquist mode does not travel; only its
diffusion acts (`rate` for that mode is `-nu * (n/2)^2 * u_hat[n/2]`).

Rate: `du_hat[k]/dt = (-i c k - nu k^2) u_hat[k]`, except Nyquist as above.

Also `fn dft(u: &[f64]) -> Vec<Complex<f64>>` and
`fn idft(u_hat: &[Complex<f64>]) -> Vec<f64>` (plain O(n^2), conventions
`u(x) = sum_k u_hat[k] e^{ikx}`, `u_hat[k] = (1/n) sum_j u_j e^{-ikx}`), so
tests and scripts can move between physical and spectral space.

### Centred finite differences (`src/fd.rs`)

```rust
pub struct CentredFD { pub c: f64, pub nu: f64, pub n: usize }
impl CentredFD {
    pub fn rate(&self, u: &[f64]) -> Vec<f64>;
    pub fn derivative(&self, u: &[f64]) -> Vec<f64>;
}
```

`rate_j = -c (u_{j+1} - u_{j-1})/(2h) + nu (u_{j+1} - 2 u_j + u_{j-1})/h^2`,
periodic wrap at both ends.

## Exact single-wave references (what the tests compare against)

A single mode must have a closed-form one-step reference so any test error is
purely temporal:

- **Fourier rate** (spatially exact): `u(x,0) = sin(k0 x)` evolves as
  `u(x,t) = exp(-nu k0^2 t) sin(k0 (x - c t))`;
  spectrally `u_hat[k0](t) = u_hat[k0](0) * exp((-i c k0 - nu k0^2) t)`.
- **Centred FD rate** (exact solution of the semi-discrete system, i.e. the
  modified-wavenumber wave): mode `e^{i k0 x_j}` has advection eigenvalue
  `i c sin(k0 h)/h` and diffusion eigenvalue `-4 nu sin^2(k0 h/2)/h^2`, so
  `u_j(t) = exp(-4 nu sin^2(k0 h/2) t / h^2) * sin(k0 x_j - c sin(k0 h) t / h)`.

## Tests (`tests/contract.rs`)

Parameters unless stated: `n = 32`, `c = 1.0`, `nu = 0.1`, `k0 = 2`,
`dt = 0.05`, one step from the single-wave initial condition.

1. **Rate-function spot checks** (both schemes): derivative of a constant
   field is zero; Fourier derivative of a single mode equals
   `(-i c k0 - nu k0^2) u_hat` applied mode-wise; FD derivative of a single
   mode equals the modified-wavenumber eigenvalues above.
2. **Nyquist rule** (Fourier scheme): with only the `k = n/2` mode excited,
   `rate[n/2]` must equal `-nu (n/2)^2 u_hat[n/2]` exactly — no advective
   (imaginary) part. A defect that lets the Nyquist mode travel must fail this.
3. **One-step accuracy, all 3 integrators x both rate functions** (6 cases):
   after one `step` from the initial condition, the state must match the
   corresponding exact reference within:
   - Euler: max abs error < 1e-2
   - Midpoint: < 1e-3
   - RK4: < 1e-6
   with `|lambda| dt ~ 0.1`, these bounds sit roughly one and three orders
   above the expected local errors.
4. **Convergence order** (all 3 integrators on the Fourier scheme, complex
   one-step errors at `dt` and `dt/2`): error ratio must fall in
   Euler (1.7, 2.3), Midpoint (3.5, 4.5), RK4 (14, 18) — a wrong-order
   implementation cannot pass.

## Verification

From `week4/`: `cargo test` — all tests green, no warnings.

## Negative controls

Before accepting the implementation: sabotage one integrator (e.g. return the
Euler result from `Midpoint`) and one rate-function defect (Nyquist advection
on); confirm the corresponding tests fail; revert; confirm green.

## Scripts

Any run/comparison/plot script for this week lives in `week4/scripts/` and is
run from there.
