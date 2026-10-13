# Week 6 Part 1 — Learn the score of a 2D Gaussian mixture

Part 1 only. Later parts of Week 6 are intentionally deferred per the task
instruction and are not specified here.

## Target distribution

Four equally weighted Gaussian components with covariance `0.09 I`, centred at

```
(3, 0), (0, 3), (-3, 0), (0, -3)
```

so each coordinate has standard deviation `sqrt(0.09) = 0.3`.

## Forward process and schedule

- `T = 1000` noise levels, indexed `1..T`.
- `beta_t` linear from `0.0001` to `0.02` inclusive: `beta = linspace(1e-4, 0.02, 1000)`.
- `alpha_t = 1 - beta_t`, `abar_t = prod_{s<=t} alpha_s`, `abar_0 = 1`.
- Closed form: `x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps`, `eps ~ N(0, I)`.

## Noise predictor

A neural network `eps_theta(x_t, t)` predicts the added noise. Training:

- uniform integer noise levels `t ~ U({1..1000})`
- loss: mean squared noise-prediction error over batch and coordinates
- 25000 updates, batch 512, seed 2026

The architecture is an MLP whose input is `(x_t, sinusoidal time embedding)`.
Model/optimizer details may be changed if an acceptance check fails, but seed,
data, budget, and tolerance are fixed.

## Score identities used for verification

- Learned score estimate: `score_theta(x_t, t) = -eps_theta(x_t, t) / sqrt(1 - abar_t)`.
- Exact noised-mixture score: at level `t`, component `k` has mean
  `sqrt(abar_t) * mu_k` and variance `abar_t * 0.09 + (1 - abar_t)`; the mixture
  score is the posterior-weighted average of component scores.
- Exact `x0 -> x_t` noising and the DDPM reverse update (equation 5 of Ho et
  al. 2020) are the other two mathematical building blocks.

## Sampling

- 2000 samples, 1000-step DDPM, seed 2026, **no clipping** (the mixture extends
  beyond `[-1, 1]`).

## Acceptance checks

1. Score comparison at `t = 50, 200, 500, 900` using 4096 **fresh** samples of
   each noised distribution. Print each relative L2 error and their mean.
   - relative L2 error: `||learned - exact||_2 / ||exact||_2` over all sample
     points and coordinates.
   - **Mean relative L2 error must be below 0.15.**
2. `evidence/score.png`
   - quiver plot of learned and exact fields at `t = 500` over the noised samples
   - relative L2 error against noise level
3. `evidence/mixture.png`
   - overlay of generated samples and fresh target samples, equal axis scales,
     x-coordinate horizontal.

## Layout and reproducibility

- `ddm.py` — production module: schedule, noising, exact score, noise
  predictor, DDPM reverse update and sampling.
- `test_ddm.py` — mathematical tests derived from the identities above.
- `scripts/` — experiment, evaluation, and plotting scripts.
- `evidence/` — `score.png`, `mixture.png`.
- `artifacts/` and `runs/` — raw runs (ignored by git).
- `README.md` — reproducible commands and measured values.