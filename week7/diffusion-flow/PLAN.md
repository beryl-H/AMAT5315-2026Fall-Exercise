# Plan — Week 7 diffusion-flow, Part 1

Per-task order follows the lecture's BLUE-PANEL order. Each task ends with a
commit scoped to `week7/diffusion-flow/`.

## Task 0 — Docs and environment (this commit)
- `SPEC.md` (fixed 5-Part acceptance), `PLAN.md`, `SOURCE.md`.
- `pyproject.toml` + `uv.lock` pinned (jax==0.6.2 CPU, numpy, matplotlib,
  pytest, imageio for GIF/MP4). Independent lock file; uv package cache reused.
- `.gitignore` for raw artifacts, checkpoints, downloads.

## Task 1 — Target and forward diffusion only (TDD)
Define correctness first: hand-derived noising coefficients, target stats,
independent streams, mistake control (a variance/std mistake must fail).
- `tests/test_target_forward.py` written first, observed failing.
- `src/diffusion.py`: `sample_target(key, n)`, `beta_schedule(T)`,
  `abar(betas)`, `noise_to_level(x0, t, abar, eps)`.
- Commit before any training or reverse code.

## Task 2 — diffusion.toml + network + training loop
- `diffusion.toml` copied exactly from the lecture sheet.
- `src/network.py`: time features `r(t)` (16 sin + 16 cos,
  `omega_j = 10000**(-j/16)`, phase `(t-1)*omega_j`), 34->128->128->128->2 MLP,
  SiLU, `Normal(1/sqrt(n))` weights, zero bias, linear output.
- `src/train.py`: Algorithm 1 — batch 512 fresh target samples, independent
  eps, uniform t in 1..1000, Adam(0.9, 0.999, 1e-8), cosine lr 0.001 -> 0,
  25000 updates, seed 2026, float32; save update 0 and
  [50,100,200,500,1000,2000,5000,10000,25000], full loss history, metadata,
  finite-weight check.
- Tests: time features (hand values at small t), forward pass shape,
  initialization statistics, determinism under fixed seed.
- Training executed in background (log + exit marker, <45 s tool calls).

## Task 3 — DDPM reverse sampler (TDD, negative control)
- `src/sample.py`: Algorithm 2 verbatim as production code — mean
  `(x - beta_t/sqrt(1-abar_t) * eps_theta)/sqrt(alpha_t)`, variance `beta_t`,
  no clipping, no noise at `t=1`.
- `tests/test_ddpm.py` against the production function with a scripted
  predictor: one shared initial Gaussian and one shared reverse-step noise
  array; hand-computed two-step mean/variance check; negative control
  (wrong variance) must fail the test.

## Task 4 — Real training run + evidence
- Train for real (no reference numbers copied). Track loss history and
  checkpoints in `artifacts/` (gitignored), summaries in `evidence/`.
- VERIFY (1): forward movie, >=2000 same points, fixed scale; print per-
  coordinate mean and variance at t=1000; thresholds 0.1/0.1.
- VERIFY (2): training movie — exactly the 10 saved stages, 10 DDPM paths per
  stage sharing starts and reverse noise, loss-so-far, honest annotation of
  early out-of-view paths (no clipping of states).
- VERIFY (3): reverse movie + final scatter, 2000 fresh samples, 1000 steps;
  per-cluster fraction/centre/coordinate-std saved explicitly; thresholds
  0.04 / 0.05 / 0.03.
- `evidence/stats.json` with every measured number; on threshold failure,
  retain results and diagnose (no seed/tolerance changes).

## Task 5 — README + animations + final commit
- PNG companions and genuine GIF/MP4 (imageio/ffmpeg) in `evidence/`.
- README: figures beside exact generating commands, seeds, settings, software
  versions, training time, measured numbers, interpretation of the loss
  plateau without optimality claims. No raw agent thinking in README.

## Stop condition
STOP after Task 5. Parts 2–5 wait for the coordinator's independent audit.
