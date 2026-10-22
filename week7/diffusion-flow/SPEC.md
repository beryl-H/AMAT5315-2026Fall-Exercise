# Week 7: Diffusion and flow matching

Scope: all five parts are implemented, executed and independently audited.
Standing numerical FAIL results remain: Part2 learnedDDIM/DDPM-A width gate
and Part3 uniform-grid velocity gate at s=.1,.5. Engineering checks do not
override these measured scientific criteria. See README.md and AUDIT.md.

Source of truth: `learning-sheet.txt` (19 pages, PDF SHA256
`4147c45cc6ac9f6e1815ba33684a1c4ed50c87d8d89ffb14f3dccab6bcfddbd5`), the lecture
"Week 7: Diffusion and flow matching" by Jin-Guo Liu and Xi-Wei Pan.
Work directory: `week7/diffusion-flow/` (independent topic directory; the prior
week7 inverse-prior exercise is untouched).

All Part 1 computation is local CPU only, float32, JAX. No cloud compute, no
model downloads before Part 4, no new MNIST training.

## Part 1 — Diffusion models on a Gaussian mixture

Correctness is defined by the lecture's VERIFY items:

1. **Target and forward diffusion.** Target is the equally weighted Gaussian
   mixture with centres `(3,0), (0,3), (-3,0), (0,-3)` and covariance
   `0.09 I` (std 0.3 per coordinate). Forward: `T = 1000` levels, `beta_t`
   linear from `1e-4` to `0.02`, `alpha_t = 1 - beta_t`,
   `abar_t = prod_{s<=t} alpha_s` with `abar_0 = 1`. Direct noising
   `x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps`.
   - Forward movie follows >=2000 SAME points through the Markov chain on a
     fixed coordinate scale.
   - At `t = 1000`: each coordinate mean `|mean| < 0.1`, each coordinate
     variance within `0.1` of `1`.

2. **Noise predictor training** exactly per `diffusion.toml`:
   MLP widths `[128,128,128]`, SiLU; 32 time features (16 sine + 16 cosine)
   for `j = 0..15` with `omega_j = 10000**(-j/16)` and phase `(t-1)*omega_j`;
   input is `concat(x[2], features[32])` = 34 features. Dense weights
   `Normal(std = 1/sqrt(input width))`, zero bias, linear 2-output.
   Adam `beta1=0.9, beta2=0.999, eps=1e-8`, lr `0.001` cosine-decayed to `0`,
   float32, `25000` updates, batch `512`, fresh target samples, independent
   Gaussian noise, uniform integer levels `1..1000` per update, seed `2026`.
   Save initialization (update 0) and updates
   `[50,100,200,500,1000,2000,5000,10000,25000]`, all 25000 loss records,
   model/training/RNG metadata and settings, and finite-weight checks.

3. **DDPM sampling.** Reverse mean
   `(x - beta[t]/sqrt(1-abar[t]) * eps_theta)/sqrt(alpha[t])`, reverse variance
   `beta[t]`, no clipping, no added noise at `t = 1`.
   - Training movie: exactly the 10 saved stages, ten DDPM paths per stage and
     the loss-so-far curve, identical starts and reverse-step noise across
     stages; early out-of-view paths annotated honestly (no state clipping).
   - Reverse movie / final scatter: 2000 fresh samples, 1000 reverse steps.
     Classified by nearest target centre: each cluster fraction within `0.04`
     of `0.25`, each per-cluster per-coordinate std within `0.03` of `0.30`,
     each centre coordinate within `0.05` of the target centre.

Acceptance gates (all must hold; on failure: retain results, diagnose, do not
silently change seeds/budget/tolerances):

- G1: independent math tests of noising coefficients (hand-derived), target
  statistics, and stream independence pass; variance/std mistakes are caught.
- G2: training runs to completion with real measured loss (no fabricated
  numbers); checkpoints and loss history saved with metadata.
- G3: DDPM reverse tests exercise the production reverse code and a negative
  control (wrong variance) fails.
- G4: final statistics satisfy the thresholds above; every statistic saved
  explicitly (all coordinates and centres, not just averages).
- G5: readable PNG companions plus genuine GIF/MP4 animations embedded/linked
  in README beside the exact generating commands, with seeds, settings,
  software versions, training time and measured numbers.

## Parts 2–5 — requirements and recorded scope

- **Part 2 — DDIM (COMPLETE)**: deterministic sampling over a selectable level grid with
  the trained predictor, plus the mixture's exact conditional-mean noise
  prediction. Verify: two DDIM runs from the same noise agree exactly
  (max diff printed, reference prints 0); DDPM from the same start with two
  noise draws lands in the same cluster ~1/4 of the time; DDIM fractions
  within 0.015 of 0.25, both samplers' widths in 0.29–0.32.
- **Part 3 — Flow matching (COMPLETE)**: train a velocity network from scratch with
  `flow.toml` (seed 2029, features use `1000*s`), path `x_s=(1-s)x_0+s z`,
  target velocity `z-x_0`; Euler sampling `s=1 -> 0`; exact marginal velocity
  of the mixture. Verify: relative RMS difference vs exact field < 0.1 at
  `s=0.1,0.5,0.9` (reference 0.011/0.017/0.058). Step-count study: DDPM, DDIM,
  flow at 1,2,5,10,20,50,100 steps (1000 levels for DDPM/DDIM), 4000 samples
  per run, energy distance to 4000 exact samples, null scale from 20
  independent exact pairs; report cluster fractions, centres, widths.
- **Part 4 — CIFAR-10**: pin pretrained `google/ddpm-cifar10-32`
  (revision `267b167dc01f0e4e61923ea244e8b988f84deb80`), load into JAX, same
  1000-step DDPM sampler, no clipping, float32, model index `t-1`; generate
  four images and visualize reverse stages.
- **Part 5 — RePaint**: fetch `inputs/images.npz`; RePaint with 50 reverse
  jumps over `1000,980,...,20,0`, two attempts per jump except the last,
  forward noise between attempts, observed pixels replaced by forward-noised
  observations; hide rows/columns 6:26; seeds 7201–7204; hidden-pixel MSE and
  observed-pixel change < 1e-5.

Repository rules: commit after each task; raw runs/checkpoints/downloads stay
out of git (tracked small summaries and evidence figures only); each part proceeded after the preceding independent coordinator checkpoint.
