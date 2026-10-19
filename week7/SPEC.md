# Week 7: inverse problems and generative priors

Implement Parts 1, 2 and 3 in order, following SOURCE.md and the repository
AGENTS.md. Correctness is defined by the selected PDF, before implementation.
Use Python 3.11, NumPy/SciPy float64 for sparse recovery and image statistics,
JAX float32 for diffusion, and matplotlib for evidence. Commit each completed
part. Preserve raw evidence, fixed seeds and all unrelated weeks. No deletion,
retraining, cloud access or credential access is part of this task.

## Part 1

Orthonormal DCT-II synthesis with constant first column. Basis pursuit is a
linear program minimizing L1 subject to exact observed equations, not image
hard thresholding. For probabilities [0.1,0.2,0.4,0.6,0.8], run 24 trials each
with n=128, five nonzero coefficients and one `default_rng(7101)`. Each trial
draws support with `choice(n,5,replace=False)`, normal values, then the mask.
Persist float64 truth/observations/reconstruction [5,24,128], boolean masks,
relative_error [5,24], rates [5]. Success is error <0.001; at p=.4 at least
23 successes, at p=.1 at least 12 failures. Plot frequency, median/10-90% error,
and the 13th trial sorted by error at p=.1 and .4. Print all five counts.

## Part 2

Ten official MNIST test rows [3,2,1,18,4,8,11,0,61,7], validation rows
[10,5,35,30,6,15,21,17,84,9], labels 0..9; float32 [32,32] in [-1,1].
Never use validation or hidden truth in a reconstruction. Each mask hides 400
interior pixels. Random masks use one `default_rng(7001)` and the ten input
images in order; central hole is [6:26,6:26]. Keep padded borders observed.
DCT uses float64 orthonormal DCT-II, k=20, 100 iterations, ties retained at
`kth*(1-1e-12)`, no clipping, restoration after every iteration.
Implement the specified `inpaint` CLI and standalone first random case.
Persist evidence.npz and run.json with exactly the prescribed arrays/metadata.
Mean hidden errors must agree within .0001 with .18959/.60171; observed error
<1e-5. Show all validation/test identities, both masks, paired reconstructions,
standalone agreement and all per-image errors.

## Part 3

Reuse the old Week 6 loader/noise predictor and immutable EMA checkpoint.
Schedule beta=.0001..02, 1000 levels, abar[0]=1, float32. Network sees t-1.
Use the lecture's skipped-level posterior mean/variance, clipped clean estimate,
noise-matched observed replacement and forward resampling. No clipping after
merge. Every attempt splits `key,kz,ke,kf=split(key,4)` and draws z,eta,f,
including unused final draws. Initial split is `key,ki=split(PRNGKey(seed))`.
For the full [20,32,32,1] batch, use 100 levels, three attempts except final
s=0 once: exactly 298 calls/seed; seeds [7201,7202,7203,7204] in order.
Keep observations/masks as the only image inputs to the sampler. Save float32
diffusion [2,10,4,32,32], float64 hidden MSE [2,10,4], mean completion error and
population diversity [2,10]. Preserve DCT results. Trace all 298 attempts for
batch index 0/seed7201, with t,s,cycle,x_t,epsilon,x_s. Record batch shape, seed,
checkpoint hash, steps and repeats. Observed final error <1e-5; ensemble identity
residual <1e-6. No required improvement over DCT and no matching of course-model
diffusion numbers. Plot conditioning, all error curves, largest/smallest gains,
and four completions of the largest-variance hole case.

## Delivery

Eight named PNGs, runnable tools, all prescribed raw evidence, exact generating
commands, actual numeric results and interpretation in README. Independently
recompute key invariants from raw arrays and replay the sampler noise algebra.
Inspect figures. Export a verified local copy/manifest for Windows access.
Student presentation/explanation and instructor acceptance are not inferred
from automated checks. Publication requires existing user authorization.
