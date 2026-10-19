# Week 7: inverse problems and generative priors

This is the local-PDF version identified in [SOURCE.md](SOURCE.md). The live
course week has changed; the selected PDF remains the acceptance authority.
See [SPEC.md](SPEC.md) and [PLAN.md](PLAN.md) for the fixed experimental contract.

## Environment

```bash
uv sync --python 3.11
uv run pytest -q
```

An independent Python 3.11 environment uses pinned NumPy, SciPy, JAX and
matplotlib. The old Week 6 environment requires Python 3.12 and JAX 0.11.2;
this task uses Python 3.11-compatible JAX 0.6.2 and reuses the model source.
Model compatibility is checked before conditional inference. No new training.
`inputs/`, `artifacts/`, environments and caches remain outside Git.

## Part 1: sparse signal recovery

```bash
uv run python sparse.py --size 128 --sparsity 5 --repeats 24 --seed 7101 --out artifacts
uv run python figures.py --part 1
```

The actual success counts are **2, 20, 24, 24, 24 out of 24**, for p=0.1, 0.2,
0.4, 0.6 and 0.8 respectively. The p=.4 goal (at least 23) and p=.1 failure
goal (at least 12) pass. The largest observed-equation residual is 1.78e-11.
An independent DCT and a different LP formulation verify all 120 objectives;
the largest L1 objective difference is 2.46e-10. A hidden-pixel perturbation
negative control is rejected. See [part1-audit.json](evidence/part1-audit.json).

![Sparse recovery](evidence/sparse-recovery.png)

This figure supports recovery at sufficient scattered observations, and
visible failure at p=.1 despite exact observed values. Its bottom rows are
sorted rank 13, original trials 18 and 4, selected by the specified rule.
The error display floor does not alter saved numeric errors. A 24-trial
frequency is specific to this experiment, not a universal threshold.

In [Candes and Romberg's abstract](https://arxiv.org/abs/math/0611957), sparsity
and coherence both enter the sample requirement. Sparse coefficients reduce
the unknown structure; incoherent measurements spread information across
those coefficients. Sparsity alone does not make every measurement geometry
recoverable. DCT-II basis pursuit here is convex; the later hard-thresholded
image iteration is a different, nonconvex approximation.

## Part 2: DCT image inpainting

The live image download belongs to another exercise. The exact MNIST test rows
in the selected PDF were reconstructed from original official IDX files with
verified digests, without selecting new images. The first/second occurrence
of each digit is checked explicitly. Validation images are displayed but never
evaluated or fed to reconstruction. See [input provenance](evidence/input-provenance.json).

```bash
uv run python prepare_inputs.py
uv run python study.py --input inputs/images.npz --out artifacts
uv run ./inpaint --method dct --input inputs/case.npz --k 20 --iterations 100 --out artifacts/case
uv run python figures.py --part 2
```

Mean missing-pixel MSE is **0.1895889244** for random masks and **0.6017061911**
for central holes, within .0001 of both lecture values. Every mask has 400
missing pixels and 624 known pixels; only 384 known pixels belong to the
original digit, because the border contributes 240. Observed maximum error is
0. The standalone tool matches its batch counterpart exactly. An independent
dense cosine-matrix reconstruction of all 20 cases differs by at most 4.14e-14.
Full per-image values and negative-control result are in
[part2-audit.json](evidence/part2-audit.json).

![Inputs and masks](evidence/input-images.png)

`input-images.png` identifies all ten validation and evaluation rows and both
masks. Cyan is missing, and black is a measured -1 intensity. For digit 0
(official row 3), padded image pixel (row 6, column 17), with value .584314,
is observed under the random mask but lies inside the unmeasured hole. Most
central strokes of this digit have no direct hole-case measurements.

![DCT reconstruction](evidence/fixed-inpainting.png)

`fixed-inpainting.png` shows both masks, all ten reconstructions and standalone
agreement. DCT values range from -1.59762 to 1.05718 and are stored without
clipping. Every grayscale panel has display scale [-1,1]; numerical errors use
raw values. No hidden pixel enters either solver.

![DCT errors](evidence/fixed-error.png)

`fixed-error.png` contains all individual errors and mean lines. The hole mean
is 3.17374 times the scattered-mask mean at the same k=20 and 100 iterations.
The scattered mask samples strokes throughout the image, whereas a contiguous
hole can erase a whole stroke. Equal counts therefore do not imply equal
information. This iteration alternates a hard transform budget with exact
observed restoration; it is not Part 1's convex basis pursuit, and 100 iterations
are a fixed budget rather than a proof of convergence. Errors are averaged
only over the 400 withheld pixels, not diluted across all 1024 pixels.
