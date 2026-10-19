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
