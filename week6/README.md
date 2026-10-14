# Week 6 — Diffusion models

Part 1 (learned score of a Gaussian mixture) is complete and reviewed.
Part 2 (MNIST UNet + diffusion CLI) is implemented and smoke-verified; its
formal 30000-update training run and the resulting loss/grid/trajectory/denoising
evidence are **pending cloud price approval**. Parts 3-5 have not started.

## Part 2 status (this update)

- `diffusion.design.toml` + `src/diffusion/` implement the sheet's train/sample
  interface exactly (`uv run diffusion train ...`, `uv run diffusion sample ...`).
- Smoke verified end to end on CPU; smoke outputs are retained under `runs/`
  and are NOT formal results.
- Formal 30000-update MNIST run: pending explicit human approval of the cloud
  price packet in `cloud/runbook.md` (no resource created, no price guessed).
- Pending after that run: `evidence/mnist-loss.png`, `evidence/mnist-grid.png`,
  `trajectory.jsonl`, `evidence/mnist-denoising.png`, then Parts 3-5.

## Part 2: environment and dependency compatibility (tested)

- Same pinned stack as Part 1, on Python 3.12.3 (WSL2, CPU):
  `jax==0.11.2`, `jaxlib==0.11.2`, `numpy==2.5.3`, `matplotlib==3.11.2`,
  `pytest==9.1.1`. Part 1 tests still pass unchanged after the packaging
  change.
- The repo is now a uv package (hatchling build, editable install):
  `uv run diffusion ...` comes from `[project.scripts]`. `uv sync` keeps the
  CPU wheels; `uv sync --extra cuda` (`jax[cuda12]==0.11.2`) is locked in
  `uv.lock` for the GPU instance but **not yet exercised** (no GPU here);
  jax 0.11.2 does publish the `cuda12` extra on PyPI (checked).
- The GPU path needs only the NVIDIA kernel driver + libcuda on the host; the
  CUDA runtime ships in the pip wheels.

## Part 2: commands and smoke measurements (CPU, WSL2, 16 cores, ~14 GiB host)

Download the dataset (git-ignored):

```bash
mkdir -p inputs
curl -fL https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz \
  -o inputs/mnist.npz
```

Verified after download: train (60000, 28, 28) uint8, test (10000, 28, 28);
preprocessed (60000, 32, 32) float32 in [-1, 1]; the 2-pixel border is -1.

Mathematical/interface tests (run before implementation, watched failing, then
implementation): 24 new tests in `test_diffusion.py`; total `33 passed` with
Part 1's 9.

```bash
uv run pytest -q
```

Smoke training (exact command, exact numbers below are SMOKE, not formal
results):

```bash
uv run diffusion train --data mnist --input inputs/mnist.npz \
  --updates 200 --batch 64 --seed 2026 --out runs/smoke
```

Measured: 200 updates in 65.0 s CPU (incl. ~4 s compile); loss 1.2505 at step 1
-> 0.0867 at step 200; `runs/smoke/loss.jsonl` has exactly 200 lines; the
checkpoint stores settings + EMA weights (roundtrip verified).

Smoke sampling (exact command; smoke, not formal):

```bash
uv run diffusion sample --data mnist --checkpoint runs/smoke/checkpoint.npz \
  --sampler ddpm --steps 1000 --n 4 --seed 1 --out runs/smoke-samples
```

Measured: 12.8 s CPU; `run.json` matches the contract (case mnist, size 32,
t_steps 1000, beta [0.0001, 0.02], sampler ddpm, sample_steps 1000, n_samples
4, seed 1); `samples.jsonl` has 4 rows with 1024 values each, all in [-1, 1]
and rounded to 3 decimals; peak RSS ~0.5 GB.

End-to-end trajectory check (retained under `runs/smoke-trajectory/`, smoke):

```bash
uv run diffusion sample --data mnist --checkpoint runs/smoke/checkpoint.npz \
  --sampler ddpm --steps 1000 --n 4 --seed 1 --out runs/smoke-trajectory --trajectory
```

Measured: 26 frames at t = 1000, 960, ..., 0; x and x0 each 8*1024 values;
first frame is pure noise (std 0.998); at t = 0 both arrays equal the final
image.

Architecture choices (documented): UNet 32->16->8 with channels (16, 32, 64),
two residual blocks per level, GroupNorm(8)+SiLU, 3x3 convs, skip connections,
nearest-neighbor upsample, sinusoidal time embedding (64) -> Dense(128) ->
SiLU -> Dense(128) injected per block; Adam lr 2e-4 constant (DDPM standard);
EMA 0.999 every update; batches drawn uniformly from all 60000 training
images; labels and the test split are never used in training.

`evidence/mnist-inputs.png` (command: `uv run python
scripts/plot_mnist_inputs.py`) was rendered and inspected: 16 preprocessed
digits with color bars, and one image at t = 0, 200, 500, 1000 - the border is
-1 and structure disappears as t increases (t = 1000 is pure noise, std 1.014,
mean -0.009).

## Part 2: cloud approval packet (STOPPED HERE)

`cloud/runbook.md` prepares the full round trip: aliyun CLI (official release
v3.5.1), read-only zone/spec/availability/price queries, security group with
SSH restricted to the human's current public IP, key pair, instance creation,
`uv sync --extra cuda`, JAX GPU verification, the exact 30000-update command,
copy-back, release by ID and `DescribeInstances` confirmation. Every
resource-creating/releasing command is marked `[GATED]`; none was executed.
`cloud/README.md` records status NOT_STARTED (no instance, no charges).
This host has no `aliyun` CLI and no credentials, so the packet contains exact
read-only quote commands for the human instead of guessed prices.

## Part 1 (complete)

Part 1 (learned score of a Gaussian mixture) is complete and reviewed.

## Setup

```bash
cd week6
uv sync
```

`uv` was installed with the official installer
(`curl -LsSf https://astral.sh/uv/install.sh | sh`). JAX runs on CPU.

## Files

- `SPEC.md` — Part 1 task specification and acceptance criteria
- `ddm.py` — schedule, forward noising, exact mixture score, noise predictor,
  DDPM reverse step and sampling
- `test_ddm.py` — mathematical tests for the exact score, noising, and reverse
  update
- `scripts/train_p1.py` — train the noise predictor
- `scripts/evaluate_p1.py` — generate samples, compare scores, draw evidence
- `scripts/verify_p1.py` — independently check persisted run settings and acceptance
- `evidence/` — figures
- `artifacts/p1/` — raw run outputs (git-ignored)

## Commands and measured values

### Mathematical tests

```bash
uv run pytest -q
```

Result after independent review: `9 passed`, including a fixed-key reverse-noise
reference and a negative control that detects omitted reverse noise.

### Training

```bash
uv run python scripts/train_p1.py
```

Config: seed `2026`, 25000 updates, batch 512, uniform levels `1..1000`, MSE
noise-prediction loss, MLP `2+64 -> 256 -> 256 -> 256 -> 256 -> 2` with SiLU,
Adam `lr=1e-3`. Target: four Gaussians centred `(3,0),(0,3),(-3,0),(0,-3)`,
covariance `0.09 I`. Schedule: `beta = linspace(1e-4, 0.02, 1000)`.

Measured: final loss `0.254224`, wall time `154.4 s`.

### Sampling and evaluation

```bash
uv run python scripts/evaluate_p1.py
```

Writes `artifacts/p1/samples.npz` (2000 samples, 1000-step DDPM, seed 2026, no
clipping), `artifacts/p1/target-samples.npz`, score errors, measurements, and
the two evidence figures.

Score check (4096 fresh noised samples per level; relative L2 error =
`||learned - exact||_2 / ||exact||_2`):

| level t | relative L2 error |
|--------:|------------------:|
| 50      | 0.155761          |
| 200     | 0.113819          |
| 500     | 0.058819          |
| 900     | 0.043249          |
| mean    | **0.092912**      |

Mean `0.092912 < 0.15`, so the acceptance check passes.

Sample check: all 2000 generated points lie within radius 1.5 of one of the four
centres (100%), matching the target's 100%. Counts within radius 1.5 by centre:

| centre   | generated | target |
|----------|----------:|-------:|
| (3, 0)   | 489       | 463    |
| (0, 3)   | 554       | 550    |
| (-3, 0)  | 479       | 495    |
| (0, -3)  | 478       | 492    |

### Evidence figures

`evidence/score.png` is produced by `scripts/evaluate_p1.py`:

- learned-score quiver and exact-score quiver at `t = 500` over the noised
  samples (120 shared subsampled arrows per field, with identical axes and scale),
- relative L2 error against noise level, with the 0.15 acceptance limit.

`evidence/mixture.png` is produced by `scripts/evaluate_p1.py`:

- overlay of 2000 generated samples and 2000 fresh target samples with equal
  axis scales.

## Commit history for this work

```text
3b01cb7 week6: specify Part 1 diffusion score task and Python environment
393c3ae week6: define Part 1 mathematical tests that fail without ddm module
f60c1ba week6: implement Part 1 diffusion math and pass mathematical tests
74b8529 week6: add Part 1 training and evaluation scripts
```

## Notes

- `artifacts/`, `runs/`, and `inputs/` are git-ignored.
- No answer-key reference numbers were used as results; all values above were
  measured from this repository's own run.
- Generated samples use no clipping because the mixture extends beyond `[-1, 1]`.

## Reviewed evidence and independent verification

```bash
uv run python scripts/evaluate_p1.py
uv run python scripts/verify_p1.py
uv run pytest -q
```

![Learned and exact scores, and per-level errors](evidence/score.png)

![Generated and target samples](evidence/mixture.png)

The dashed score line is the **mean** acceptance threshold, not a requirement
that every individual level be below 0.15. At t=50 the error is 0.155761;
the four-level mean is 0.092912 and passes the prescribed check. The mode
counts, radius-1.5 coverage and coordinate ranges are persisted in
`artifacts/p1/measurements.json`, rather than inferred from the plot.

Independent verification after review: 9 tests passed; the standalone
verifier passed; all loss step numbers 1..25000 were present, all losses
were finite, and the saved samples had shape (2000,2). The learned and
exact score arrows at t=500 point towards greater probability near the
centre of the heavily noised distribution. Generated samples recover all
four clean-data modes with a similar spread to fresh target samples.

## Execution status

Part 1 was implemented and trained through native Linux pi in WSL, using the
human-authorized provider. An independent Codex review requested readable
quiver scales and meaningful reverse-noise tests; pi implemented those
corrections and a standalone verifier.

Part 2 is implemented and smoke-verified: design contract, failing tests,
`diffusion` package (UNet + DDPM + EMA + trajectory), MNIST download,
`evidence/mnist-inputs.png`, and the gated cloud packet. The formal
30000-update MNIST run waits for explicit human approval of the price packet;
no cloud resource exists. Parts 3-5 have not started. No push yet.
