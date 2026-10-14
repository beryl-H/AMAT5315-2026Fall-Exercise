# Week 6 — Diffusion models

Part 1 (learned score of a Gaussian mixture) is complete and reviewed.
Part 2 (MNIST UNet + diffusion CLI) is **complete**: the formal 30000-update
run was trained on the human-purchased Alibaba Cloud A10, independently
verified, copied back with SHA256 equality, and the local preview/trajectory
evidence was produced. Parts 3-5 have not started.

## Part 2 status (this update)

- Formal training verified: exit 0, wall **368 s** on `ecs.gn7i-c8g1.2xlarge`
  (NVIDIA A10 23028 MiB, driver 580.126.09, jax 0.11.2 on `cuda:0`),
  ~84 steps/s at mid-run; loss 1.250534 -> **0.013941** (min 0.007980 at step
  27973); 30000 ordered finite losses; exact settings (updates 30000, batch
  64, seed 2026); 126 finite EMA arrays.
- `checkpoints/mnist.npz` promoted (was absent) from the verified copy.
- Preview sampling ran **locally on CPU** after copy-back (no extra GPU
  tasks): 100 samples + separate 8-sample 1000-step trajectory, exit 0,
  138.2 s.
- Instance release (exact ID `i-t4n7f2zqpa44cb3hh7uc`) is the human's gated
  action; no further instance work is needed by this repo.
- Honest incident record: a coordination race during initial setup produced a
  second training process; root stopped both processes and preserved the
  mixed output in the remote `runs/train-mnist`. That run is **INVALID**, was
  never promoted, and remains preserved on the instance for audit. The valid
  run is root's locked `runs/train-mnist-a10-20261014-1421` (tmux
  `mnist-verified`, flock-guarded), whose artifacts are the only ones used
  here.

## Part 2: formal training evidence

Deployed source: local `git archive` of HEAD `36c56ee7e27e7db4335ab16b196f3f57e3551515`
(SHA256 `144fb0f9...f1a2`, verified on both sides; dataset SHA256
`731c5ac6...673d1` verified on both sides). Remote environment: uv 0.13.1,
`uv sync --extra cuda` exit 0, JAX GPU assertion passed before training
(`gpu-check.log`).

Remote command (run by root's lock script, unchanged model/settings):

```bash
uv run diffusion train --data mnist --input inputs/mnist.npz \
  --updates 30000 --batch 64 --seed 2026 --out runs/train-mnist-a10-20261014-1421
```

Measured: start 2026-10-14T14:21:00+08:00, end 14:27:09+08:00, wall **368 s**,
exit **0**, ~84 steps/s. Retrieval: `retrievals/20261014-1427-formal-mnist/`
(run dir + full train log + gpu/start/end/exit/wall markers + gpu-check.log +
sync.log); remote/local SHA256 verified for all 10 files;
`scripts/verify_formal_train.py` all checks passed.

`evidence/mnist-loss.png` (command: `uv run python scripts/plot_mnist_loss.py
--loss retrievals/20261014-1427-formal-mnist/train-mnist-a10-20261014-1421/loss.jsonl`):
loss on a logarithmic axis falls from order one toward ~0.014 and plateaus
around 0.02-0.03, matching the expected trend.

## Part 2: preview and trajectory evidence (local CPU, from the formal checkpoint)

```bash
uv run diffusion sample --data mnist --checkpoint checkpoints/mnist.npz \
  --sampler ddpm --steps 1000 --n 100 --seed 2026 --out runs/mnist-preview --trajectory
```

Measured: exit 0, **138.2 s** CPU; `samples.jsonl` 100 rows, `trajectory.jsonl`
26 frames. The denoising recording is copied to `week6/trajectory.jsonl` for
the course viewer (https://giggleliu.github.io/AMAT5315-2026Fall/week6-viewer.html);
the human loads the viewer and saves the official t = 200 PNG as
`evidence/mnist-denoising.png` replacement if desired - the current
`evidence/mnist-denoising.png` is a labelled matplotlib substitute (command:
`uv run python scripts/plot_mnist_denoising.py --trajectory trajectory.jsonl`),
showing the current image above and predicted clean image below at
t = 1000, 800, 200, 0: the prediction becomes recognisable earlier than the
noisy state, as the lecture describes.

`evidence/mnist-grid.png` (command: `uv run python scripts/plot_mnist_grid.py
--samples runs/mnist-preview/samples.jsonl`): the 100 generated digits as a
10x10 grid - recognisable digits of varied classes and shapes.

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
results; full unfiltered logs incl. `/usr/bin/time -v` are kept next to the
outputs):

```bash
/usr/bin/time -v uv run diffusion train --data mnist --input inputs/mnist.npz \
  --updates 200 --batch 64 --seed 2026 --out runs/smoke 2>&1 | tee runs/smoke/train.log
```

Measured: 200 updates in **81.8 s** CPU (incl. compile), peak RSS **1.70 GB**,
exit 0; loss 1.2505 at step 1 -> 0.0867 at step 200; `runs/smoke/loss.jsonl`
has exactly 200 lines; the checkpoint stores settings + EMA weights (126 arrays,
all finite, roundtrip verified).

Smoke sampling (exact command; smoke, not formal):

```bash
/usr/bin/time -v uv run diffusion sample --data mnist --checkpoint runs/smoke/checkpoint.npz \
  --sampler ddpm --steps 1000 --n 4 --seed 1 --out runs/smoke-samples 2>&1 | tee runs/smoke-samples/sample.log
```

Measured: 16.3 s CPU, peak RSS 0.50 GB; `run.json` matches the contract
verbatim; `samples.jsonl` has 4 rows with 1024 values each, all in [-1, 1] and
rounded to 3 decimals.

Trajectory integration check on the SMOKE checkpoint (retained under
`runs/smoke-trajectory/`; smoke, not formal):

```bash
uv run diffusion sample --data mnist --checkpoint runs/smoke/checkpoint.npz \
  --sampler ddpm --steps 1000 --n 4 --seed 1 --out runs/smoke-trajectory --trajectory
```

Independent verification (recorded in `runs/smoke/verify.log`):

```bash
uv run python scripts/verify_smoke.py
```

All checks passed: no traceback in any smoke log; train exited 0; loss steps
exactly 1..200 in order and finite; EMA checkpoint loads with the fixed
settings; run.json matches the contract; samples are 4x1024, finite, in
[-1, 1], rounded to 3 decimals; trajectory has 26 frames at t =
1000, 960, ..., 0 with 8192 finite values per array; the t = 0 frame has
`x == x0` holding the **clipped saved final image** (within [-1, 1]), while
intermediate frames may be unbounded (392,183 out-of-range values in the
undertrained smoke run, all finite - allowed by the sheet; no clipping rule is
invented for intermediates).

Architecture choices (documented): UNet 32->16->8 with channels (16, 32, 64),
two residual blocks per level, GroupNorm(8)+SiLU, 3x3 convs, skip connections,
nearest-neighbor upsample, sinusoidal time embedding (64) -> Dense(128) ->
SiLU -> Dense(128) injected per block; Adam lr 2e-4 constant (DDPM standard);
EMA 0.999 every update; batches drawn uniformly from all 60000 training
images; labels and the test split are never used in training.

`evidence/mnist-inputs.png` (command: `uv run python
scripts/plot_mnist_inputs.py`) was reworked per review and re-inspected: a
GridSpec with four image columns plus a **dedicated narrow color-bar column**
for the digit block, and a **separate noise-row color bar**; the script
asserts that no two axes bounding boxes overlap before saving, and the PNG is
written with `bbox_inches="tight"` so no label or title is cropped. The
rendered figure shows 16 preprocessed digits with color bars, and one image at
t = 0, 200, 500, 1000 - the border is -1 and structure disappears as t
increases (t = 1000 is pure noise, std 1.014, mean -0.009).

## Part 2: cloud approval packet (approved and executed; instance release pending)

`cloud/runbook.md` (rewritten in English from the human's Chinese A10 guide
`阿里云A10租用操作.md`) provides a **manual console sequence first** (no CLI or
account configuration assumed) plus the gated CLI path: interactive
`aliyun configure` (OAuth preferred, AK fallback via prompts; keys never in
history or git), read-only `DescribeZones`/`DescribeInstanceTypes`/
`DescribeAvailableResource`/**`DescribePrice`** (spec: 8 vCPU, 30 GiB, A10
24 GB; system disk 60 GiB; PayByTraffic with explicit bandwidth cap), local
RSA key pair with public-key-only `ImportKeyPair` (never `CreateKeyPair`,
whose private key goes to stdout), `RunInstances` with required `VSwitchId`
and matching disk settings and no password, deployment of the **current local
tested source** via `git archive` (tracked files only - no `.venv/`, secrets,
`inputs/`, `runs/`, `artifacts/`), separate `mnist.npz` upload with SHA256
verification (731c5ac6...), a GPU assertion (`platform == "gpu"`, driver
>= 525 for CUDA 12) before training, tmux + saved logs, copy-back into a NEW
retrieval folder with remote/local SHA256 and 30000-record verification before
promoting the checkpoint, and gated release as **StopInstance ->
`DeleteInstance --Force false` -> `DescribeInstances` for the exact ID**
(release is DeleteInstance; it permanently deletes disks and needs a separate
explicit human confirmation). Every resource-creating/releasing command is
marked `[GATED]`; none was executed, and the actual price is UNKNOWN.
`cloud/README.md` records that this task created no resources (it makes no
claim about the account's global state or costs).

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

Part 2 is complete: the formal 30000-update run was trained on the
human-purchased A10 (root's flock-locked run; exit 0, 368 s), independently
verified, copied back with SHA256 equality, promoted to
`checkpoints/mnist.npz`, and the loss/grid/trajectory evidence was generated
from a local-CPU preview sampling of the formal checkpoint. The initial
setup race (two training processes, stopped by root, output preserved
remotely) is recorded honestly in `cloud/README.md`; that run was never
promoted. The instance `i-t4n7f2zqpa44cb3hh7uc` is release-ready pending the
human's exact-ID confirmation; actual bill pending. The official viewer PNG
at t = 200 is to be saved by the human from trajectory.jsonl; the committed
`evidence/mnist-denoising.png` is a labelled matplotlib substitute. Parts 3-5
have not started. No push yet.
