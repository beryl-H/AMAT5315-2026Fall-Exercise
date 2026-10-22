# Week 7 execution and independent verification

Authority: the updated 19-page **Diffusion and flow matching** learning
sheet, SHA256 `4147c45cc6ac9f6e1815ba33684a1c4ed50c87d8d89ffb14f3dccab6bcfddbd5`.
The old inverse-problems exercise remains in the parent directory. Its
historical implementation was direct Codex work; it is not attributed to pi.

## Agents and execution boundary

Implementation of this updated exercise was assisted by the **native WSL
pi** executable under Ubuntu-24.04, using the existing `agent-plan` provider,
`glm-5.3-flash` model and Ark Coding Plan endpoint. The coordinator used:

```text
pi --offline --approve --no-extensions --provider agent-plan \
   --model glm-5.3-flash --thinking low --mode json --print \
   --session-dir <private task session directory> \
   --session-id 52dc4b20-847e-46de-b05f-b9d8c97d5aee @<part-specific prompt>
```

The wrapper checked the native executable and provider/model settings and
injected the existing credential through an environment reference. No keys
are in this repository. [Agent execution receipts](evidence/agent_execution.json)
retain tool counts and real stop/exit statuses; private sessions/JSONL,
reasoning, credentials, downloaded weights and environments are excluded.

Codex coordinated and independently verified equations, training updates,
raw trajectories, RNG streams, numerical summaries, figures and delivery.
Codex also made explicitly recorded review fixes: portable checkpoint-header
fixture/scalar handling, Week6 NHWC gallery repair, final completion-marker
ordering and postprocessing hash checks, the independent fresh-path
regression test, and documentation consolidation. A separate read-only
reviewer inspected the committed change and the corrected follow-up.

The original pi hit `AccountQuotaExceeded` before Part5 implementation on
October21. The user selected continuation with the same pi after October22
02:35 Asia/Shanghai. That choice arrived after the requested time, so work
resumed immediately; no recurring automation or provider/model substitution
was created. Coordinator interruptions preserved files and logs.

## Measured checks and limits

| Part | Independently checked result | Status |
|---|---|---|
| 1 | 25,000 updates x512, seed2026; ten checkpoints; forward moments, generated populations/centres/widths | PASS |
| 2 | DDIM all-state repeat difference0; DDPM same-cluster fraction0.2455; DDIM fractions | PASS |
| 2 | Strict coordinate widths in0.29–0.32; minima0.279478 and0.283907 | FAIL retained |
| 3 | 25,000 updates x512, seed2029; 29 validated studies/58 distinct keys; flow20 energy0.000265740 and width error0.0261169 | PASS |
| 3 | All81 fixed grid points, relative RMS0.3361455/0.2852369/0.0725452 | FAIL at s=.1,.5 |
| 4 | All328 arrays/35,746,307 parameters exact; JAX vs official PyTorch max abs<=2.384e-6, relative RMS<=3.328e-7 | PASS |
| 4 | Four1000-step DDPM images, seed7401, all1001 states finite, weights unchanged, no state clipping | PASS; recognizability qualitative |
| 5 | Sixteen completions,396 merged event states,1192 distinct normal-draw keys, sixteen real-model transition replays | PASS |
| 5 | All16 observed maxchanges0, outside-hole variance0, inside-hole disagreement positive; hidden MSE average0.229608085 | Invariants PASS; MSE descriptive |

Part1 and flow training took87.3s and71.7s respectively; Part4 generation
took150.5s. These are measured local CPU timings, not universal performance
claims. The separate official reference used PyTorch2.8.0+cpu,
diffusers0.40.0, safetensors0.8.0 and NumPy2.2.6. Optional accelerate was
absent; the loader used its memory-mode fallback and numerical checks passed.

The Part3 accepted study is `artifacts/part3-validated/raw/`. The earlier
key-reuse study is explicitly `INVALID_RNG` under `artifacts/part3/` and was
not used for accepted plots/metrics. Raw signed energy estimates are retained;
only the logarithmic plot floors at1e-4. The20-pair null95% quantile is
0.001924273. On-marginal field errors are supplementary, not a substitute
for the failed uniform-grid gate. Gaussian tails have very low positive
density. Exact-target width fluctuations suggest a sampling contribution
but do not prove the learned predictor/sampler is error-free.

## Part5 original failure and saved-data recovery

The formal inference source was commit
`0a0c8a4a05055017a00f9351ab0e94405fdd8547`. It generated all16 completions
and all4x99 merged states in66.127564s, saving the actual keys and inputs.
The unchanged sampler SHA is
`d3cdba112ff74e2da098a115137568c2068b7fa208d35d24192b093890b61720`.
The driver then failed on a stale final summary-key lookup: **actual exit1**,
after raw arrays and metrics had been saved. The failure log was retained;
the original metadata/raw data were not overwritten or regenerated.

```bash
.venv/bin/python scripts/part5_postprocess.py
.venv/bin/python scripts/part5_figures.py
```

The saved-data command has no UNet/sampler calls. It recomputes all MSEs,
known-pixel invariants and variance, checks actual model/input/source file
hashes, and writes a separate recovery receipt. The future inference entry
point now refuses populated partial/completed directories and writes its
completion marker only after raw/metrics/checks/summary have succeeded.
This IO correction was reviewed without repeating the formal inference.

The coordinator separately replayed initialization and all forward-noised
known pixels, checked all1192 distinct normal-draw keys, and recomputed
16 selected transitions with the real frozen JAX model. The maximum
transition error was6.931e-7 against the predeclared1e-5 tolerance. Every
reported hidden MSE was recomputed from exactly1200 hidden values/image.
Mean and population variance were checked from all four seeds; raw
per-channel values and a shared RGB-mean variance scale are retained.
Variance measures disagreement, not calibrated uncertainty or error.

## Current verification and files

Executed checks include:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts tests
uv lock --check --offline
git diff --check
git diff --cached --check
```

The latest command outputs/exit statuses and source hashes are in
`evidence/final_checks.json`. The independent audits are in
[evidence/independent](evidence/independent). The portable fresh-path tests
use NumPy float64 equations and the declared RNG protocol against the actual
compiled production branch, and reject the coarse-DDPM-variance control.
Earlier checkpoints were82 tests at Part4,92 at Part5 engineering, and93
after wrapper validation; the final suite also includes the coordinator's
two independent fresh-path cases.

New source, tests, scripts, locked environment, small JSON summaries and
figures live in `week7/diffusion-flow/`. The repository and parent Week7
README gain current-topic pointers; the complete old parent README body
is preserved. Valid raw runs, frozen model files, the PDF and the required
Week6 gallery dependency are retained in the local final package. Available
failed/invalid runs and private transport/agent logs remain at their original
task-local locations. No files, branches, repositories or environments were
deleted, and no force push is used.

Corrections found during the task include cumulative-alpha indexing,
reverse-key reuse, checkpoint ordering, plotting interfaces, the invalid
first Part3 RNG protocol, UNet convolution bias/linear orientation, fixture
portability and Week6 NHWC layout. Later review found stale README tables
(including a transposed Part5 MSE table), a missing driver import and
completion-marker/reporting issues. These were corrected and checked;
the standing numerical FAIL results remain visible. Nothing is submitted
to a course platform; the evidence is prepared for the start of Week8.

The final environment check found that the earlier lock omitted the already
installed/directly pinned safetensors0.6.2 added for Part4. A normal verified
PyPI metadata resolution added only that exact version to uv.lock; no
installed environment or other dependency version was changed. The initial
offline failure and the one trailing-space documentation failure are retained
in separate verification logs. The transport command's deprecated native-TLS
flag is not required by the reproducible setup commands.
