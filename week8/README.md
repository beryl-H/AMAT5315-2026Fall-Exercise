# Week 8: Autoregressive models, scaling and grokking

Specification: `SPEC.md` (fixed acceptance criteria, declared before
implementation). Environment: Python 3.11, JAX 0.6.2 CPU, numpy 2.2.6
(`uv.lock`). Corpus `data/input.txt` (git-ignored), SHA256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

Tests: `uv run pytest tests/ -q`. Tests were written before each
implementation; the preserved failing logs are under `evidence/test-logs/`.

Agent record: implemented with native WSL pi (provider `volcengine-plan`,
model `glm-5-3-flash`); logs in `/mnt/d/Keria/Course/AMAT5315/week8-support/`.
Raw run artifacts (checkpoint weights, logs) are git-ignored in `artifacts/`.

## Part 1 — Text into a probability model

Question: how much held-out likelihood does each step of context buy, under a
fixed corpus, vocabulary and block protocol?

Producing command:

```bash
uv run python run_part1.py
```

Protocol: sorted 65-character vocabulary; training = first 1,003,854
characters; validation = last 111,540; validation scored in blocks of 128
inputs predicting the shifted-by-one targets (871 full blocks + one final
block of 51) so each of the 111,539 validation characters except the first is
scored exactly once. Losses are token-weighted nats (sum −ln p and counts
before dividing); PPL = exp(loss). Unigram/bigram use add-one smoothing
fitted on training text only.

Measured results (evidence/part1-summary.json):

| predictor | loss (nats/char) | PPL |
|---|---|---|
| uniform | 4.174387 (= ln 65) | 65.000 |
| unigram (add-one) | 3.347305 | 28.426 |
| bigram (add-one) | 2.481889 | 11.964 |
| two-target example (1/2, 1/4) | 1.039721 | 2.828427 |

All six SPEC acceptance checks PASS. Each step of context lowers the held-out
loss: frequency (−0.827 nats vs uniform) and one character of context
(further −0.865 nats).

Evidence figures:

- `part1-unigram.png` — unigram probabilities with the uniform 1/65 level;
  shows which characters carry the frequency gain (space is the most
  frequent, near 0.15).
- `part1-bigram-after-t.png` — P(next | 't') distribution; 'h' takes 0.341,
  showing one character of context already captures much of English spelling.
- `part1-scores.png` — held-out loss and PPL bars for the three predictors.

## Part 2 — Causal transformer on a CPU

Question: can a small transformer trained on a CPU beat the bigram while its
earlier predictions provably ignore later characters?

Producing command (formal run, exclusive-new attempt, preflight-checked):

```bash
JAX_PLATFORMS=cpu taskset -c 0-3 uv run python run_part2.py --attempt formal-01
```

Architecture: pre-LN decoder, token+position embeddings, 2 layers × width 64,
4 causal attention heads, MLP 256, untied head with bias — exactly 116,673
parameters (= 24·64² + 286·64 + 65). Training: 3000 AdamW steps, batch 16,
lr 3e-3 with 100-step warmup then cosine to 3e-4, weight decay 0.1, seed 1337.

Formal measured results (attempt `formal-01`, exit code 0, 160.8 s train):

- Full held-out loss (block protocol, all 111,539 targets): **1.699743**,
  PPL **5.473** — below the 1.85 gate and the bigram 2.481889 (all seven
  SPEC gates PASS; no gate was tuned or rerun).
- Causality probe over 8 corpus window pairs sharing a 64-character prefix:
  **exactly 0.0** — earlier logits are bitwise unchanged by later characters.
- Attention audit: largest above-diagonal weight **exactly 0.0**, largest
  row-sum deviation 2.4e-7, every layer/head.
- Unmasked negative control (300 steps, no mask): probe scored WITHOUT the
  mask is **2.996** (leakage detected); the same model scored WITH the mask
  gives exactly 0 by construction — the control validates the audit.
- Generation (500 chars, prompt `"ROMEO:\n"`, seed 7, reloaded+hash-verified
  checkpoint): temperature 0.5 produces speaker names, line breaks and
  English-shaped words; temperature 1.0 is noisier with more invented words
  (`evidence/part2-sample-temp*.txt`).

Evidence figures:

- `part2-loss-curves.png` — training and validation loss vs step with the
  unigram 3.347 and bigram 2.482 levels; the transformer crosses the bigram
  early and settles at 1.70.
- `part2-attention.png` — first-layer attention heads for
  `"ROMEO:\nBut soft, what light"`; the area above the diagonal is exactly
  zero (the causal mask) and rows sum to one.
- `part2-sample-temp0.5.txt`, `part2-sample-temp1.0.txt` — the two samples.

Input fingerprint and environment (all parts): corpus `data/input.txt` SHA256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`; learning-sheet
PDF SHA256 `4a9f5d0cc5223ed6c1863410cc1f4e03b890dba21649265ca8a91eacd0b95101`;
Python 3.11.17, JAX/jaxlib 0.6.2 (TFRT_CPU_0, CPU-only, `JAX_PLATFORMS=cpu`),
numpy 2.2.6, matplotlib 3.10.3, uv 0.13.0 (`evidence/environment.json`).

## Part 3 — Scaling law and blind prediction

Question: does held-out loss follow a power law in parameters, and can the
loss of a LARGER, never-fitted model be predicted BEFORE training it?

Protocol (frozen in `SPEC.md` before any run): widths 8–128 × seeds 1,2 at a
fixed 1500-step budget; width 8 excluded (bigram plateau); loss-space WLS fit
of `L = a·N^(−α)` with per-width σ = |L₁−L₂|/2 (floor 1e-3), Birge factor
max(1, √(χ²/6)); deterministic 10,000-draw bootstrap (seed 20260). The blind
prediction for width 192 (890,112 non-embedding parameters) was sealed to
`artifacts/part3/formal-fit-01/prediction/prediction192.json` (SHA256
`74f6c3da…92bc5b`, timestamp **2026-10-28T04:20:00Z**) BEFORE the width-192
runs started (first worker start **04:20:15.6Z**, dispatch 04:20:13.1Z); the
prediction is older than every training step (anti-leakage guard PASS).

Formal result — a legitimate FAILED blind prediction:

- Sealed prediction: **1.639543**, 95% interval **[1.611909, 1.667537]**
  (width 0.0556).
- Actual width-192 holdout losses 1.714366 / 1.730032, seed mean
  **1.722199** — OUTSIDE the interval by 0.0547 nats.
- Fit over widths 12–128: α = **0.0648**, 95% CI [0.0589, 0.0708];
  Birge **1.853**. α is in the expected 0.05–0.10 band and the interval
  width is < 0.2 — the prediction machinery worked, the extrapolation missed.
- No refit, no interval widening, no seed search, no retraining, no target
  leakage: `prediction192.json` is immutable.

Producing command for the figure and summary (read-only rebuild from the raw
18 losses, the sealed receipt and the two holdout runs):

```bash
uv run python make_part3_evidence.py
```

- `part3-scaling-blind-miss.png` — seed points/means, the pure power-law fit
  with its 95% bootstrap band, the sealed blind prediction diamond and the
  trained width-192 point ABOVE the interval (the honest MISS).
- `part3-scaling-summary.json` — rebuilt fit, regenerated bootstrap
  quantiles (match the sealed interval), gate booleans (the inside-interval
  gate is FALSE by design of the honest report), hashes and timestamps.

Scientific interpretation, stated cautiously: at a fixed 1500-step budget the
larger model has less compute per parameter, so compute-limited flattening at
width 192 is a plausible explanation — but it is NOT proven; we ran no
additional experiments (e.g. longer budgets) to distinguish mechanisms. The
sheet's earlier floor-term (Protocol A) blind failures are COURSE REFERENCE
HISTORY, not experiments performed here.

## Part 4 — Grokking (formal-decoder-02): TRAINING STILL RUNNING

Status: **Part 4 is running until its final summary is available.** The 12
fixed jobs (main seeds 0–4 at 30%/wd 1; wd-0 seeds 0–2; 50%-fraction seeds
0–2; one random-label run, seed 0, 3000 steps) are coordinated by Codex under
`artifacts/part4/formal-decoder-02/manager-status.json`. Interim numbers MUST
NOT be read as conclusions; this section will be completed with the measured
evidence only after all runs finish.

Protocol: `PART4_DECODER_PROTOCOL.md` (frozen before any decoder run, at
commit 473efb3; `grok.py` SHA256 `d0501c3b…1d447`). One-layer decoder, NO
LayerNorm, width 64, 4 heads, MLP 256, learned 3×64 positional embeddings,
causal mask; input [a, b, '='], answer read at '='; 62,497 parameters;
full-batch AdamW lr 3e-3, betas (0.9, 0.98), wd 1 (0 for the wd0 controls);
log every 100 steps plus step 0; stop at first logged test acc ≥ 0.99 plus
1000 steps, cap 10000.

Invalidated predecessor (preserved, excluded): `artifacts/part4/formal-01`
used a symmetric variant (no positional embeddings, no causal mask). It
cannot represent the random-label control: 393 conflicting unordered training
pairs imply a maximum reachable training accuracy of 0.860786 < 0.99. It was
stopped with all files preserved and is NOT mixed into any decoder result.
The correction (learned positions + causal mask, only LayerNorm removed) is a
verified capacity fix documented in the protocol, not metric tuning.

Postprocessing plan (implemented now, executed only after completion):

```bash
# only when manager-status.json reports COMPLETE and all 12 metrics exist;
# otherwise the script exits nonzero WITHOUT writing final evidence:
uv run python make_part4_evidence.py
```

will produce `part4-grok-curves.png` (accuracy/loss/parameter norm vs step,
log-x, step-0 initialization annotated, all 5 main seeds visible including
the no-grok seed), `part4-controls.png` (test accuracy vs step for main, wd0,
50% and random-label controls with the chance 1/97 line),
`part4-fourier.png` (Fourier power
k=1..48 at initialization/memorization/final per seed, final key frequencies
marked, own-snapshot top-5 shares printed; snapshots are labelled FINAL when
no grok occurred and show an explicit "unavailable" when a memorization
checkpoint is absent) and `part4-ablations.png` (trained / keep key5 / remove
key5 / 10 random-other draws each, with the 0.95/0.05 gates), plus
`part4-summary.json` and `part4-summary.md` (per-seed memorization step,
grok step, delay ratio, final accuracies, top-5 share, all ablations and the
exact gate booleans: grok within 10000, delay > 10, share > 0.8, keep key
> 0.95, remove key < 0.05, remove other > 0.95 in all 10 draws, keep other
< 0.05 in all 10 draws — failed gates stay failed). Source/protocol/metrics
hashes, the actual worker commands, the CPU-only environment, the invalidated
predecessor exclusion and artifact paths are recorded in the summary. A
`--preview` flag exists for nonformal rendering checks into
`evidence-preview/` only.
