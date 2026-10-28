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
