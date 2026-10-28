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
