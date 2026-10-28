# Week 8 SPEC: Autoregressive models, scaling and grokking

Source: `week8-learning-sheet.pdf`, SHA256
`4a9f5d0cc5223ed6c1863410cc1f4e03b890dba21649265ca8a91eacd0b95101`
(extracted text: `/mnt/d/Keria/Course/AMAT5315/week8-support/learning-sheet.txt`).
Correctness is defined by that sheet before any implementation. CPU only
(uv + JAX CPU). All acceptance numbers below are fixed in advance and must not
be weakened to make a run pass. Failures are reported honestly.

## Common

- Corpus: TinyShakespeare, SHA256
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`,
  1,115,394 bytes. `data/` and `artifacts/` are git-ignored; `evidence/`
  figures and compact JSON summaries are tracked.
- Environment: Python 3.11 venv via uv; jax==0.6.2, jaxlib==0.6.2,
  numpy==2.2.6, matplotlib==3.10.3, pytest==8.4.2 (locked in `uv.lock`).
- Every evidence figure has a README entry giving its producing command and
  the scientific question it answers.
- Tests are written before implementation; the failing pytest log of each
  test-first step is preserved under `evidence/test-logs/`.
- Agents: pi (native WSL, provider `volcengine-plan`, model `glm-5-3-flash`)
  assisted implementation; coordinating Codex audits independently. Recorded
  per run in `artifacts/<run>/manifest.json`.

## Part 1 — Text into a probability model

Protocol (fixed):

- Vocabulary: the 65 characters of `data/input.txt`, sorted ascending; id =
  rank in sorted order.
- Split: training = first 1,003,854 characters, validation = last 111,540.
- Scoring blocks: validation is scored in blocks of 128 inputs each
  predicting the next character at every position, targets never overlapping:
  block `i` (0-based, full blocks) inputs `val[128i:128i+128]`, targets
  `val[128i+1:128i+129]`. After 871 full blocks (111,488 scored targets) a
  final block inputs `val[111488:111539]` (51 inputs) and scores the last 51
  targets `val[111489:111540]`. Exactly 111,539 validation characters are
  scored, each exactly once; the first validation character only supplies
  context.
- Loss: token-weighted cross-entropy in nats = (sum of −ln p over scored
  targets)/(number of scored targets); PPL = exp(loss). Perplexities are
  never averaged per block.

Acceptance criteria:

1. `data/input.txt` SHA256 equals the fingerprint above.
2. Printed counts are exactly 1,003,854 train / 111,540 validation / 111,539
   scored.
3. Uniform predictor: loss = ln 65 = 4.174387 (7 d.p.), PPL = 65 exactly.
4. Two-target example (probabilities 1/2 and 1/4): loss = 1.039721 (7 d.p.),
   PPL = 2.828427 (6 d.p.). (Averaging reciprocal probabilities would print
   3 — this must NOT appear.)
5. Unigram (add-one smoothing on training text only): loss 3.347305 ± 1e-5,
   PPL 28.426 (3 d.p.).
6. Bigram (add-one, conditioned on previous character, training text only):
   loss 2.481889 ± 1e-5, PPL 11.964 (3 d.p.).
7. Monotone: uniform loss > unigram loss > bigram loss.
8. Figures: (a) unigram probabilities as bars with the uniform 1/65 level
   marked (space ≈ 0.15); (b) bigram distribution after `t` (h ≈ 0.34);
   (c) held-out loss and PPL of the three predictors as bars.

## Part 2 — Causal transformer on a CPU

Architecture (fixed; parameter count is an acceptance criterion):

- Pre-LN decoder: learned token embedding 65×64 and learned absolute
  position embedding 128×64; 2 layers, each = LayerNorm → causal
  self-attention (4 heads, head width 16, one Wq/Wk/Wv/Wo each 64×64 with
  biases) → residual → LayerNorm → MLP 64→256→64 with biases → residual;
  final LayerNorm; untied output head 64→65 with bias. Softmax over 65
  characters. Total parameters: exactly 116,673 (reference value; any
  mismatch must be explained, not hidden).
- Training: 128-character windows of training text; AdamW 3000 steps,
  batch 16, lr 3e-3 with 100 warmup steps then cosine decay to 3e-4,
  weight decay 0.1, seed 1337. Log train and validation loss; validation
  curve points use the first 96 blocks of the validation split; save final
  weights as a checkpoint; then score the FULL validation text with the
  Part-1 block protocol.

Acceptance criteria:

1. Parameter count of the implemented model equals 116,673 exactly.
2. Full held-out loss (block protocol, token-weighted) < 1.85 — a numerical
   gate, not a guarantee; the actual value is reported whether it passes or
   not. Must also beat the bigram 2.481889.
3. Causality probe: 8 pairs of 128-character windows that agree in their
   first 64 characters and differ afterwards; the largest change in the
   logits of the first 64 positions must be exactly 0.0.
4. Attention audit: in every layer and every head, all above-diagonal
   weights are exactly 0 and every row sums to 1; the largest above-diagonal
   weight is printed. Figure: attention weights of every head of the first
   layer for the prompt `"ROMEO:\nBut soft, what light"`.
5. Unmasked negative control (predeclared: the same model trained 300 steps
   with the mask removed, seed 1337): its probe must be > 0 (leakage
   detected). This is the sabotage control for criteria 3–4.
6. Generation: 500 characters from the prompt `"ROMEO:\n"` at temperatures
   0.5 and 1.0, seed 7, sampling from the saved final weights. Text shows
   speaker names, line breaks, English-shaped words; temperature 1.0 noisier
   than 0.5.
7. Figure: training and validation loss vs step with unigram and bigram
   levels drawn.

## Part 3 — Scaling law and blind prediction (SPEC ONLY; formal runs not launched this checkpoint)

Protocol and predeclared conventions (fixed before any Part-3 run):

- Models: 2-layer, 2-head decoders of widths 8, 12, 16, 24, 32, 48, 64, 96,
  128, seeds 1 and 2; identical budget: 1500 steps × 16 windows × 128
  characters (3.1M characters), Part-2 schedule adapted to 1500 steps
  (warmup 100, cosine 3e-3 → 3e-4, AdamW wd 0.1). N = non-embedding
  parameters = total − (token emb + position emb + output head).
- Execution: one training per CPU worker, at most 4 concurrent workers,
  total WSL memory held < 6 GB. Full held-out loss via the Part-1 block
  protocol for every run.
- Fit (corrected per spec-review-01, frozen before any Part-3 run): width 8
excluded (bigram plateau). Fit L(N) = a·N^(−α) by direct nonlinear weighted
least squares IN LOSS SPACE on the 8 seed-mean losses of widths 12..128,
minimizing χ² = Σᵢ ((Lᵢ − a·Nᵢ^(−α))/σᵢ)². Conditional on α the scale is
profiled analytically: a(α) = Σᵢ LᵢNᵢ^(−α)/σᵢ² / Σᵢ Nᵢ^(−2α)/σᵢ², leaving a
1-D search over α. Per-point uncertainty is the standard error of the seed
mean of two runs: σᵢ = |L(seed1) − L(seed2)| / 2 (the sample standard
deviation of two runs is |L1−L2|/√2; the mean's standard error divides by
√2 again), floored at 1e-3 if zero (zero-scatter convention), then inflated
by max(1, √(χ²/dof)) (Birge), dof = 8 − 2 = 6.
Uncertainty (declared before runs): deterministic parametric bootstrap IN
LOSS SPACE, 10,000 draws, fixed seed 20260: for each width draw synthetic
seed means Lᵢ* ~ Normal(fitted L(Nᵢ), σᵢ·Birge), refit (a, α) by the same
profiled WLS, predict L(192), take the 2.5–97.5 percentile interval. The
same bootstrap also yields the α interval.
- Blind protocol: the width-192 prediction with interval, UTC timestamp and
  SHA256 of the prediction file are saved BEFORE either width-192 training
  starts; then width 192 is trained with both seeds. The fit is never
  touched after width-192 results are seen. Interval width must be < 0.2
  nats; α expected within 0.05..0.10. Trained mean inside or outside the
  interval is reported honestly either way.
- Figures: log-log seed means with fit, 95% band, excluded width 8 and
  bigram level; α with uncertainty and Birge ratio printed.

## Part 4 architecture and policy freeze (before any formal run)

Choices not fixed by the sheet, frozen here before implementation:

- Vocabulary: 98 tokens = residues 0..96 plus `=` (id 97). Input sequence is
  exactly `[a, b, =]` (3 tokens); NO positional embeddings (the `=` token
  marks the answer slot); NO causal mask (all three inputs are given, the
  answer never appears as input). Answer read from logits at position 2,
  softmax over 98 tokens, cross-entropy on the true residue c.
- One layer, no LayerNorm, width 64, 4 heads (head width 16, qkv + out
  projections with biases), MLP 64→256→64 with biases, untied head 64→98.
  Total parameters = 98·64 + (3·(64·64+64) + 64·64+64) + (64·256+256 +
  256·64+64) + (64·98+98) = 62,370. All weights init N(0, sqrt(2/(64+97))),
  biases zero (gives initial parameter norm ≈ 28, the reference value).
- AdamW lr 3e-3, betas (0.9, 0.98), weight decay 1, FULL batch (all training
  pairs each step), no gradient clipping, float32.
- Logging grid: every 100 steps AND step 0 (initialization: loss, accuracy,
  parameter norm recorded before any update). Memorization step = first
  LOGGED step with train accuracy ≥ 0.99; grokking step = first LOGGED step
  with test accuracy ≥ 0.99; stop rule: stop at grok_step + 1000, capped at
  10000 steps. Crossings from the actual log grid only; checkpoints saved at
  initialization, memorization and final, incrementally, per-run exclusive
  attempt directory.
- Parameter norm ‖θ‖₂ over ALL parameters (weights and biases).
- Fourier convention: np.fft.rfft of each residue-embedding dimension over
  the residue index a = 0..96; power_k = Σ_dims |X_k|² for k = 1..48 (97 odd
  ⇒ k 1..48 are exactly the non-DC bins; conjugate pairs folded by |·|²).
  Top-5 share denominator = total power over k=1..48 (DC EXCLUDED from the
  share; DC is preserved in every ablation). `=` embedding row is never
  part of the DFT or any ablation. Top-5 at each snapshot from that
  snapshot's own spectrum.
- Ablation (final model): keep-only K (E_res ← DC + Σ_{k∈K} trig components),
  remove K (E_res ← E_res − Σ_{k∈K}), remove/keep 5 random OTHER frequencies
  (10 draws, one rng `np.random.default_rng(4497)`, drawn from the
  complement of the final top-5 within 1..48, no overlap).

## Part 4 — Grokking modular addition (protocol; formal runs not launched)

Protocol and predeclared conventions (fixed before any Part-4 run):

- Task: c = (a+b) mod 97, all 9409 ordered pairs; input tokens a, b, `=`,
  answer read at position of `=`. One-layer pre-LN-free transformer
  (no LayerNorm), width 64, 4 heads, MLP 256, residue embeddings 97×64 plus
  `=` embedding.
- Split (label-independent, frozen): one rng `np.random.default_rng(897)`,
  permutation of the 9409 pair indices; the same permutation is reused for
  every Part-4 run. Training sizes (coordinator instruction 2026-10-28,
  superseding the earlier floor note): 30% split = round(0.30·9409) = 2823
  pairs; 50% split = round(0.50·9409) = 4704 pairs of the same permutation
  (nested). The split never sees labels; train/test indices never overlap.
- Random-label control (corrected per spec-review-01, declared before
  runs): a GLOBAL class permutation π((a+b) mod 97) would preserve the
  modular-addition structure, so it is NOT used. Instead the vector of all
  9409 true labels is permuted ACROSS PAIR INDICES with one fixed declared
  rng `np.random.default_rng(8970)`: labels* = labels[permutation(9409)].
  This preserves class frequencies exactly while destroying the
  pair-to-label relationship, and applies to train and test pairs alike.
  Chance test accuracy is 1/97 in expectation but is NOT a guaranteed
  synthetic outcome and is not treated as one. One run, main settings,
  seed 0, 3000 steps.
- Main runs: full-batch AdamW, lr 3e-3, betas (0.9, 0.98), weight decay 1,
  up to 10000 steps, stop 1000 steps after test accuracy ≥ 0.99; log train/
  test loss, accuracies and parameter norm every 100 steps (with the
  initialization values as step-0 record, preserved crossings); seeds 0..4.
  Controls: seeds 0..2 with wd 0; seeds 0..2 with 50% split.
- Fourier analysis: DFT of each residue-embedding dimension over the residue
  index a = 0..96; power at frequency k = sum of squared cos+sin coefficients
  over dimensions, k = 1..48 (DC k=0 excluded from the top-5 selection but
  never ablated). Top-5 frequencies and their power share recorded at
  initialization, at memorization (first logged step with train acc ≥ 0.99)
  and at final. Ablations on the final model: keep only top-5, remove top-5,
  remove 5 random other frequencies (10 fixed draws), keep only 5 random
  other frequencies. The `=` token embedding is separate and never part of
  the DFT/ablation. Top-5 selection at memorization and final is from that
  snapshot's own spectrum.
- Gates: every seed groks within 10000 steps; delay ratio (grok/memorize
  step) > 10; post-grok top-5 power share > 0.8; keep-only-top-5 test acc
  > 0.95; remove-top-5 or keep-only-others test acc < 0.05; removing other
  frequencies keeps acc > 0.95. Any gate failure is reported honestly; no
  seed hunting.
- Figures: acc/loss/norm vs step (log axis) all seeds; test acc vs step for
  main/wd0/50%/random-label with chance line; Fourier power at the three
  snapshots; ablation bars.

## Implementation plan (checked in order)

- [x] T0 Setup: week8/ project, uv lock, corpus download + fingerprint check
- [x] T1 SPEC.md with fixed acceptance criteria (this file)
- [x] T2 Part-1 tests written first; failing log preserved; commit (dd50d27)
- [x] T3 Part-1 implementation and formal scoring run; all six acceptance
      checks PASS (commit d09f365, evidence/part1-summary.json)
- [x] T4 Part-2 tests written first; failing log preserved; commit (ece62db)
- [x] T5 Part-2 implementation + unit negative control; 18/18 tests PASS
      (commit 0a624de)
- [x] T6 Part-2 formal run (attempt `formal-01`, exit 0): full validation
      loss 1.699743 < 1.85 gate, probe exactly 0.0, attention audit clean,
      negative control detects leakage, samples + figures + README
      (commit 91f87b8)
- [x] T7 Chinese learning-and-demonstration guide written to
      `/mnt/d/Keria/Course/AMAT5315/week8-support/week8-学习指南.md` with
      concise final results of Parts 1-2
- [x] T8a Part-3 code prepared and tested per corrected SPEC (scaling.py,
      sweep.py, generic-width transformer; 49/49 tests; smoke-01 with 2
      steps marked nonformal; width-192 shape smoke without optimizer)
      (commit 2974088) — formal sweep NOT launched; prediction receipt not
      yet written
- [ ] T8b (after independent audit) Part-3 sweep of widths 8..128 × seeds
      1,2, immutable prediction192 receipt, then width-192 × seeds 1,2
- [ ] T9 (after independent audit) Part-4 runs, Fourier analysis, ablations

Pre-run corrections applied from spec-review-01.md (commit history
preserved; no budgets/seeds/tolerances changed): Part-3 SEM derivation,
loss-space WLS + bootstrap; Part-4 label permutation across pair indices,
floor-based split sizes.
