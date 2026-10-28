# Week 8 (revised): Language models and the transformer — SPEC

Authoritative source: `week8-learning-sheet.pdf` (24 pages), snapshot SHA256
`4e33bea929459c01511efc0298308a93d08d99ee7df38372bbf52e89be69cc35`, text
extraction in `/mnt/d/Keria/Course/AMAT5315/week8-support/revised-4e33bea92945/learning-sheet-agent.txt`
(page images of key pages in the same folder). The Part 4 glyph is unavailable in
the PDF text layer; the user explicitly selected **算** (user-choices.json).

This is a wholly new six-part task. The pre-existing `week8/` work (commit
24a860b, 12-page-sheet era) is historical and must remain byte-identical; all new
work lives in `week8/language-models/`. No old file may be modified, moved or
deleted. Public repo artifacts are in English; the final study guide /
explanations are in Chinese. No passwords, tokens, or raw model reasoning in
committed outputs.

## Environment and tooling (frozen)

- `uv`-managed project in this directory; Python 3.12; CPU default.
- numpy 2.2.6, jax/jaxlib 0.6.2 (CPU), matplotlib 3.10.3, pytest 8.4.2,
  sentencepiece 0.2.2, safetensors 0.8.0, scipy 1.17.1 (`uv.lock`).
- Models in JAX, checks in NumPy. No PyTorch, no cloud services, no downloads
  beyond the pinned files below.
- Layout: source in `src/`, tests in `tests/`, figures + result JSON in
  `evidence/` (committed), raw runs/checkpoints in `artifacts/` and data in
  `data/` (git-ignored). Corpus, raw runs and checkpoints never enter git.

## Pinned inputs (primary public URLs; provenance recorded in data/MANIFEST.md)

| item | repo @ revision | file |
|---|---|---|
| embedding table | `minishlab/potion-base-8M` @ `bf8b056` | `model.safetensors`, `tokenizer.json` |
| corpus | `roneneldan/TinyStories` @ `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | `TinyStoriesV2-GPT4-valid.txt` |
| BPE tokenizer | `karpathy/tinyllamas` @ `0bd21da` | `stories260K/tok512.model` |
| scaling runs | `epoch-research/analyzing-chinchilla` @ `92258837425e1b5f2851d624287f0120583a3d0e` | `data/svg_extracted_data.csv` |

For each: URL, SHA256, byte size, schema are recorded in `INPUTS.md` and
`evidence/input-manifest.json` (published, outside the git-ignored `data/`).
Resolved full commit SHAs are recorded where the public API confirmed them
within a bounded 30 s request; otherwise the pinned short revision is noted as
such. Never download a full model framework. If a file hash or schema
mismatches, report the actual value — no silent fallback, no substitute
worksheet, no external reference-code search.

Download failures are preserved, not hidden: the WSL curl exit-55 failure on
the first Chinchilla attempt and the root's Windows urllib retrieval (receipt
in support/, `chinchilla-download-receipt.json`) are recorded in `INPUTS.md`.

## Global RNG policy (frozen before any formal result)

All seeds are predeclared here; no seed hunting, no threshold relaxation.

| purpose | seed(s) |
|---|---|
| Part 1 control shuffle (single) | 1730 (sheet-specified) |
| Part 1 twenty-shuffle control distribution | 1731..1750 |
| Part 2 ancestral sampler | 531501 (negative control uses the same seed) |
| Part 3 memory-store draws (median of 3) | 101, 102, 103 |
| Part 3 MLP init + data order | 0 |
| Part 4 picture/keys draws: 20 validation draws | 531504..531523 (showcase = 531504) |
| Part 5 training init/data | 0 |
| Part 5 sampler verification | 531505; story openings seed 1848 (sheet); entropy sequences 531506 |
| Part 6 bootstrap resamples | 0..399 |

NumPy `default_rng(seed)`; JAX `PRNGKey(seed)`. Every result JSON names its seed.

## Part 1 — Embeddings: relations as directions

**DO.** Load `potion-base-8M` with `safetensors` (numpy backend, no PyTorch):
E is 29528 × 256. Word → token id via `tokenizer.json`; each word must resolve
to exactly one vocabulary row (single-row lookup; no word-level mean pooling, no
subvocabulary averaging). Report actual OOV/multi-token cases if any. Build the
control copy with rows shuffled by seed 1730.

27 pairs, three relations of nine:
- country→capital: iran tehran, turkey ankara, russia moscow, china beijing,
  france paris, sweden stockholm, japan tokyo, peru lima, kenya nairobi
- male→female: king queen, man woman, he she, brother sister, father mother,
  son daughter, uncle aunt, boy girl, husband wife
- adjective→comparative: big bigger, small smaller, fast faster, dark darker,
  cold colder, old older, strong stronger, long longer, tall taller

Scale each row to unit length; pair offset = unit(b) − unit(a); pair score =
cosine(offset, mean offset of the other 8 pairs) (leave-one-out). Relation mean
= mean of its 9 pair scores. Figure: per relation, all 9 pairs drawn as unit
arrows at angle arccos(cosine) from the horizontal axis, table above the axis,
control below (all pairs shown; nothing cherry-picked).

**VERIFY gates.**
- Table means ≥ 0.3 each (reference: 0.47, 0.55, 0.43).
- Control (seed 1730) means within ±0.15 (reference: 0.03, −0.01, 0.01);
  over 20 shuffles (seeds 1731..1750) no relation mean exceeds ±0.09.
- Embedding key/shape checks: keys match tokenizer ids, shape (29528, 256).

## Part 2 — Count models (Markov chains, n-grams)

**Input preparation (this phase).**
- Split corpus on `<|endoftext|>`, strip surrounding whitespace, drop empty
  stories. First 90% of whole stories (file order) = train, rest = validation.
  Gate: exactly **24,867 train / 2,763 validation** stories.
- Encode both splits with `tok512.model` (SentencePiece), EOS token between
  stories, no BOS, no silent normalizer/preprocessing changes.
- Gate: ≈ 2.05 characters per token on the corpus; decoding returns every
  validation story exactly except **3** (tokenizer merges a double space into
  one) — the 3 exceptions must be identified and recorded, others round-trip.
- Bits-per-character denominators use Unicode characters of the original story
  texts (not decoded text); the EOS contributes loss but no characters; the
  first token of the stream is conditioned on the EOS context. This
  normalization is shared with Part 5.

**DO (later phase).** Uniform + n-gram models n = 1..6 from BPE training-token
counts (memory-efficient sparse counts — never a 512^6 dense table). Add-k
smoothing with k ∈ {1e-4, 1e-3, 0.01, 0.1, 1} chosen on the **last 10% of
training tokens** (fit counts on the inner first 90% for the choice, then refit
on the full training tokens with the chosen k); validation is never used for
tuning. Report held-out cross-entropy (nats/token), perplexity, bits/char.

Normalization check by explicit enumeration: sum the model over every token
sequence of length 1, 2, 3 (all 512, 512², 512³ values actually enumerated in
chunks — not a row-factorization shortcut), each within 1e-12 of 1.

Sampler check: 250,000 ancestral tokens from the bigram (seed 531501); pooled
chi-squared over the next-token counts after the 20 most frequent training
tokens, with rare bins pooled so every expected count ≥ 5; p-value > 0.001
(reference 0.37). Negative control: a sampler conditioning on the token two
places back must give p ≈ 0 (report actual).

Curse of dimensionality: fraction of held-out n-grams unseen in training for
n = 1..6, plotted with held-out bits/char. Gates (within 0.03 of): uniform
6.238, unigram 4.839, bigram 3.038, trigram 2.094 nats/token; best n = 4 at
1.310 bits/char, 1.46 at n = 5, 1.79 at n = 6; unseen fractions within 2 points
of 0.6/4.2/12.2/23.5 % for n = 3..6; perplexities within 3 %.

## Part 3 — The MLP as a key–value memory

**DO.** Memory stores in NumPy: M random unit keys and values in d dims;
recall f(x) = Σ σ(kᵢ·x) vᵢ with stores outer-product, least-squares linear, and
ReLU^p (p = 1, 2, 3). For d ∈ {16, 32, 64}, M = 8..65536 (powers of 2), query
**every** store with **every** stored key (no subsampled queries, no analytic
substitutes). Median of 3 draws (seeds 101–103).

R definition (all stores): with H = K pinv(K) the recall matrix of the
least-squares store, and per store the recall of key k_j written as
f(k_j) = c_j v_j + leak_j with own coefficient c_j, the
interference-to-signal ratio is

    R_j = ‖f(k_j) − c_j v_j‖ / (|c_j| · ‖v_j‖).

- Least-squares store: c_j = h_jj (diagonal of H), leak = recall − own. For
  M ≤ d, H = I, R = 0 exactly. For M > d, expected R ≈ √(M/d − 1); capacity
  (R = 1) ≈ 2d. **Regression test (mandatory):** d = 1, M = 2, keys
  [[1],[1]], values [[1],[1]]: H = 0.5·ones, recall = 1, own = 0.5,
  interference = 0.5, R = 1 (the naive ‖recall − v_j‖/‖v_j‖ metric gives 0 —
  wrong capacity — and is forbidden).
- Outer-product and ReLU^p stores: c_j = σ(k_j·k_j) = 1, so R_j reduces to
  ‖f(k_j) − v_j‖ / ‖v_j‖.

Algorithm: chunked BLAS. No full M×M allocation at any M (65536² float64 =
32 GB). For M ≤ d the least-squares recall uses SVD/`lstsq`/dual form (never
inverting a singular KᵀK). For M > d: solve the d×d system for K⁺V (recall =
K @ (K⁺V)) and compute the H diagonal vectorized as h_jj = Σ_a K[j,a]·(K⁺)[a,j]
via the d×M pseudoinverse — M·d work, no M×M matrix. The three ReLU powers are
evaluated inside each query chunk (shared K Kᵀ tile). A preflight benchmark at
small M estimates the full sweep cost before it is run; allocations are
bounded, never unbounded. Coordinator resource benchmark (support/
memory-tile-benchmark.json): one exact 65536-key tile of 256 queries, all three
ReLU powers, d = 64 float64: 0.486 s, peak RSS 484 MiB, 8 BLAS threads;
extrapolated full sweep (all d, 3 draws, dyadic M) ≈ 871 s. This is a resource
benchmark only, not scientific evidence; the whole-week "~15 min" of the sheet
is the sheet's own claim, not a measured value here.

Capacity = M where the median-R curve crosses 1, interpolated linearly in
log-log; if no crossing is reached inside the grid, the result is reported as
"> 65536" (a lower bound), never misclassified. Plots: R vs M at d = 64 on
log-log with predicted lines (slope 1/2 beyond R ≈ 0.1); capacity vs d with
fitted slopes.

**Gates.**
- *Mandatory:* outer-product crosses R = 1 between M = 50 and 85;
  least-squares R < 1e-10 up to M = 64 and crosses 1 between 100 and 170;
  slope vs d within 0.1 of 1 for outer/LS/ReLU, within 0.15 of 2 for ReLU²,
  > 2.5 for ReLU³. Reference values (67.4; 5e-14; 129; ReLU²/ReLU³ crossings
  2,871/38,124; slopes 0.96/0.99/0.95, 1.96, 2.79) are descriptive unless the
  interval above states a tolerance.

**MLP next-token model.** 64-d embedding, one hidden layer of 512 tanh units,
softmax output over 512 tokens; input = previous BPE token. 4000 AdamW steps,
batch = exactly 16384 examples per step (random positions of the training token
stream; a weighted exact-bigram diagnostic is allowed for testing the optimum
but never substituted for the 16384-example formal batch), lr 0.006, linear
warmup 200 steps, cosine decay to 0.1× peak, Adam betas (0.9, 0.999), weight
decay 0.01 on weight matrices only, init seed 0. Conventional choices not given
by the sheet are frozen here, before any run; they are not fitted to metrics
and not copied from an unknown reference. Optimization may compress duplicated
contexts: with a one-token context the logits for all 512 previous-token ids
can be computed once per step and the 16384 sampled examples gathered (loss and
gradient exactly equal to the direct batch); the 16384 training indices are
still drawn every step, and loss/gradient equivalence with the direct batch
must be demonstrated by test before the compressed path is used. A full-batch
substitution is forbidden. Gate: held-out loss within 0.01 nats of the
bigram's 3.038 (reference +0.003).

## Part 4 — Attention: softmax store recalls a picture

**DO.** 32 × 32 picture of the character **算** (U+7B97; user-selected), pixels
+1 (ink) / −1 (background). Rendering provenance recorded (font name/version,
rasterization protocol, bitmap hash); the copyrighted font file itself is never
committed — only the derived 32 × 32 ±1 bitmap. Store it as one memory among M
in d = 64 (value = the 1024 flattened pixels); the other M − 1 values are random
±1 pictures; all memories get random unit keys (independent). Stores:
outer-product, ReLU, ReLU², softmax at β = 8, for M ∈ {64, 256, 1024, 4096}.

Recall with the stored key; the normalized recall divides the answer by the
stored picture's own weight. R = ‖recall − w_own·picture‖ / (|w_own| ·
‖picture‖), where w_own is the stored key's own softmax/σ weight; the picture
(1024 pixels of ±1) has norm exactly 32 — a fixture test catches any
scalar-denominator (factor-32) error. 20 validation draws (seeds
531504..531523), median reported; the predeclared showcase draw (seed 531504)
is used for the figure — chosen before running, never picked for prettiness.
Raw R of the showcase draw and the 20-draw median are reported for **all** M
and all stores. Figure: one row per M, one column per store, dot area ∝ pixel
value, red when R < 1; print the ISR of every recall.

**Gates.**
- *Mandatory:* softmax store R < 0.1 at **every** M (reference 0.057 at 4096
  vs predicted 0.058 — the value is descriptive, the bound is not);
  ReLU² at M = 4096: R > 1 (reference 1.2); at M = 1024 the outer-product R
  within 15 % of 4 and ReLU R within 15 % of 2.83 (median of 20 draws).
- *Descriptive reference only:* ReLU² at M = 1024 ≈ 0.60 (predicted 0.61) —
  reported as measured, no gate tolerance specified by the sheet.

## Part 5 — Transformer: training and generating

**DO.** Pre-norm transformer decoder in JAX: learned position vectors added to
token embeddings; each block = LN → multi-head causal attention → residual →
LN → GELU MLP (4× width) → residual; final LN + readout; **every** linear map
has a bias. 2 layers, width 96, 4 heads, context 64, MLP 384. 6000 AdamW steps
of 32 sequences (runs of 65 tokens), lr 0.003, linear warmup 300, cosine decay
to a tenth, betas (0.9, 0.999), weight decay 0.01 on weight matrices only,
gradient clip 1.0, float32 on CPU, init seed 0, log every 200 steps, initial +
final weights saved and reloaded for independent scoring. Literal settings are
recorded in the run JSON. Training ≈ minutes on CPU (reference: 88 s).

**Held-out scoring.** Validation token stream (stories with EOS between, same
encoding as training). Held-out targets are the common target stream tok[1:]
(first stream token is context only), shared by the n-gram, MLP and
transformer models so all models are scored on the same N − 1 targets.
Transformer windows: consecutive 64-target windows starting at 0, 64, 128, …
with inputs tok[start:start+64] and targets tok[start+1:start+65]; successive
65-token raw segments therefore overlap in exactly one boundary token (shared
as input only, never double-scored) and every target token is scored exactly
once. The final partial window contributes its remaining targets with the
available (cropped) context; no other tail is dropped. Each position
conditions only on its in-window prefix (no per-token independence assumption,
no EOS-stop, no exposure bias from variable-length stories). Context longer
than 64 is cropped to the last 64 tokens with consistent positions. bpc =
(total nats / ln 2) / total Unicode characters of the validation story texts —
identical normalization to Part 2.

Joint-normalization enumeration note: the initial distribution of the stream
(the bigram row conditioned on the EOS context) is explicitly normalized and
may be used as-is for the length-1 enumeration.

**Gates.**
- *Mandatory:* bpc below the best count model (4-gram, 1.310); reference
  1.207. Position-wise mean loss at ≥ 32 tokens of context below the trigram's
  2.09 nats (reference 1.65 over the last 8 positions — value descriptive).
  Total parameter count of the exact architecture (V·d + T·d + 2·block + final
  LN + readout, all linears biased) = **328832**; independent check recorded
  when the model is built.
- *Descriptive reference curve descriptors, not gates:* worse than the bigram
  at 1 token of context; passes the trigram after 3 tokens; keeps improving to
  64. Plotted with bigram/trigram lines and a two-standard-error band.
- *Causality negative control (mandatory, structural):* (a) every attention
  weight matrix has strictly zero weights above the diagonal (mask applied);
  (b) suffix invariance: changing tokens after position t must not change the
  logits at positions ≤ t. Asserting that unmasking a trained decoder must
  lower the loss is **not** part of the control (not guaranteed); any measured
  effect is reported as an observation.

**Samplers.** Temperature T (Eq. 25), top-k 20, top-p 0.9 at T = 0.8, min-p 0.1
at T = 0.7. Top-p keeps the smallest set of most likely tokens whose cumulative
probability reaches the threshold, **including the token that crosses it**; a
sampler that drops the crossing token is a known defect (p ≈ 0 expected).
Verification: one fixed broad context, 100,000 draws per sampler (seed 531505),
chi-squared against each sampler's exact kept-and-renormalized distribution,
rare bins pooled to expected ≥ 5, df = bins − 1; every p-value > 0.001
(reference 0.12–0.98). Temperature 1e-4 must return the argmax token every
draw. Story openings: 60 tokens, seed 1848, T = 1, same seed for every model
(uniform/unigram/bigram/trigram/transformer), EOS excluded from display.

**Self-consistency.** 128 sequences × 128 tokens sampled at T = 1 (seed
531506); per-sequence mean surprise minus the model's mean predicted entropy;
z-score computed **over the 128 sequences** (not over tokens): gate |z| ≤ 4
(reference 1.6). Repeat with sequences sampled at T = 0.7 scored by the T = 1
model: |z| must exceed 40 (reference −68 over 128 sequences).

Optional extensions (only after all required gates): attention-head maps; MLP
hidden-unit memory audit.

## Part 6 — Scaling laws

**DO.** Read `chinchilla-runs.csv` (245 runs: N, C, final loss L). D = C/(6N).
Drop the five runs of highest loss (240 remain). Fit L = E + A/N^α + B/D^β.

Literal fit specification (frozen before any formal fit):
- Parameters θ = (log E, log A, log B, α, β); positivity via log transform for
  E, A, B; α, β optimized directly with box bounds [0.05, 1.5].
- Residual r_i = ln L_pred(θ) − ln L_obs per run; objective
  F(θ) = (1/(δ² n)) Σᵢ huber(rᵢ, δ), δ = 1e-3, n = 240. The constant 1/δ²
  scaling does not change the minimizer but avoids false early convergence at
  the L-BFGS default ftol.
- Optimizer: scipy L-BFGS-B with analytic gradient; ftol = 1e-15,
  gtol = 1e-12, maxiter = 20000; convergence accepted only if the actual
  final gradient norm is consistent (no early-stop claims).
- Multistart grid (3⁵ = 243 starts, best kept): E ∈ {1.5, 1.7, 1.9},
  A ∈ {10, 100, 1000}, B ∈ {100, 1000, 10000}, α ∈ {0.25, 0.35, 0.45},
  β ∈ {0.25, 0.35, 0.45}.

Then 400 bootstrap resamples (240 runs with replacement, seeds 0..399), each
refit started from the best full fit; standard error = std of each parameter;
intervals for derived quantities are percentile bootstrap (no normal
assumption). Derive N*, D* analytically via G = (αA/βB)^{1/(α+β)} and verify
6 N* D* = C numerically. Report D*/N* at C = 5.76e23 with a 95 % bootstrap
interval.

**Gates.**
- α ∈ [0.32, 0.38], β ∈ [0.32, 0.42], E ∈ [1.75, 1.90]
  (reference 0.347 ± 0.014, 0.367 ± 0.021, 1.817 ± 0.025, A = 478, B = 2143).
- Own-fit mean log residual within ±0.005 (reference −0.0007); the printed
  parameters (E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28) must give a
  mean residual above 0.01 (reference +0.017).
- D*/N* at 5.76e23 within [10, 30] (reference 17.9, 95 % CI 7.7–37).
- The fit uses the published runs only; our TinyStories model is never fitted
  into the law; this is distinct from any historical blind-fit task.

## Delivery contract for the bounded first phase (preflight)

1. Scaffold: SPEC.md, PLAN.md, pyproject.toml + uv.lock, README skeleton,
   ignore rules, TASK_STATUS.md.
2. Pinned downloads with full provenance (INPUTS.md + evidence/input-manifest.json,
   data git-ignored).
3. Tests written first for input preparation (corpus split, tokenizer, Part 1
   module) derived from this SPEC; initial failures demonstrated and preserved
   in evidence/test-logs/; tests committed before implementation.
4. Implementation of input prep + Part 1 module; data/schema/tiny unit checks
   and Part 1 inference executed; gates checked and reported as measured
   (pass or fail, actual values).
5. TASK_STATUS.md updated; READY_FOR_PREFLIGHT.json binding source/input
   hashes, commands and results. **No MLP/Transformer training, no full memory
   sweeps, no formal n-gram/statistical/bootstrap runs in this phase.** Old
   week8 results untouched; a final README link at repo root is the only
   permitted later touch of existing files.

Failure policy: any unsupported glyph, OOV pair, schema error, hash mismatch,
or failed gate is reported with its actual measured value. Nothing is marked
passed without a number behind it.
