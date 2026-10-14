# Week 6 — Energy-based models, Part 1 SPEC (dataset understanding)

Scope: **Part 1 only** this turn. Parts 2–4 are planned but not implemented.
All work confined to `week6/energy-based/`. Parent diffusion work untouched.

## Goal (from the learning sheet)

Image galleries with digit/object labels and class-frequency plots per
training/test split that identify what one sample represents and how the
training and test empirical distributions compare.

## Requirements

1. **Environment**: `uv`, CPU default. Independent `pyproject.toml`/`uv.lock`/
   `.venv` inside `week6/energy-based/`. `matplotlib` PNG figures, `pytest`
   tests. Numeric stack (JAX) installed for later parts.
2. **Data (exact source files, no substitutes)**:
   - MNIST: 4 gzip IDX files from
     `https://storage.googleapis.com/cvdf-datasets/mnist`
     (`{train,t10k}-{images-idx3-ubyte,labels-idx1-ubyte}.gz`).
   - CIFAR-10 binary tarball from `https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz`.
   - Provenance (URL, effective URL, size, SHA-256, fetch date) recorded in
     `evidence/provenance.json` and README.
   - `inputs/`, `runs/`, `artifacts/`, `.venv/` are git-ignored.
3. **Loaders (safe, reusable, tested)**:
   - IDX images: validate magic 2051, counts, 28×28 dims; uint8 0..255.
   - IDX labels: magic 2049, count matches images; uint8 0..9.
   - CIFAR binary records: 3073 bytes = 1 label + 1024 R + 1024 G + 1024 B;
     all 5 train batches + test batch read; no unsafe tar extraction
     (iterate expected members, read bytes only).
   - Pairing and split identity preserved (original train/test assignment).
4. **Visualization (reusable)**:
   - Grayscale/RGB gallery with optional labels, native pixels
     (`interpolation="nearest"`), correct class names.
   - Deterministic gallery selection: **first 5 stored examples per class per
     split** (documented to avoid cherry-picking).
   - Class-frequency bar charts from **all** labels of each split.
5. **Verification (Part 1 VERIFY items)**:
   - MNIST: 60000 train / 10000 test, 28×28 grayscale, uint8 0..255.
     Digit 1 most common and 5 least common in **both** splits.
   - CIFAR-10: 50000 train / 10000 test, 32×32 RGB;
     exactly 5000 train / 1000 test per class (fraction 0.1).
   - Counts/fractions printed; summary saved to `evidence/part1_summary.json`.
   - Tests include negative controls: wrong RGB channel order caught,
     mismatched MNIST counts caught, invalid IDX magic caught,
     class-frequency normalization correct, split separation preserved.
6. **README**: commands paired with figures; explanatory text on empirical
   distribution vs unknown population, gallery appearance vs frequency,
   pixel representation, balanced gallery ≠ balanced dataset, source identity.

## Acceptance

- `uv run pytest -q` green (loaders + viz + summary checks).
- `uv run python scripts/make_evidence.py` regenerates figures + summary.
- Real-data verification printed and consistent with the sheet's numbers.
- Commits scoped to `week6/energy-based/` only, order: spec/plan → tests →
  implementation → evidence.

## Future parts (settings recorded, NOT implemented now)

Part 2 RBM/CD reference settings: 128 hidden units, 20 epochs, batch 100,
plain gradient descent lr 0.1, seed 2026, matched initialization and data
order for CD-1/CD-10. Parts 3–4: classical & modern Hopfield per sheet.

---

# Part 2 SPEC: RBM, CD learning, fresh generation, clamped completion

Goal (sheet): compare reconstructing an input digit, completing a partly
observed digit, and generating a new one; explain how training chain length
and sampling length affect the images.

## Model (exact definitions)

- Binarize original MNIST pixels at half intensity: `v_i = 1` if pixel >= 128
  (127 -> 0, 128 -> 1); spin `s_i = 2 v_i - 1`. 784 visible bits.
- Energy `E(v,h) = -a.v - b.h - v W h`; visible free energy
  `F(v) = -a.v - sum_j softplus(b_j + (v W)_j)` (eq. 4).
- Alternating Bernoulli block Gibbs from sigmoid conditionals (eq. 6-7).
- CD surrogate: `mean_batch[F(v0) - F(stop_gradient(vk))]` (eq. 10); chains
  start at each training batch (no persistent CD).

## Training settings (recorded before run)

128 hidden units; 20 epochs; batch 100; plain SGD lr 0.1; seed 2026; CD-1 and
CD-10 with EXACT shared initial parameters and per-epoch data permutations
(sha256 hashes of init arrays and each epoch permutation stored). Visible
bias initialized from logit of TRAIN pixel means clipped to
[0.001, 0.999] (documented choice: prevents +-inf logits for always-on/off
pixels; uses no held-out data); hidden bias 0; W ~ N(0, 0.01^2). Epoch loop
JIT-compiled with lax.scan (no per-minibatch Python overhead). Full
60000 images / 20 epochs; test split never used for fitting. Explicit
PRNG key management documented in code.

## Tests (before implementation)

Tiny 2-visible x 2-hidden RBM exactly enumerated in NumPy: energy, free
energy vs hidden marginalization; conditional probabilities; exact Gibbs
transition stationarity; CD autodiff gradient vs independent analytic
positive-minus-negative sufficient statistics with fixed endpoints;
stop-gradient negative control with a differentiable synthetic endpoint;
binarization boundary; clamped-pixel invariance across steps; Dice incl.
blank=0 and known values; shared init/perm hash equality checks.

## Evidence (VERIFY 1-4)

1. First 16 binarized train images + enlarged spin image + schematic network
   diagram (784 visible / 128 hidden, cross-layer couplings only).
2. Paired CD-surrogate / held-out one-step reconstruction (Hamming fraction,
   10000 test images) per epoch INCLUDING initialization; runtimes, settings,
   hashes. Acceptance: final recon < 0.5 x initial for BOTH models.
3. Fresh generation: 16 chains, matched random Bernoulli(0.5) starts for both
   models; same evolving chains saved at steps 1,10,100,1000; first-16
   held-out digits and their one-step reconstructions shown separately.
4. Completion: first 64 held-out digits, 8 chains each, lower 14 rows
   missing, upper 14 clamped every step; matched random inits across models
   and true/wrong-upper conditions; wrong upper = first cyclic
   different-label held-out digit within the 64 (mapping fixed before any
   outcome, indices recorded). Ink Dice on missing half
   (2|A cap B|/(|A|+|B|), blank->0 including both-blank); average 8 chains
   per digit, then mean +- SEM (ddof=1) over 64 digits. Acceptance at step
   1000: true-minus-wrong >= 0.1 for BOTH models. Fixed first-4-digit
   completion galleries + masked/true images.

Metrics saved as compact tracked JSON + PNGs; parameters/keys/hashes in
git-ignored artifacts/<run>/. No AIS, no NLL claims. Surrogate need not
decrease; only the Hamming reconstruction criterion is asserted.

---

# Part 3 SPEC: classical Hopfield network capacity

Goal (sheet): Hebbian couplings make stored patterns stable states of an
Ising energy; the unstable-bit fraction follows the crosstalk prediction;
recall collapses near alpha_c ~= 0.138.

## Model (exact definitions)

- Patterns xi^mu_i = +/-1 iid uniform, P patterns of N spins.
- Hebbian couplings K = Xi^T Xi with exact integer arithmetic (cast to
  int32 BEFORE the matmul to avoid int8 overflow), diagonal set to 0;
  J = K / N (normalization used only for energy/overlap reporting).
- Energy E(s) = -0.5 s J s; overlap m_mu = xi_mu . s / N; identity
  E(s) = -N/2 sum_mu m_mu^2 + P/2.
- Async zero-temperature updates s_i <- sign(h_i) with h_i = sum_j J_ij s_j;
  a ZERO local field leaves the spin unchanged; random update order per
  chain and per sweep (independent derived keys); sweeps repeat until one
  full sweep flips nothing (safety cap 200 sweeps -> flagged incomplete).
  Integer fields for sign/tie decisions avoid float32 fictitious ties.

## Study (fixed before outcomes)

N in {100, 200, 400, 800}; alpha in 0.05..0.25 inclusive step 0.01;
P = round(alpha*N); 20 independent pattern sets per point; seed 2026 fixed
before any outcome; derived-key formula documented in the driver. Each set
yields BOTH the unstable-bit fraction (xi_i h_i < 0 over ALL stored
patterns, zeros not counted) and recall of the first min(10, P) stored
patterns STARTING AT THE EXACT stored pattern, from the same patterns.
Per-set aggregation first, then SEM over the 20 sets (no pseudoreplication).

## Tests (before implementation, calling production functions)

Symmetric zero-diagonal Hebb; energy-overlap identity; strictly decreasing
energy per flip (tie holds); incremental fields match recomputed K s after
flips; converged states have all-local-fields stable; seeded permutation
reproducibility; NumPy reference agrees on deterministic update order;
negative controls (wrong-pattern couplings -> fraction near 0.5; non-zero
diagonal -> suppressed fraction); single-pattern single-bit corruption at
N=4 retrieved deterministically.

## Acceptance

- Unstable fractions with SEM vs Phi(-1/sqrt(alpha)); finite-N
  Phi(-sqrt((N-1)/(P-1))) curves added. N=800 alpha=.20 near 1.27e-2;
  N=100 alpha=.10 below the large-N curve as finite-N predicts.
- Mean final overlap and success fraction (m>=0.9) vs alpha per N, alpha_c
  0.138 marked. First .9-crossing by linear interpolation between adjacent
  loads (method reported); crossings decrease with N and within 0.02 of
  .186/.170/.156/.150 (N=100..800). Min mean overlap over alpha<=0.12
  reported (>0.988 expected).
- Raw per-set arrays in ignored artifacts; compact tracked JSON + PNGs.
