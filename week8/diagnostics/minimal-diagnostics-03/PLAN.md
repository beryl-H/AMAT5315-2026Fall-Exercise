# Minimal Diagnostics 03 — Pre-Registration Plan v2 (Part 4, seed 0)

Date: 2026-10-28 · Author: pi (native WSL, volcengine-plan / glm-5-3-flash)
Status: **DESIGNED, NOT EXECUTED** — revised before execution per Codex design
review (`review-before-execution.md`). v1 is preserved unchanged as
`PLAN-v1.md` / `PLAN-v1.json` (and the superseded draft script as
`diag09_precision_gradients-v1-draft-superseded.py`); revisions below are
pre-execution design changes, not responses to outcomes. No experiment has
run. No training is performed or proposed. Only inference and gradient
measurement on already-saved checkpoints.

## 0. Scope and hard constraints (restated)

- Canonical repo: `week8/`; frozen `grok.py` sha256
  `d0501c3baf88a1194665123e47abc0a5b0035d0b203a56ca0dcc92853bc1d447`
  (asserted by every script at import; mismatch ⇒ abort).
- No training, no replay, no seed/split/init/hyperparameter/scientific-gate
  changes, no deletes, no commits, no remote actions. All course gates
  unchanged; the decision rules below are engineering diagnostics only.
- Only existing saved checkpoints are loaded. Weights are **never
  regenerated** (`init_params` is never called; x64 never triggers a
  re-init). Arrays are loaded from npz and explicitly cast.
- Fixed comparison set: main seeds 0–4, same frozen split
  (`grok.fixed_split(0.30)`, RNG 897, 2823 train / 6586 test).
  `wd0-s0` is used only as a negative-mechanism descriptor. No new seeds.
- No cross-seed tensor transplant. All interventions act on **transient
  in-memory copies** inside the diagnostic process; official artifacts are
  never written.
- Results namespace: each script writes **only inside its own exclusive
  subdirectory** `results/diag09/` and `results/diag10/` (created if absent;
  the script's own result file asserted not to exist via `os.O_EXCL`-style
  check; common parent `results/` allowed). No prior result or plan file is
  ever overwritten; harness failures are retained (see §6).
- Associations are not causal claims. No claim that seed 0 would
  "eventually grok" and no "init is the cause" claim is permitted. H1/H2
  outcomes also never *eliminate* float32 training-dynamics effects in
  general — they speak only to the saved snapshots measured here.

## 1. Why these two diagnostics (delta vs prior work — corrected narrative)

Prior diagnostics (seed0-investigation-01, rev 2) validated the optimizer
formula and autodiff-vs-FD agreement **only at tested points**. Correction
of the v1 narrative: diag02's unused `WIDTH = 16` local variable had no
effect — `grok.init_params` uses the frozen `WIDTH = 64`, so diag02 probed
**actual width-64 init params** (not synthetic width-16 ones) on a 64-pair
subset, with float64 FD. It never touched the saved memorization/final
weights, the full 2823-pair batch, or float32-vs-float64 on identical
values. No prior diagnostic inspected the actual 3×3 masked attention maps
of any checkpoint. The exact seed0 replay proved determinism but says
nothing about either hypothesis. Both hypotheses remain genuinely untested
and are grounded in the local 12-page sheet: the sheet's mechanism story is
Fourier-concentrated rotation computed through softmax attention (Part 4
UNDERSTAND), and the actual model has learned positional embeddings +
causal mask (erratum E1).

H1 is sharpened by one observed fact: at the final checkpoint seed 0 has
the **smallest** train loss of all main seeds (8.44e-7, vs 2.3e-4…1.5e-3
for seeds 1–4), i.e. it memorized hardest, so float32 representation of
very small gradients is most stressed exactly there.

## 2. Snapshot / input set (both diagnostics)

Identical to v1 (verified hash table in `PLAN.json`): `init/memorization/
final` for main-s0..s4 plus `wd0-s0/checkpoint-final.npz` (descriptor).
Data: `grok.build_dataset()`, `grok.fixed_split(0.30)` — frozen, no
re-sampling, no subset selection anywhere.

## 3. Diagnostic 1 — H1 float32 vs float64 gradients on identical saved values

**Script**: `diag09_precision_gradients.py` → writes only
`results/diag09/diag09_precision_gradients.json`.

**H1 statement under test**: float32 forward/backward loses materially
small task gradients at/after memorization on the saved seed-0 weights, in
a way seeds 1–4 (same float32 pipeline, grokked) do not show — i.e.
precision corruption as a *seed-0-discriminating* candidate mechanism.

### 3.1 Procedure (order fixed)

1. Import JAX with x64 **disabled**; assert `grok.py` source hash; assert
   own result file absent.
2. Load `checkpoint-{init,memorization,final}.npz` for main-s0..s4 via
   `grok.load_params`; assert dtypes float32. **Pass 1 (ALL float32 passes
   happen here, before any x64 toggle)**: full-batch train loss `L32` and
   per-leaf `g32 = jax.value_and_grad` of the frozen `grok.loss_acc` train
   loss (the signal the optimizer actually consumed); leaves stored in
   numpy float64 containers (values unchanged). The loaded float32 param
   trees stay in memory and are reused in pass 2 — the same arrays, no
   reload, no re-init (asserted: dtype float32 immediately before casting).
3. Baseline tie-in per snapshot against the **exact corresponding log
   row**: init ↔ step-0 row, memorization ↔ row at
   `completion-receipt.json`'s `memorization_step`, final ↔ last row
   (`steps_run`). Require relative agreement ≤ 1e-5 (diag06 precedent:
   exact reproduction). Mismatch ⇒ harness defect ⇒ abort (retained).
4. Precision-context switch **after all float32 passes**:
   `jax.config.update("jax_enable_x64", True)`; assert a float64 constant
   op reports float64 and a retained float32 array still float32; record
   both flags. **Pass 2 (ALL float64 passes)**: cast the same arrays
   explicitly (`x.astype(jnp.float64)`, exact/lossless); recompute `L64`,
   `g64`. If the runtime toggle fails ⇒ abort with failure record (no
   silent fallback).
5. **Metric-detector validation (deterministic; validates the metric
   implementation, NOT the float32 backend)** on one snapshot's `g64`:
   - (a) self-comparison: relL2 must be exactly 0, cosine 1;
   - (b) injected 10% scaling `g64 × 1.1`: relL2 must equal 0.1 (up to
     float64 roundoff) and must be flagged material (> 1e-3);
   - (c) injected meaningful zero: zero the single largest-|g64|
     coordinate of a non-degenerate block; the meaningful-zero count must
     increment and the block must flag.
   (a)–(c) failing ⇒ diagnostic invalid, values retained. No claim that
   this validates the whole precision backend. Conclusions are qualified:
   a matched float64 computation may itself differ by backend paths.
6. Writes only `results/diag09/diag09_precision_gradients.json`.

### 3.2 Exact metric definitions (numpy float64; decision-level floor)

- Raw per-leaf metrics (reported for every leaf, including potentially
  gauge-sensitive leaves such as the key bias `bk`):
  `abs_diff = ||g32−g64||₂`, `norm64 = ||g64||₂`, `relL2 = abs_diff /
  max(norm64, 1e-30)`, `cosine` (null when `norm32 < 1e-30` or
  `norm64 < 1e-30`; never coerced), `nonfinite32/64`.
- **Decision basis**: only **whole-vector and block vectors**
  (embedding {emb,pos}; attention {wq,bq,wk,bk,wv,bv,wo,bo};
  mlp {mlp1.w,mlp1.b,mlp2.w,mlp2.b}; head {head.w,head.b}) with
  `norm64 ≥ 1e-12`. Blocks/whole below that are reported as
  `degenerate: true` and **excluded from material classification** (their
  raw metrics are still reported).
- **Meaningful zero** (never inferred from roundoff residuals): coordinate
  counted only when `g32[i] == 0` **and** `|g64[i]| > max(1e-12,
  1e-6·||g64_whole||₂)`. A `g32 == 0` against merely tiny `g64` is **never
  classified as material**.
- Loss: `loss_abs_diff = L32 − L64`, `loss_rel = |L32−L64| / max(|L64|,
  1e-30)`.
- `x64_flag` and dtypes recorded for both passes.

### 3.3 Decision rule (engineering diagnostic, fixed now — NOT a course gate)

Thresholds justified a priori: float32 unit roundoff ≈ 1.19e-7; benign
accumulation over ~62.5k coordinates ⇒ expected whole-vector relL2 ≲ 1e-5;
threshold 100× above.

- **Material defect at a snapshot**: on the whole vector or any
  non-degenerate block (`norm64 ≥ 1e-12`): `relL2 > 1e-3` **or**
  `cosine < 0.999` (when defined) **or** meaningful-zero count > 0 **or**
  `nonfinite32 > 0`.
- **Verdicts (three-way, honest):**
  - `supported_seed0_specific_engineering`: material defect at seed0
    memorization and/or final, none at seed0 init, none at any seeds 1–4
    snapshot.
  - `defect_present_not_seed0_specific`: material defects also occur in
    seeds 1–4 and/or seed0 init ⇒ **no seed0-specific support**; this is
    explicitly **not** an elimination of float32 training-dynamics effects.
  - `no_material_defect_at_snapshots`: no material defect anywhere
    measured ⇒ H1 rejected **at these snapshots**; also **not** an
    elimination of float32 training-dynamics effects.
- No categorical causal claims in any branch.
- Positive association with the failure is only claimed in the first
  branch, and even there it is a snapshot-level association, not a
  demonstrated cause of the non-grokking (dynamics not measured).

### 3.4 Controls, budget, conclusions

- Negative anchor: init snapshots (large gradients) must show negligible
  relL2 — sanity that the comparison itself is tight.
- Positive control: §3.1 step 5 (deterministic metric-detector validation;
  replaces v1's bfloat16 param quantization, which changed both weights
  and numerics unpredictably). Official model and files untouched; no new
  forward copies needed for the control.
- Budget: 16 snapshots × 2 passes × (forward+backward on 2823×3 tokens,
  62.5k params) ⇒ < 5 min CPU, < 1 GB RAM, single process,
  `JAX_PLATFORMS=cpu`.
- **Allowed conclusion**: whether float32 autodiff on the *saved* weights
  materially distorts loss/gradient values, and whether that distortion is
  seed-0-specific. **Not allowed**: that float32 precision caused seed 0's
  non-grokking within 10000 steps; anything about steps > 10000; any
  model change; any claim that the absence of a defect eliminates
  precision effects during training.

## 4. Diagnostic 2 — H2 attention saturation / order sensitivity (narrowed)

**Script**: `diag10_attention_structure.py` → writes only
`results/diag10/diag10_attention_structure.json`.

**H2 statement under test (narrowed per review)**: seed 0's learned
attention at memorization/final **saturates** and/or is **order-sensitive**
on symmetric unseen pairs, relative to all grokked seeds. Positional-only
input dependence is recorded descriptively but is not a positive criterion
(position sensitivity alone is not proof of a shortcut, and pos→0 affects
all models' input geometry).

### 4.1 Procedure

1. Frozen import; source-hash assert; own result file absent.
2. Snapshots: init/memorization/final for main-s0..s4 + wd0-s0 final
   (descriptor). Inputs: **all 9409 pairs**, deterministic, predefined.
3. **Diagnostic attention forward**: the frozen helper
   `grok.attention_weights` returns only the first batch sample
   (`[0][None]`), so a small diagnostic forward (same frozen ops, float32,
   no x64) returns logits **and** per-sample attention maps for all B
   inputs. **Baseline-match gate before any metric or intervention is
   trusted**: on the fixed test split, custom logits vs frozen
   `grok.forward` logits must satisfy `|Δ| ≤ 1e-5 + 1e-6·|ref|`
   elementwise (f32-appropriate, explicit), and the recomputed test
   accuracy from the custom forward must match the **exact corresponding
   log row** (init ↔ step 0; memorization ↔ `memorization_step` row;
   final ↔ last row) to relative 1e-6. Failure ⇒ harness defect ⇒ abort
   (retained).
4. Attention metrics per snapshot/head, computed safely (`0·log0 → 0`):
   answer row (row 2; positions 0,1,2 all causal-visible) with w the
   masked softmax weights:
   - per head, per pair: `H = −Σ_{j: w_j>0} w_j ln w_j` (nats);
   - per head: `entropy_mean/p10/p50/p90` over pairs; `sat_frac` =
     fraction of pairs with `max_j w_j > 0.99`; `maxw_mean`;
   - **aggregates (exactly defined)**: `head_mean_entropy` = mean over the
     4 heads of per-head pair-means; `head_mean_sat_frac` = mean over the
     4 heads of per-head `sat_frac`;
   - descriptive: `input_dep_std` (max over positions of the std of w over
     pairs), row-1 entropy mean (row 0 is mask-forced to [1,0,0]),
     masked pre-softmax score min/max/range (finite entries).
5. **Order-sensitivity test (replaces v1's invalid whole-set swap)**:
   fixed set `S = {(a,b) : a ≠ b, test_mask[a,b] ∧ test_mask[b,a]}`
   (both orientations unseen; deterministic from the frozen split).
   `|S|` and the sha256 of the sorted flat indices of `S` are reported.
   Metric: **paired prediction symmetry disagreement** = fraction of
   `(a,b) ∈ S` with `argmax logits(a,b,'=') ≠ argmax logits(b,a,'=')`
   (targets are identical under the swap). Also reported: mean absolute
   logit difference over S. No aggregate-accuracy-swap is computed
   (invariant and leak-prone — removed).
6. **Interventions**: single-variable, transient copies only, each beside
   its matched baseline:
   - I1 `pos→0`: zero positional embeddings — **descriptive only** (no
     Δacc threshold criterion; affects all models' input geometry).
   - Harness-liveness control: I1 must change logits on at least one pair
     for every snapshot; bitwise identity ⇒ harness defect ⇒ abort
     (retained).
   - v1's I2 whole-set swap and I3 `a→0` are **removed** (I2 was
     invariant/leaky; I3 changes the true target and is a destructive
     input ablation, not task accuracy). Score softening (β=0.5) is
     **omitted for minimality**; noted here so its absence is deliberate.
7. Writes only `results/diag10/diag10_attention_structure.json`.

### 4.2 Decision rule (engineering diagnostic, fixed now — NOT a course gate)

Pre-registered; thresholds fixed before execution; evaluated at **final**,
with memorization required to be directionally consistent for support:

- **H2a saturation signature**: seed0 `head_mean_entropy` at final strictly
  below every grokked seed's **and** seed0 `head_mean_sat_frac > 0.5`
  while every grokked seed's `< 0.1`.
- **H2c order-sensitivity signature**: seed0 paired symmetry disagreement
  on S `> 0.5` while every grokked seed's `< 0.05` (formal-diagnostic
  positive-association criterion; predeclared, **not causal**).
- **Supported (engineering level)**: H2a or H2c holds at final **and** the
  same metric moves in the same direction at memorization.
- **Falsified (within envelope)**: every seed0 metric (entropies, sat
  fractions, disagreement, input-dependence) lies within the [min, max]
  span of seeds 1–4 at the same checkpoints.
- **INCONCLUSIVE (outside envelope)**: no signature fires but at least one
  seed0 metric lies outside the seeds-1–4 envelope. Reported honestly as
  inconclusive — never forced into a binary.
- wd0-s0 final is a descriptor only, never in pass/fail logic.
- **Causality caveat (binding)**: static-snapshot attention metrics and
  order-disagreement are conditional associations; they cannot establish a
  necessary cause of past learning failure; entropy alone never proves
  cause; position sensitivity alone is not proof of a shortcut.

### 4.3 Budget, conclusions

- Budget: 16 snapshots × (baseline forward on 9409×3 + one I1 forward)
  ⇒ < 5 min CPU, < 1 GB RAM, single process, `JAX_PLATFORMS=cpu`.
- **Allowed conclusion**: whether seed 0's saved attention differs
  mechanistically (saturation / order sensitivity) from all grokked seeds
  at the same stages, or is inconclusive. **Not allowed**: causal claims
  about why seed 0 did not grok; claims about steps > 10000;
  "eventual grok" or "init is the cause" claims.

## 5. Precision-context management (both scripts)

- x64 **off at import** (training-native float32); **all** float32 passes
  complete before the single x64 toggle (no per-snapshot toggling that
  would contaminate later float32 passes); both flag values recorded.
- Weights only ever loaded from npz and explicitly cast; `init_params`
  never called (the x64 random-stream difference is therefore irrelevant).
- After enabling x64: float64-op and float32-retention dtype assertions.

## 6. Execution protocol

1. Root executes checks; on authorization each script runs **once** from
   `week8/diagnostics/minimal-diagnostics-03/` with the repo `.venv`.
2. Single-shot: each script asserts its own result file is new (exclusive
   subdir, `O_EXCL` semantics) and refuses to overwrite anything, including
   its own output and any prior plan/READY file.
3. **Harness failures are retained**: a failing attempt is kept as
   `attempt01` (result/log) and any fix lands in a **new** attempt file
   only after re-review; metrics are never changed after outcomes exist.
4. A findings note (association-level language only) will be added after
   execution as a new file; this plan is never edited post hoc.

## 7. What would and would not be evidence

| Outcome | Meaning |
|---|---|
| H1 `supported_seed0_specific` | candidate mechanism elevated to engineering-level association; still not a proven cause |
| H1 `defect_present_not_seed0_specific` | precision distortion exists at snapshots but is not seed-0-discriminating; no support for H1 as discriminator; dynamics effects not excluded |
| H1 `no_material_defect_at_snapshots` | float32-vs-float64 discrepancy eliminated **at these snapshots**; training-dynamics precision effects not excluded |
| H2a/H2c signature | mechanistic association supporting saturation/order-sensitivity description of seed 0's saved model |
| H2 within seeds-1–4 envelope | saturation/order-sensitivity story eliminated at these snapshots |
| H2 INCONCLUSIVE | metrics outside envelope but no pre-declared signature; honest non-binary outcome |
| harness failure | diagnostic invalid; retained; fix only in a new reviewed attempt |
