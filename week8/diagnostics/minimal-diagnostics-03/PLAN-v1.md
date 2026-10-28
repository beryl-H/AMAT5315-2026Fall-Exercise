# Minimal Diagnostics 03 — Pre-Registration Plan (Part 4, seed 0)

Date: 2026-10-29 · Author: pi (native WSL, volcengine-plan / glm-5-3-flash)
Status: **DESIGNED, NOT EXECUTED** — awaiting Codex review of this plan before
any diagnostic script runs. No training is performed or proposed. Only
inference and gradient measurement on already-saved checkpoints.

## 0. Scope and hard constraints (restated)

- Canonical repo: `week8/`; frozen `grok.py` sha256
  `d0501c3baf88a1194665123e47abc0a5b0035d0b203a56ca0dcc92853bc1d447`
  (asserted by every script at import; mismatch ⇒ abort).
- No training, no replay, no seed/split/init/hyperparameter/scientific-gate
  changes, no deletes, no commits, no remote actions.
- Only existing saved checkpoints are loaded. Weights are **never
  regenerated** (in particular, enabling x64 must never trigger
  `init_params`; arrays are loaded from npz and explicitly cast).
- Fixed comparison set: main seeds 0–4, same frozen split
  (`grok.fixed_split(0.30)`, RNG 897, 2823 train / 6586 test).
  `wd0-s0` is used only as a negative-mechanism control. No new seeds.
- No cross-seed tensor transplant. All interventions act on **transient
  in-memory copies** inside the diagnostic process; official artifacts are
  never written.
- Results go to a new exclusive namespace
  `week8/diagnostics/minimal-diagnostics-03/results/` (scripts assert it
  does not exist beforehand and create it; nothing outside it is written;
  no prior result or plan file is ever overwritten).
- Correlations/associations are not causal claims. No claim that seed 0
  would "eventually grok" or that init is "the cause" is permitted.

## 1. Why these two diagnostics (delta vs prior work)

Prior diagnostics (seed0-investigation-01, rev 2) validated the optimizer
formula and autodiff-vs-FD agreement **only at tested points**: diag02 used
synthetic init params (width 16, 64-pair subset) — it never touched the
saved memorization/final weights, the full 2823-pair batch, or float32-vs-
float64 on identical values. No prior diagnostic inspected the actual 3×3
masked attention maps of any checkpoint. The exact seed0 replay proved
determinism but says nothing about either hypothesis below. Both candidate
hypotheses therefore remain genuinely untested, and both are grounded in the
local 12-page sheet: the sheet's mechanism story is Fourier-concentrated
rotation computed through softmax attention (Part 4 UNDERSTAND), and the
actual model has learned positional embeddings + causal mask (erratum E1).

H1 is sharpened by one observed fact: at the final checkpoint seed 0 has the
**smallest** train loss of all main seeds (8.44e-7, vs 2.3e-4…1.5e-3 for
seeds 1–4), i.e. it memorized hardest, so float32 representation of very
small gradients is most stressed exactly there.

## 2. Snapshot / input set (both diagnostics)

| Snapshot | Files | sha256 (verified 2026-10-29) |
|---|---|---|
| main-s0 init/mem/final | `artifacts/part4/formal-decoder-02/main-s0/checkpoint-{init,memorization,final}.npz` | `c618d453…` / `3c6472e1…` / `2978381f…` |
| main-s1 init/mem/final | same pattern | `c5698c5b…` / `bfc4ffb0…` / `f3a67da5…` |
| main-s2 init/mem/final | same pattern | `756b055d…` / `e2bbdede…` / `4102e382…` |
| main-s3 init/mem/final | same pattern | `768aacf1…` / `573c4192…` / `af70b2e2…` |
| main-s4 init/mem/final | same pattern | `c9d0913b…` / `5856c322…` / `b180bcce…` |
| wd0-s0 final (control) | `…/wd0-s0/checkpoint-final.npz` | `7873cab1…` |

(All 64-hex values recorded in full in `PLAN.json` and re-asserted by the
scripts from these files at run time.) Full hash table also in
`READY.json`. Data: `grok.build_dataset()`, `grok.fixed_split(0.30)` —
frozen, no re-sampling, no subset selection anywhere.

## 3. Diagnostic 1 — H1 float32 vs float64 gradients on identical saved values

**Script**: `diag09_precision_gradients.py` → writes
`results/diag09_precision_gradients.json` only.

**H1 statement under test**: float32 forward/backward loses materially
small task gradients at/after memorization on the saved seed-0 weights,
in a way that seeds 1–4 (same float32 pipeline, grokked) do not show —
i.e. precision corruption would be a *seed-0-discriminating* candidate
mechanism, not a global benign rounding floor.

### 3.1 Procedure

1. Import JAX with x64 **disabled** (native float32 regime, as in training).
   Assert `grok.py` source hash; assert exclusive results namespace.
2. Load `checkpoint-{init,memorization,final}.npz` for main-s0..s4 via
   `grok.load_params` (arrays stay float32). Assert dtypes are float32.
   Never call `init_params` anywhere in the script.
3. Baseline tie-in (gate, pre-registered): for each snapshot, recompute
   full-batch train loss with `grok.loss_acc` (frozen, jit) in float32 and
   compare to the run's last logged `train_loss` at the matching step;
   require relative agreement ≤ 1e-5 (diag06 already showed exact
   reproduction; a mismatch aborts the diagnostic as a harness defect).
4. Float32 pass: `loss32` and per-leaf `g32 = jax.value_and_grad` of the
   full-batch **train** loss (the signal the optimizer actually consumed).
   Store as numpy float64 arrays (values unchanged; only container dtype).
5. Precision-context switch: `jax.config.update("jax_enable_x64", True)`
   **after** the float32 pass, verified by asserting a float64 constant op
   reports float64 and a retained float32 array still reports float32.
   Cast the *same loaded arrays* explicitly to float64
   (`x.astype(jnp.float64)` — exact, lossless). Recompute `loss64`, `g64`.
   If the runtime toggle fails, the script **aborts with a failure record**
   (no silent fallback, no re-planning inside the script).
6. Positive defect control (sensitivity of the metric, transient only):
   quantize a transient copy of one grokked seed's final params to bfloat16
   and recompute its gradient; the pipeline must flag it (relL2 > 0.01 vs
   the float64 reference on identical values). Confirms the metrics detect
   a genuine precision defect. The official model and files are untouched.
7. No writes except `results/diag09_precision_gradients.json`.

### 3.2 Exact metric definitions (computed in numpy float64; floor f = 1e-30)

Per snapshot, per leaf tensor, per block (embedding = {emb, pos};
attention = {wq,bq,wk,bk,wv,bv,wo,bo}; mlp = {mlp1.w,mlp1.b,mlp2.w,mlp2.b};
head = {head.w,head.b}), and for the whole flattened gradient:

- `abs_diff = ||g32 − g64||₂`
- `relL2 = ||g32 − g64||₂ / max(||g64||₂, f)` (robust near-zero denominator;
  `||g64||₂ < f` is additionally reported as `g64_degenerate: true`)
- `cosine = ⟨g32,g64⟩ / (||g32||₂ · max(||g64||₂, f))`; reported as `null`
  (counted separately) when `||g32||₂ < f` **or** `||g64||₂ < f` — never
  coerced to 0/−1
- `zero_loss_count` = #{i : g32[i] == 0 ∧ |g64[i]| > 0} (coordinates float32
  flushed to exactly zero)
- `spurious_count` = #{i : |g32[i]| > 0 ∧ g64[i] == 0}
- `nonfinite32`, `nonfinite64` counts
- Loss: `loss_abs_diff = L32 − L64`, `loss_rel = |L32−L64| / max(|L64|, f)`
- `x64_flag_at_pass` recorded for both passes; dtypes recorded.

### 3.3 Decision rule (engineering diagnostic, fixed now — NOT a course gate)

Thresholds justified a priori: float32 unit roundoff ≈ 1.19e-7; benign
accumulation over ~62.5k coordinates gives expected whole-vector relL2
≲ 1e-5, so a threshold 100× above that is conservative.

- **Material float32 gradient defect at snapshot X**: whole-gradient
  `relL2 > 1e-3` **or** any block `relL2 > 1e-3` **or** `cosine < 0.999`
  (when defined) **or** `zero_loss_count > 0` while `||g64||₂ ≥ f`.
- **H1 supported as discriminating (engineering level)**: a material defect
  at seed0 memorization and/or final, **and** no material defect at seed0
  init, **and** no material defect at any snapshot of seeds 1–4.
- **H1 rejected (at these snapshots)**: no material defect at seed0
  memorization/final, or defects present equally in seeds 1–4
  (non-discriminating global float32 behaviour — reported, not causal).
- **Positive control fails** ⇒ whole diagnostic recorded as invalid; values
  retained; no re-tuning of thresholds inside the session.

### 3.4 Controls, budget, conclusions

- Negative-ish anchor: init snapshots (large gradients) must show negligible
  relL2 — sanity that the comparison itself is tight.
- Positive control: §3.1 step 6 (bfloat16 quantization, transient).
- Budget: 16 snapshots × (forward+backward on 2823×3 tokens, 62.5k params)
  + 1 control ⇒ < 5 min CPU, < 1 GB RAM, single process, `JAX_PLATFORMS=cpu`.
- **Allowed conclusion**: whether float32 autodiff on the *saved* weights
  materially distorts loss/gradient values, and whether that distortion is
  seed-0-specific. **Not allowed**: any claim that float32 precision caused
  seed 0's non-grokking within 10000 steps (training dynamics not measured
  here), any extrapolation beyond step 10000, any model change.

## 4. Diagnostic 2 — H2 attention saturation / positional shortcut

**Script**: `diag10_attention_structure.py` → writes
`results/diag10_attention_structure.json` only.

**H2 statement under test**: seed 0's learned attention (memorization/final)
saturates or implements a positional/order-dependent shortcut that restricts
the content-based, symmetry-compatible computation the grokked seeds use.

### 4.1 Procedure

1. Frozen import; source-hash assert; exclusive namespace assert.
2. Snapshots: init/memorization/final for main-s0..s4, plus wd0-s0 final
   (negative-mechanism control). Inputs: **all 9409 pairs**
   (`grok.build_dataset()`), deterministic, predefined — no subset choice.
3. Baseline-match gate (pre-registered, before any intervention is
   trusted): with `grok.loss_acc` (frozen), recompute test accuracy for
   every snapshot; require exact agreement with the logged final/step
   accuracy to relative 1e-6 (diag06 precedent). Attention extraction uses
   the frozen `grok.attention_weights` only.
4. Descriptive attention metrics per snapshot/head (below). Rows 0 and 1 of
   the causal 3×3 map are near-degenerate (row 0 forced to [1,0,0]); the
   informative row is the answer row (index 2, attending to positions
   0,1,2); row 1 is recorded for completeness.
5. Interventions — one variable at a time, each on a **transient copy**,
   always reported next to its matched baseline:
   - I1 `pos→0`: zero the positional embeddings (3×64) only.
   - I2 `swap a,b`: evaluate on (b,a,'=') pairs only (inputs changed; c is
     symmetric so a rule-based model should be robust; an order-tied
     shortcut should not).
   - I3 `a→0`: replace the first operand with token 0 for all pairs
     (content-ablation descriptor).
   After each intervention, test loss/accuracy and the same attention
   metrics are recomputed. Harness-liveness control: I1 must change the
   logits on at least one pair for every snapshot (assert; bitwise identity
   ⇒ harness defect ⇒ abort with failure record).
6. No writes except `results/diag10_attention_structure.json`.

### 4.2 Exact metric definitions (per snapshot, per head, answer row)

Over the 9409 pairs, with w ∈ Δ³ the masked softmax row at position 2 and
s the pre-softmax masked scores (finite entries only):

- `entropy_mean`, `entropy_p10/p50/p90` of H = −Σ_j w_j ln w_j (nats)
- `sat_frac` = fraction of pairs with max_j w_j > 0.99
- `maxw_mean` = mean of max_j w_j
- `score_min`, `score_max`, `score_range` (finite masked logits)
- `input_dep_std` = max over j∈{0,1,2} of std over pairs of w_j
  (positional-only attention ⇒ ≈ 0; content-based ⇒ ≫ 0)
- Interventions: `Δacc = acc_int − acc_base`, `Δentropy_mean`,
  and for I2 additionally `acc_swap` on swapped pairs.

### 4.3 Decision rule (engineering diagnostic, fixed now — NOT a course gate)

Pre-registered, evaluated per metric against the span of seeds 1–4 at the
same checkpoint (no post-hoc threshold promotion):

- **Saturation signature (H2a)**: seed0 `entropy_mean` at final **below**
  every grokked seed's **and** `sat_frac > 0.5` while every grokked seed's
  `sat_frac < 0.1`.
- **Positional-shortcut signature (H2b)**: seed0 `input_dep_std < 1e-4` at
  final while every grokked seed's `> 1e-4`; **or** I1 `|Δacc| > 0.5` for
  seed0 while every grokked seed's `|Δacc| < 0.1`; **or** I2 `acc_swap <
  acc_base − 0.5` for seed0 while every grokked seed drops by < 0.1.
- **H2 supported at engineering level**: H2a or H2b holds at final (and is
  directionally present at memorization).
- **H2 falsified (at these snapshots)**: seed0's metrics and intervention
  deltas all lie within the [min, max] span of seeds 1–4.
- wd0-s0 final is reported as a descriptor (expected memorization-like
  pattern); it is not part of the pass/fail logic.
- Controls: baseline-match gate (§4.1 step 3) and harness-liveness (§4.1
  step 5). Either failing ⇒ diagnostic invalid, values retained.

### 4.4 Caveat recorded in the plan (binding on interpretation)

Attention entropy/saturation and intervention sensitivity are **conditional
associations measured at static snapshots**. Even a clean H2a/H2b signature
does not establish that this attention pattern *caused* seed 0's learning
failure — distinguishing mechanistic ablation from necessary cause would
require dynamics (out of scope, no training authorized). Entropy alone can
never prove cause.

### 4.5 Budget, conclusions

- Budget: 16 snapshots × (1 baseline + 3 interventions) × forward on
  9409×3 tokens ⇒ < 5 min CPU, < 1 GB RAM, single process, `JAX_PLATFORMS=cpu`.
- **Allowed conclusion**: whether seed 0's saved attention differs
  mechanistically (saturation / positional dependence / order sensitivity)
  from all grokked seeds at the same training stages. **Not allowed**: a
  causal claim that this pattern prevented grokking; any claim about steps
  beyond 10000; any claim of "eventual grok" or "init is the cause".

## 5. Precision-context management (both scripts)

- x64 is **off at import** (training-native float32 regime); all float32
  metrics are computed before any x64 toggle; the flag value at each phase
  is recorded in the output JSON.
- Weights are only ever loaded from npz and explicitly cast; `init_params`
  is never called (so the x64 random-stream difference is irrelevant).
- After enabling x64: assert a float64 op yields float64 and a retained
  float32 array is still float32. No global dtype casting of the reported
  float32 quantities.

## 6. Execution protocol

1. Codex reviews `PLAN.json` / `PLAN.md` (this pre-registration).
2. On authorization, run each script once from `week8/diagnostics/
   minimal-diagnostics-03/` with the repo `.venv`; stdout concise
   (one PASS/FAIL line each); full values in the results JSONs.
3. Scripts are single-shot: they assert the results namespace is new and
   refuse to overwrite any existing file, including their own outputs and
   any prior plan/READY file. Trial failures are retained as-is.
4. A short findings note (association-level language only) will be added
   after execution, in a new file — this plan is never edited post hoc.

## 7. What would and would not be evidence

| Outcome | Meaning |
|---|---|
| H1 material, seed-0-only defect | candidate mechanism elevated to "supported at engineering level"; still not a proven cause of non-grokking |
| H1 defect in all seeds equally | float32 rounding floor exists but is non-discriminating; reported, no causal role |
| H1 absent | float32-vs-float64 discrepancy eliminated at saved snapshots |
| H2a/H2b signature | mechanistic association supporting a restricted-attention description of seed 0's saved model |
| H2 within seeds-1–4 span | positional/saturation story eliminated at these snapshots |
| both falsified | neither candidate survives; root cause remains unresolved (honest outcome) |
