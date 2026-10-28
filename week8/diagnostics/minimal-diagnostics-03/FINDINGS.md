# Findings — Minimal Diagnostics 03 (Part 4, seed 0)

Date: 2026-10-28 · Author: pi (native WSL, volcengine-plan / glm-5-3-flash)
Execution: Codex ran the two frozen diagnostics (preflight metric-implementation
fixes recorded in `RELEASE.json` **before any outcome existed**); both exited 0
(13.4 s / 16.4 s), all controls passed, no training or optimizer updates, the
147 original artifacts remain immutable. Plans/scripts/results are unchanged.
Machine-readable version: `FINDINGS.json`.

## H1 — float32 vs float64 gradients on identical saved values

**Verdict (pre-declared engineering rule): `defect_present_not_seed0_specific`.**

| snapshot | train loss (logged) | whole relL2 | whole cosine | loss rel err |
|---|---|---|---|---|
| main-s0/init | 4.595 | 3.51e-7 | 1 − 6e-13 | 6.1e-8 |
| main-s0/memorization | 0.0295 | 1.89e-5 | 1 − 1.7e-10 | 8.8e-7 |
| **main-s0/final** | **8.44e-7** | **0.109106** | **0.994030** | **1.84e-2** |
| main-s1/final | 2.27e-4 | 2.68e-4 | 1 − 3.4e-8 | 2.8e-5 |
| main-s2/final | 4.07e-5 | 1.86e-3 | 1 − 1.6e-6 | 4.4e-4 |
| main-s3/final | 8.19e-5 | 1.27e-3 | 1 − 7.0e-7 | 1.6e-4 |
| main-s4/final | 1.47e-3 | 6.42e-5 | 1 − 1.9e-9 | 3.4e-6 |

- main-s0/final blocks: embedding 0.0697, attention 0.0723, MLP 0.1821,
  head 0.1445. No meaningful zeros anywhere (gauge/roundoff-safe rule).
- Seeds 2 and 3 also cross the pre-declared material threshold (1e-3), so the
  strict verdict is **not** seed-0-specific — reported exactly as
  pre-registered, without reinterpretation.
- Robust association: distortion at final scales with memorization depth
  (s0 has the smallest train loss and ~59× the next-largest relL2); init and
  memorization snapshots are negligible everywhere (≤ 2.3e-5).
- Metric-detector validation PASS; x64 flag discipline recorded (all float32
  passes before the single toggle).
- **Not supported**: that float32 precision caused seed 0's non-grokking;
  elimination of float32 training-dynamics effects in general; anything
  about steps > 10000; any model change.

## H2 — attention saturation / order sensitivity (narrowed)

**Verdict (pre-declared rule): `inconclusive_outside_envelope`.**

| snapshot (main) | head-mean entropy | head-mean sat frac | symmetry disagreement (|S|=4510) |
|---|---|---|---|
| s0 init | 1.0985 | 0.0000 | 0.0177 |
| s0 memorization | 0.1603 | 0.5723 | 0.9011 |
| **s0 final** | **0.1817** | **0.5034** | **0.7938** |
| s1 memorization / final | 0.1469 / 0.5237 | 0.5958 / 0.0003 | 0.9361 / 0.0000 |
| s2 memorization / final | 0.1702 / 0.4903 | 0.5399 / 0.0036 | 0.8479 / 0.0000 |
| s3 memorization / final | 0.1644 / 0.4947 | 0.5635 / 0.0036 | 0.9100 / 0.0000 |
| s4 memorization / final | 0.1698 / 0.4623 | 0.5415 / 0.0093 | 0.8075 / 0.0000 |
| wd0-s0 final (descriptor) | 0.1448 | 0.6065 | 0.8989 |

- At **final**, seed 0 is far outside the grokked envelope: entropy 0.182 vs
  0.462–0.524; sat 0.503 vs 0.0003–0.0093; disagreement 0.794 on the 4510
  both-heldout ordered pairs vs 0.000 for every grokked seed.
- At **memorization**, seed 0 is *not* more extreme than all grokked seeds
  (s1 is higher on sat and disagreement, lower on entropy), and all seeds
  look alike at that stage. The pre-declared support rule required the final
  signature to be directionally present at memorization → **INCONCLUSIVE**.
  Thresholds were not rewritten to convert this into a pass or a fail.
- wd0-s0 final resembles the same memorization-like pattern — a descriptor
  only, consistent with "not yet generalized", never evidence of cause.
- **Not supported**: any causal claim that seed 0's final attention pattern
  prevented grokking; entropy/position-sensitivity as proof of a shortcut;
  any historical-learning-cause claim from static snapshots.

## Interpretation discipline

- **Final-snapshot associations** (reported above): float32 gradient
  distortion grows with memorization depth; seed 0's final attention is
  saturated and order-asymmetric relative to grokked seeds. Both are
  measurements on saved weights at specific steps.
- **Historical causation**: UNRESOLVED. Nothing here establishes why seed 0
  did not grok within the mandated 10000-step budget; whether it would grok
  beyond the budget remains unknown and untested. The scientific gates and
  all course criteria are unchanged.

## Follow-up (declared separately — NOT part of the pre-registration above)

New post-hoc hypothesis, proposed **after** these outcomes and timestamped in
`followup-ce-cancellation/FOLLOWUP_PLAN.json`: cancellation in the standard
float32 log-softmax cross-entropy gradient (`p_true − 1` at nearly zero
training loss). One focused probe on the same saved values
(`followup_ce_cancellation.py`, not executed; awaiting Codex preflight).
Scope is explicitly pointwise numeric mechanism — it cannot retroactively
describe or fix past grokking.
