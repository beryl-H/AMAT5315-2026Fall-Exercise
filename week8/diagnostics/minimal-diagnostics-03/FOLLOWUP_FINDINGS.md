# Follow-up Findings — CE Cancellation Probe (final documentation turn)

Date: 2026-10-28 · Attribution: hypothesis/script authored by pi (native WSL,
volcengine-plan / glm-5-3-flash); preflight metric fixes and execution by
Codex (`followup-ce-cancellation/RELEASE_FOLLOWUP.json` — '='-position logit
selection, per-example float64 gate, validity checks; original draft retained
as `followup_ce_cancellation-pi-v2-before-preflight.py`). Single execution:
21.19 s, exit 0. **New training runs: 0 · optimizer updates: 0.**

## Result (all five predeclared fixed finals; no cherry-picking)

| checkpoint | orig-form f32 relL2 vs f64 | stable-form f32 relL2 | improvement | criterion |
|---|---|---|---|---|
| main-s0/final | 0.109105728 | 1.240794e-5 | **8793.216** | TRUE |
| main-s1/final | 2.67745e-4 | 4.52225e-6 | 59.2062 | TRUE |
| main-s2/final | 1.85893e-3 | 3.00407e-6 | 618.803 | TRUE |
| main-s3/final | 1.27417e-3 | 6.5205e-6 | 195.41 | TRUE |
| main-s4/final | 6.37731e-5 | 6.38484e-6 | **9.9882059967** | **FALSE (kept false; not rounded)** |

Harness: float64 form equivalence PASS (fixtures + per-example full batch);
accuracy unchanged; arrays immutable. Loss rel err at s0 final drops from
1.84e-2 to 7.68e-8. Independent Codex analytic audit
(`independent-cancellation-audit.json`): stable AD vs analytic target-
complement gradient relL2 8.97e-7; independent stable f32-vs-f64 1.240794e-5
— confirms the mechanism independently.

## Qualifications (binding)

- **Identified**: the float32-vs-float64 gradient discrepancy at the saved
  final snapshots is **locally** caused by cancellation in the standard
  log-softmax CE gradient (`p_true − 1` at near-zero loss). LOCAL numerical
  mechanism only — **NOT** a historical cause of seed 0's non-grokking.
- **H1** (diag09): strict predeclared verdict stands `defect_present_not_
  seed0_specific` — the distortion is not unique to seed 0.
- **H2** (diag10): memorization-stage temporal condition failed →
  **INCONCLUSIVE**, unchanged.
- **Historical root cause: UNRESOLVED.** Seed 0 scientific gates: unchanged
  failure. No budget extension, seed hunt, or further diagnostics proposed.

## Erratum (new record; prior FINDINGS.md left untouched)

- E1: FINDINGS.md H1 table says s0/init cosine "1 − 6e-13"; the exact value
  0.9999999999999367 is **1 − 6.3e-14**. Typo in the summary only; the
  underlying result JSON was always exact.
- E2: wording discipline — the 5-snapshot association (distortion grows with
  memorization depth) is a measured association, not a "robust" causal one.
