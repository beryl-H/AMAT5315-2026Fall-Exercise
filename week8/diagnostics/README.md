# Part 4 diagnostic evidence

These records supplement the original `formal-decoder-02` experiment. They do
not replace any failed seed or relax any worksheet criterion. The formal result
remains **4/5 main seeds passing all gates; main seed 0 did not grok by 10000
steps**. Part 3's blind scaling prediction also remains a miss.

## Current evidence and interpretation

- [Worksheet conformance](seed0-investigation-01/WORKSHEET_CONFORMANCE.md):
  explicit setup requirements versus measured scientific gates.
- [Initial investigation](seed0-investigation-01/root-cause-report.md):
  optimizer, gradient, split and checkpoint checks, with unresolved causes.
- [Exact replay verification](seed0-investigation-01/independent-replay-comparison.json):
  one previously declared replay reproduced all 101 numeric log points and all
  three checkpoint arrays bitwise; the original 147 run artifacts were unchanged.
- [Minimal diagnostic plan](minimal-diagnostics-03/PLAN.md) and
  [execution release](minimal-diagnostics-03/RELEASE.json): fixed tests of
  precision and attention on saved checkpoints, with zero optimizer updates.
- [Primary findings](minimal-diagnostics-03/FINDINGS.md): precision differences
  were not seed-0-exclusive; the attention hypothesis's predeclared temporal
  condition failed, so its result is inconclusive.
- [CE follow-up plan](minimal-diagnostics-03/followup-ce-cancellation/FOLLOWUP_PLAN.json),
  [follow-up findings](minimal-diagnostics-03/FOLLOWUP_FINDINGS.md) and
  [final interpretation](minimal-diagnostics-03/FINAL_INTERPRETATION.json):
  an equivalent stable cross-entropy reduced seed 0's saved-final-checkpoint
  float32 gradient relative error from about 0.1091 to 1.2408e-5 (8793x).
  This identifies a **local numerical cancellation mechanism**, not the
  historical cause of non-grokking. Seed 4's improvement was 9.988x and remains
  below the predeclared 10x diagnostic criterion.
- [Independent analytic-gradient audit](minimal-diagnostics-03/audit/independent-cancellation-audit.json)
  and [integrity receipt](minimal-diagnostics-03/audit/final-integrity.json).

## Reading and execution boundaries

The active scripts are `diag09_precision_gradients.py`,
`diag10_attention_structure.py`, and
`followup-ce-cancellation/followup_ce_cancellation.py`. `RELEASE*.json` records
are authoritative for executed versions; earlier `READY*.json` files describe
Pi drafts before Codex preflight corrections. Files named `*-before-preflight*`,
`*-draft-superseded*`, and `PLAN-v1.*` are retained audit history, **not active
implementations**. Earlier diagnostic failures and revisions are recorded in
[initial-check-history.json](seed0-investigation-01/initial-check-history.json).

Saved diagnostic outputs are inference/backward measurements, not new training
runs. They preserve the original seeds, split, initialization, hyperparameters,
10000-step cap and scientific gates. Static snapshots cannot establish what
caused the prior optimization trajectory. No model correction or new formal
training is claimed.

The worksheet keeps corpus and checkpoints out of Git; those remain in local
`week8/data/` and `week8/artifacts/`. Raw Pi sessions/authentication material and
the discontinued external-reference lookup are not published. Some original
audit helper commands retain canonical WSL/support paths; they are execution
records, not standalone portable commands. Diagnostic scripts refuse to
overwrite existing result files.

Pi in native WSL (`volcengine-plan / glm-5-3-flash`) authored diagnostic designs
and implementations. Codex performed preflight corrections, dispatch and
independent checks, as recorded in the release receipts. The original formal
model/protocol and failure artifacts remain unchanged.
