# Part 4 Seed 0 Root-Cause Report (investigation seed0-investigation-01, rev 2)

Date: 2026-10-28 · Investigator: pi (native WSL, volcengine-plan / glm-5-3-flash)
Rev 2: incorporates Codex independent-review corrections (causal language
removed, initial-check history preserved, robust negative control diag08
added, erratum for the stale `grok.py` docstring, validation aligned to the
predeclared exact replay).

Scope: diagnostic only. No training launched by Pi, no frozen-source
modification, no deletes, no seed/split/threshold/step-cap/init changes.
The only training activity in scope is Codex's single predeclared exact
seed-0 replay (`week8-support/part4-seed0-investigation/replay-predeclaration.json`).

## Question

Main seed 0 of `formal-decoder-02` memorizes at step 200 but does not reach
test accuracy 0.99 within the 10000-step budget (final test acc 0.0954),
while seeds 1–4 grok at logged steps 5000/5500/4500/6700. The sheet's
reference narrative says "every seed of yours should grok within 10000
steps". Implementation error or finite-optimization behaviour?

## Conservative verdict

**No confirmed implementation defect. The failure root cause is UNRESOLVED.**
An init-dependent optimization hypothesis is plausible but unproven. The
seed-0 scientific gates remain **unmet** exactly as reported in
`EVIDENCE_AUDIT.md`: grok_within_10000, delay>10, top-5 share>0.8
(0.3964), keep_only_key5>0.95 (0.1456), remove_random_other5>0.95 in all
10 draws (min 0.0844); seed 0 passes only remove_key5<0.05 (0.0112) and
keep_only_random_other5<0.05 (max 0.0137). Nothing in this investigation
changes those gates or their outcome.

## What was checked (observational evidence, exact numbers from result JSONs)

1. **Optimizer formula (diag01 + Codex independent test).**
   `grok.grok_adamw_step` (frozen, read-only) matches an independently
   written canonical decoupled-AdamW reference. Pi's check: worst
   scale-relative diff 5.39e-6 over 200 steps across lr ∈ {3e-3, 1e-2},
   wd ∈ {0, 1, 50} (absolute worst 4.41e-6; see initial-check-history for
   the failed first criterion and its revision). Pi's original diag01
   compared only parameters; Codex's independent float64 test
   (`independent-optimizer.json`) additionally compared the m and v states
   at every step (tolerance 1e-12): worst 2.45e-15 — PASS. These checks
   validate the update rule at tested points; they do not address float32
   training-dynamics effects over 10000 steps.

2. **Gradients (diag02).** `value_and_grad` agrees with float64 central
   finite differences on 128 probed coordinates (worst absolute
   disagreement 1.15e-10, criterion |fd−ad| ≤ 1e-6 + 1e-3·|ad|; the first,
   purely relative criterion failed near tiny gradients — history
   preserved). Again: tested-points validation only; no claim about
   accumulated float32 dynamics.

3. **Split and configuration (diag03).** Recomputed split: 2823 train /
   6586 test, disjoint, 50% nested; split independent of ambient RNG and
   run seed (frozen RNG 897). All 12 formal runs share one source-hash set
   and protocol settings. Seeds 1–4 grok on the same split seed 0 failed
   on, so the data is observationally not the differentiator.

4. **Initialization streams (diag04).** Each saved `checkpoint-init.npz`
   for seeds 0–4 is bitwise equal to `init_params(jax.random.PRNGKey(seed))`;
   inits pairwise distinct; init norms 27.64–27.87, top-5 shares
   0.122–0.130 (sheet reference ~28, ~0.13). Seed 0's init is exactly the
   intended stream: no seed mixing or contamination was found.

5. **Trajectory comparison (diag05) — descriptive, not predictive.**
   - Seed 0's test accuracy rises monotonically 0.017 (step 1000) → 0.095
     (step 10000). Seeds 1/2/3/4 exhibited the same accuracy values at
     steps 3800/4500/3000/1700 respectively; best log-space alignment puts
     seed 0's lag at ~5500–8100 steps.
   - Test-loss shows weight-decay-driven cycles (local maxima at 1300/
     2800/4200/5800/7200/8700, period ≈ 1450–1500), as in all main runs.
   - Log-linear growth rate 1.9e-4/step (doubling ≈ 3665 steps). Any
     extrapolation assumes the rate persists; grokking is non-linear, so
     **no prediction of eventual grokking is made or supported**.
   - Controls: wd0-s0 (same seed, wd 0) flat at 0.015–0.017 across steps
     6000–10000; random-labels-s0 at chance (0.0091). Seed 0 (wd 1) is
     outside those observed regimes in these windows.
   - Fourier: seed 0's final top-5 share 0.396 lies between its
     memorization share (0.160) and the grokked seeds' final shares
     (0.880–0.944). This is a snapshot descriptor; "mid-transition" is an
     interpretation, not an established fact.
   These observations are consistent with — but do not prove — that seed 0
   was still progressing toward generalization when the budget ended.
   Whether seed 0 groks beyond step 10000 is **unknown**; no authorized
   run tests it, and no claim either way is made.

6. **Log reproducibility (diag06).** Independent rescoring of all five
   `checkpoint-final.npz` reproduces the logged train/test accuracies
   exactly (diff 0.0); receipts match logs. This validates the scoring
   pipeline, not the training dynamics.

7. **Negative controls (diag07 v1-x64 preserved; diag08 robust).**
   - Preserved versioned artifacts: `diag07_negative_controls_v1-x64.py`
     and `.../results/diag07_negative_controls_v1-x64.json`
     (NC1 wrong beta2: rel diff 4.913e-3; NC2 coupled wd: 3.727e-1;
     NC3 sign-flipped gradient: disagreement 1.697e-2; all detected).
     The earlier float32 NC3 attempt produced FD = 0.0 against analytic
     gradient 8.864e-3 (float32 loss quantization made the FD
     uninformative); that defective control was rejected and its exact
     values are preserved in `initial-check-history.json` and the raw
     session log — never overwritten silently.
   - **diag08 (new)**: float64 FD sign-flip control at a coordinate with
     analytic gradient 4.286e-2 (> 1e-3), h ∈ {1e-3, 1e-4, 1e-5}: FD
     matches the unmodified gradient at every h, is nonzero, and rejects
     the sign-flipped gradient at every h. PASS.
   - The checks are therefore demonstrably sensitive to planted defects.

8. **Erratum (no code change).** The `grok.py` module docstring still
   describes the invalidated symmetric variant ("no positional embeddings,
   no causal mask", "62,305") while the actual frozen code and protocol
   have positional embeddings (192 params), a causal mask, and 62,497
   parameters. See `ERRATA-grok-docstring.md`. diag04's bitwise init
   verification confirms the trained artifacts correspond to the decoder
   code, i.e. the docstring is stale, not the model.

## Hypotheses (explicitly unresolved)

| Hypothesis | Status |
|---|---|
| Optimizer formula bug | no evidence found (diag01, diag02, Codex state test) — tested points only |
| Seed-specific split/data defect | no evidence found (diag03) |
| Init RNG stream contamination | no evidence found (diag04 bitwise match) |
| float32 precision failure in scoring/gradients | no evidence found at tested points; float32 training-dynamics effects NOT eliminated by these checks |
| wd0-style non-generalizing regime | inconsistent with observed seed-0 trajectory in the observed window |
| Init-dependent optimization delay (unlucky draw from an unspecified init RNG stream) | **plausible, unproven**; the sheet does not specify the stream, so the reference's per-seed grok steps are not transferable |
| Slow-but-eventual grokking beyond 10000 | **unknown**; untested and unauthorized |
| Undetected mechanism excluding generalization for this init | cannot be excluded by the evidence collected |

## Validation (aligned to the predeclared replay; no other training)

The only authorized training activity is Codex's **single predeclared
exact seed-0 replay** in `artifacts/part4/seed0-replay-investigation-01/
main-s0` (frozen `grok.py` hash `d0501c3b…`, seed 0, split 897, fraction
0.30, wd 1, lr 3e-3, betas (0.9, 0.98), cap 10000, pinned to CPU core 4), declared in
`week8-support/part4-seed0-investigation/replay-predeclaration.json` as a
reproducibility test, not a replacement attempt. Acceptance: compare all
logged numeric columns and raw checkpoint arrays with the original
(timestamps excluded) under unchanged gates; report any mismatch; no
retries; no extension beyond 10000 is planned or authorized. Pi performs
no replay outcome claims here — comparison belongs to Codex's independent
review. The earlier rev-1 suggestion of a 30000-step extension is **withdrawn**.

## Constraints compliance

- No training or retraining launched by Pi; frozen sources unmodified
  (hashes in `diagnostic-receipt.json`); nothing deleted; no remote writes.
- Diagnostic-check tolerances were revised during the session; all initial
  failures and exact values are preserved in `initial-check-history.json`
  and the retained raw Pi session log; scientific gates unchanged.
- No seed, split-seed, threshold, step-cap, or initialization changes;
  no outcome-driven retries; no commits before Codex final review.
- No secrets or private notes in any report.

## Files (rev 2)

- `root-cause-report.md` (this file), `diagnostic-receipt.json`
- `initial-check-history.json` — failed/revised initial checks with exact values
- `diag01–diag08*.py`, `diag07_negative_controls_v1-x64.py`, `run_all.sh`, `results/*.json`
- `ERRATA-grok-docstring.md`
- Copies (report + compact diagnostics) in
  `week8-support/part4-seed0-investigation/`
