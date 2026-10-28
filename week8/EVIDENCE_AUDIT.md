# Week 8 evidence audit — execution stages, attribution, commands, gates

This file records HOW the Week 8 evidence was produced and checked. It is a
documentation record only; frozen planning/checklist files were not mutated.
No training, tuning or threshold modification was performed in this closeout.

## Frozen inputs and source hashes (verified against receipts)

| item | SHA256 |
|---|---|
| corpus `week8/data/input.txt` (git-ignored) | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| learning-sheet PDF (`week8-learning-sheet.pdf`) | `4a9f5d0cc5223ed6c1863410cc1f4e03b890dba21649265ca8a91eacd0b95101` |
| `week8/grok.py` | `d0501c3baf88a1194665123e47abc0a5b0035d0b203a56ca0dcc92853bc1d447` |
| `week8/SPEC.md` | `3af0d721e80e2d5070223a4e1ebc5e5bac17f6c6b01edc8b34c1fbd110b3bee1` |
| `week8/run_part4.py` | `f2ba51117119d285df63b6a69dcbf9a5710c4fff23ee956882a33b6c5ff19693` |
| `week8/PART4_DECODER_PROTOCOL.md` | `dc30615431deb72d12d810d7a1b72bf487297b37419924b1093623d9b07a9076` |
| sealed blind prediction `artifacts/part3/formal-fit-01/prediction/prediction192.json` | file `8c39f12d4314179fce66354f4291dc4a464a44a6592c846e937112e8fcfd1661` (sealed payload hash recorded inside: `74f6c3daf7d6809bfb168f1c6310b1459e02efb5a6f3abea0e99d7716392bc5b`) |
| `week8/uv.lock` | `c510bfd3e9605f29d67ae8847f46a46c3a9219272bc2bc0dcb93fbef3a0114a1` |

Per-run metrics/settings/checkpoint hashes for the 12 Part 4 runs are in
`evidence/part4-summary.json` (`per_run.*.metrics_sha256` etc.) and were
matched against the manager receipts by both `make_part4_evidence.py` and the
independent audit.

Environment: Python 3.11.17, JAX/jaxlib 0.6.2 (TFRT_CPU_0, CPU-only,
`JAX_PLATFORMS=cpu`), numpy 2.2.6, scipy 1.17.1, matplotlib 3.10.3,
uv 0.13.0 (`evidence/environment.json`). No GPU/cloud was used anywhere.

## Actor attribution

- **Pi (native WSL, provider `volcengine-plan`, model `glm-5-3-flash`)**
  authored the models, workers and fit code (`baselines.py`,
  `transformer.py`, `grok.py`, `scaling.py`, `run_part1.py`, `run_part2.py`,
  `sweep.py` and scaling fit code, `run_part4.py`, `make_part3_evidence.py`,
  `make_part4_evidence.py`) and the original main evidence scripts, plus
  this file's subject matter. One earlier Pi turn (turn 08) ended with
  `429AccountQuotaExceeded` with model stopReason=error although the wrapper returned exit 0 — recorded as a failure,
  not a success; service reset 13:07:34 followed by a bounded no-tools
  connection test with `stopReason=stop`.
- **Codex** coordinated all formal workers (dispatch/manager scripts), wrote
  the dispatch layers and independent tests, repaired the wd
  parameter-threading calls in the Part 4 controls (a real wd0 bug, not
  transient contention), committed the Pi-authored `run_part4.py` in
  `7b9719b`, made mechanical plot/timestamp/entrypoint fixes, repaired the
  Part 4 postprocessing None-delay formatting and label overlaps (commit
  `904ec5b`), and performed the independent audits.

## Execution stages (all attempts preserved; nothing deleted)

1. **Part 1** — `uv run python run_part1.py`; deterministic; all 6 SPEC
   acceptance checks PASS (`evidence/part1-summary.json`).
2. **Part 2** — smoke, then formal `attempt formal-01`
   (`JAX_PLATFORMS=cpu taskset -c 0-3 uv run python run_part2.py --attempt
   formal-01`); 116,673 parameters; held-out loss 1.699744, PPL 5.47255;
   causality probe exactly 0; unmasked negative control detects leakage
   (2.996). Independently re-verified
   (`evidence/independent/independent-part2-audit.json`).
3. **Part 3** — smoke runs, then 18 fit runs (widths 8–128 × seeds 1,2,
   1500 steps) plus 2 holdout-192 runs dispatched via the Codex coordination
   layer at `week8-support/coordinate_part3.py` (exact executed source kept
   as `coordinator-source.py` in each attempt). The portable repository copy
   and safe `runner_part3.sh` wrapper were added after these runs; the old
   unsafe wrapper was never executed. Per-worker command pattern:
   `env JAX_PLATFORMS=cpu taskset -c <core> .venv/bin/python sweep.py
   --width W --seed S --steps 1500 --out <attempt>/wWWW-sS`). Sealed blind
   prediction written 2026-10-28T04:20:00Z, BEFORE any width-192 step
   (first worker start 04:20:15.6Z). Actual holdout mean 1.722199 is
   OUTSIDE the sealed 95% interval [1.611909, 1.667537] — a legitimate
   FAILED blind prediction; no refit/widening/leakage; `prediction192.json`
   immutable. Postprocessing: `uv run python make_part3_evidence.py`.
4. **Part 4 predecessor (INVALIDATED)** — `artifacts/part4/formal-01` used a
   symmetric no-position/no-mask variant. Verified capacity defect: 393
   conflicting unordered training pairs ⇒ maximum reachable train accuracy
   0.860786 < 0.99, so the required random-label memorization control is
   impossible. Stopped, all files preserved, record
   `evidence/independent/part4-symmetry-defect.json` and
   `artifacts/part4/formal-01/INVALIDATED-PROTOCOL.json`. Its two completed
   main runs are NEVER mixed into decoder results. `grok_symmetric_variant.py`
   preserves the model implementation with a provenance comment; the byte-exact
   original is recoverable from commit `7b9719b` and is also saved under
   `week8-support/audit/grok-symmetric-variant-exact-7b9719b.py`.
5. **Part 4 corrected decoder** — protocol `PART4_DECODER_PROTOCOL.md`
   frozen BEFORE training (commit `473efb3`: learned 3×64 positions + causal
   mask, only LayerNorm removed; 62,497 parameters; decoder capacity for the
   393 conflicting pairs verified by `tests/test_part4_decoder.py`). 12
   fixed jobs dispatched by `coordinate_part4.py` under
   `artifacts/part4/formal-decoder-02/` (worker pattern:
   `env JAX_PLATFORMS=cpu taskset -c <core> .venv/bin/python -c "...grok.
   run_one(outdir=..., seed=..., fraction=..., wd=..., max_steps=...,
   log_every=100)"`; per-run exact commands in
   `evidence/part4-summary.json`). Manager completed with status COMPLETE;
   peak combined RSS of the active workers was 1432.8 MB (not a single-worker
   measurement), as recorded by the manager and independent audit.
6. **Part 4 postprocessing** — `uv run python make_part4_evidence.py`
   (strict-gated on manager COMPLETE + 12 hash-matched metrics). First
   execution exited 1: the legitimate no-grok seed 0 has `delay_ratio null`,
   which a `.2f` format could not render. Codex repaired the formatting and
   overlapping labels/legends and removed the blanket ablation title —
   mechanical output fixes only, no scientific change, no retraining
   (commit `904ec5b`, record
   `evidence/independent/final-postprocessing-corrections.json`). All 10
   final PNGs were visually checked.
7. **Independent Part 4 audit** —
   `evidence/independent/independent-part4-final-audit.json`: independent
   data/splits and saved-weight scoring; NumPy rfft/irfft ablations
   independent of the Pi trig projection; no training replay.
   **Engineering audit PASS on all 12 runs**; **all_main_scientific_gates_pass
   = FALSE** (seed 0 did not grok). Audit precision note: one reported
   float32 mean loss (wd0-s0, ≈235.527) differed from the independent
   float64 mean by 9.5e-6 — within four float32 ULPs; audit threshold
   clarified, no scientific threshold changed.

## Test evidence

- `uv run pytest tests/ -q` — 86 passed, 14 upstream Matplotlib/pyparsing
  deprecation warnings; tests were written before implementation and cover
  the frozen model, fit, scorers and the decoder capacity check. The suite
  was independently rerun against the unchanged frozen sources.
- All 12 Part 4 saved-weight audits passed (engineering PASS per run).
- Negative controls performed: unmasked-attention leakage probe (Part 2),
  synthetic parameter recovery / leakage guards (Part 3), conflicting-pair
  capacity check (Part 4 decoder), plus the preserved invalidated variant.

## Final Part 4 numbers and gates (failed gates reported verbatim)

Main seeds (30% split, wd 1): memorization step 200 for all; grok steps
null/5000/5500/4500/6700 for seeds 0–4 (delays null/25.0/27.5/22.5/33.5);
final test accuracy 0.0954/1.0000/1.0000/1.0000/1.0000; final top-5 shares
0.3964/0.9295/0.8798/0.9283/0.9443. Seed 0 fails: grok_within_10000,
delay>10, top-5 share>0.8, keep_only_key5>0.95 (0.1456), and
remove_random_other5>0.95 in all 10 draws (min 0.0844). Seed 0 passes only
remove_key5<0.05 (0.0112) and keep_only_random_other5<0.05 in all 10 draws
(max 0.0137). Seeds 1–4 pass all seven gates (per-seed booleans in
`evidence/part4-summary.json` and the audit JSON).

Controls (observed results, NOT guaranteed successes): wd0 seeds 0–2 no
grok by 10000 steps, final test accuracy 0.0167/0.0093/0.0375, norms ≈ 72–74;
50%-fraction seeds grok at 500/600/1000 (delays 2.5/3.0/5.0); random-labels
seed 0 memorizes at step 200 and ends the OBSERVED 3000-step window at test
accuracy 0.0091 (chance ≈ 0.010) — a finite-window observation, not a claim
about all future training.

## Limits and interpretation discipline

- The Part 3 MISS is a legitimate failed blind extrapolation. Candidate
  explanations (power-law mismatch, finite-seed uncertainty, training regime
  at fixed token exposure D = 3,072,000) remain hypotheses; no new
  experiment distinguished causes. D is fixed token exposure, NOT fixed
  FLOPs: total dense-model compute still grows with model size at fixed D.
- The Part 4 scientific conclusion "the model computes with its key
  frequencies" is supported by the saved-frequency ablations and applies
  only to the seeds whose measured gates pass (seeds 1–4). Parameter-norm
  curves suggest a representation change; they do not prove a literal lookup
  table or a universal cause.
- No human/scientific acceptance is claimed from tests or audits. Nothing
  was submitted or published; no remote push was performed.
