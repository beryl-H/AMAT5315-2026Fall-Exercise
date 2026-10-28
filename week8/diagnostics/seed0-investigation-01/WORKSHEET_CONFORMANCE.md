# WORKSHEET_CONFORMANCE.md — Week 8 Part 4 final conformance assessment

Date: 2026-10-28 · Author: pi (native WSL, volcengine-plan / glm-5-3-flash)
Authoritative input: the LOCAL 12-page sheet
`/mnt/d/Keria/Course/AMAT5315/week8-learning-sheet.pdf`
(sha256 `4a9f5d0cc5223ed6c1863410cc1f4e03b890dba21649265ca8a91eacd0b95101`)
and its text extraction `week8-support/learning-sheet.txt`.
No external reference code was obtained or used. This document is
user-facing; it contains no raw agent reasoning or credentials.

## 0. Instruction record (steering history)

1. Earlier phase: seed0 diagnostic (reports rev 1→rev 2), then Codex
   predeclared exact seed0 replay — **completed**: all 101 logged numeric
   rows and all 3 checkpoint arrays (init/memorization/final) bitwise
   identical to the original; all 147 original artifacts unchanged
   (`week8-support/part4-seed0-investigation/final-verification.json`).
   Scientific gates remain failed. No root cause proven; no confirmed model
   bug; init-dependent optimization is only a hypothesis.
2. A reference-code comparison phase (`reference-comparison-02`) was started
   and then **interrupted by user steering**: “按照讲义完成即可，不要去对比代码”
   (follow the worksheet; do not compare against code). External
   reference-code discovery/comparison is STOPPED; partial fetched files are
   preserved as superseded context only (`reference-comparison-02/SUPERSEDED.md`)
   and are used in no conclusion. The live course website's week 8 task was
   revised and does not match this sheet; the local sheet is authoritative.
3. Current phase: worksheet-only conformance assessment. No training, no new
   tests duplicating earlier checks, no tuning, no seed/split/init/budget/
   gate changes, no deletes, no remote actions, no commits.

## 1. Setup compliance (explicit DO requirements, Part 4 + general)

Independent crosscheck command (read-only):
`python3` re-derivation over `artifacts/part4/formal-decoder-02/*/` of
`settings.json`, `completion-receipt.json`, `train-log.jsonl`,
`fourier.json`, `ablations.json` (exact rows reproduced below in §2).

| # | Sheet requirement (Part 4 DO) | Observed | Compliance |
|---|---|---|---|
| D1 | One-layer transformer **without LayerNorm**, width 64, 4 heads, MLP 256; tokens `a, b, '='`; answer read at `'='`; task c=(a+b) mod 97 | `grok.py` frozen (sha `d0501c3b…`): matches; decoder structure per frozen protocol; docstring erratum only (see §5-E1) | COMPLIANT |
| D2 | Train on a **random 30%** of all pairs | fraction 0.30 → 2823/6586, disjoint, label-independent (frozen RNG 897) | COMPLIANT |
| D3 | **Full-batch AdamW, lr 3e-3, betas (0.9, 0.98), weight decay 1** | settings.json of all 12 runs: lr 0.003, betas (0.9, 0.98), wd 1.0 (wd0 controls 0.0); full batch (all 2823 pairs per step) | COMPLIANT |
| D4 | Run **up to 10000 steps**, stop 1000 steps after test acc ≥ 0.99 | `steps_run ≤ 10000` in all 12; grokked runs stopped exactly at grok+1000 (6000/6500/5500/7700; frac50 1500/1600/2000); non-grokking runs ran to the 10000/3000 cap | COMPLIANT |
| D5 | Log train/test loss, accuracy, **parameter norm every 100 steps for seeds 0–4** | main-s0..s4: log grid step 0 + every 100 verified (101 rows at full budget); all four quantities present in every row | COMPLIANT |
| D6 | Repeat **seeds 0–2 with weight decay 0** | wd0-s0/s1/s2 present, wd 0.0 | COMPLIANT |
| D7 | Repeat with a **50% training fraction** (seeds 0–2) | frac50-s0/s1/s2, fraction 0.5 (nested superset of the 30% split) | COMPLIANT |
| D8 | **Random-label run**: 3000 steps, main settings, seed 0 | random-labels-s0: max_steps 3000, permuted labels, seed 0, ran exactly 3000 | COMPLIANT |
| D9 | DFT of the residue embedding over the residue index; record top-5 power share at **init, memorization, after grokking** | fourier.json for all 12 runs (rfft over a=0..96, k=1..48, DC excluded from share) | COMPLIANT |
| D10 | Ablations: keep-only key5, remove key5, remove 5 random other (**10 draws**), keep-only 5 random other | ablations.json for all 12 runs (10 draws, frozen RNG 4497) | COMPLIANT |
| D11 | General: work in `week8/` with **uv + JAX, CPU default**; figures in `evidence/`; raw runs with settings in `artifacts/`; keep downloads/raw runs out of git | `.venv` + `uv.lock`, `JAX_PLATFORMS=cpu`; 4 part4 figures in `evidence/`; 12 run dirs each with `settings.json`; `.gitignore` excludes `artifacts/`, `data/` | COMPLIANT |
| D12 | General: commit after each task | repo history contains per-task commits through delivery; current diagnostic documents intentionally **uncommitted** pending Codex final review (per instruction) | COMPLIANT (historical); current phase on hold by instruction |
| D13 | Wrap-up: README with figures + producing commands; record fingerprint, settings, environment; push source, environment, evidence | README documents all part4 figures/commands/settings/environment; corpus fingerprint recorded (Part 1); **push not performed** — explicitly a pending user action, nothing was pushed | PARTIALLY COMPLETE (push pending; user-level, not model-level) |

Items NOT specified by the sheet (therefore **not violations**, values
recorded for transparency): initialization std/RNG stream per tensor,
exact split-permutation definition, `eps`, dtype, JIT settings, optimizer
state threading, logging JSON format, checkpoint formats, random-label
shuffle mechanism, ablation draw procedure details.

## 2. VERIFY items — exact observed rows (independently recomputed)

### VERIFY (1) — memorization long before generalization

Sheet reference expectations: every seed memorizes by step 200; grokking
steps 6500/5900/9200/3600/3300 (median 5900); delay ratio 16–46; **"Every
seed of yours should grok within 10000 steps, with a delay ratio above
10."**

Main runs (30%, wd 1), exact recomputed rows:

| seed | mem step | grok step | steps_run | delay | grok≤10000 | delay>10 |
|---|---|---|---|---|---|---|
| 0 | 200 | **null** | 10000 | **null** | **FAIL** | **FAIL** |
| 1 | 200 | 5000 | 6000 | 25.0 | PASS | PASS |
| 2 | 200 | 5500 | 6500 | 27.5 | PASS | PASS |
| 3 | 200 | 4500 | 5500 | 22.5 | PASS | PASS |
| 4 | 200 | 6700 | 7700 | 33.5 | PASS | PASS |

Descriptive reference context (not gates): reference test loss "near 10
nats" during memorization — ours peaks ~21–35 nats at wd-1 cycle maxima;
reference norm "≈46 during memorization, ≈32 after grokking" — ours peaks
≈48–51, falls to ≈28–33. Same qualitative behaviour (norm rises then falls
at grokking); exact bands differ from the reference implementation.

### VERIFY (2) — weight decay, data fraction and real labels control grokking

Sheet operative claim: without weight decay **no seed generalizes in 10000
steps**; with 50% fraction generalization tracks training; random labels
memorize by 200 with test accuracy at chance.

| control | observed |
|---|---|
| wd0-s0/s1/s2 | grok null in all; final test acc 0.0167/0.0093/0.0375 (all ≪ 0.05 → no generalization); norms 72.90/73.42/72.45 (> 69) |
| frac50-s0/s1/s2 | grok 500/600/1000; delays 2.5/3.0/5.0 (delay almost disappears, as stated) |
| random-labels-s0 | mem 200; final test acc 0.0091 ≈ chance 1/97 ≈ 0.0103 (finite-window observation) |

Descriptive note (not a gate): the reference narrative's wd0 final-accuracy
band "0.04–0.14" is higher than ours (0.009–0.038); the sheet's operative
requirement — no generalization within 10000 — holds for every wd0 seed.

### VERIFY (3) — embedding concentrates on a few Fourier frequencies

Sheet gate: **"Your share after grokking should exceed 0.8."** Reference
context: init share ≈ 0.13; after grokking 0.86–0.94; reference seed 0 uses
frequencies {2, 3, 32, 33, 38}.

| run | init share | mem share | final share | >0.8 |
|---|---|---|---|---|
| main-s0 | 0.1266 | 0.1601 | **0.3964** | **FAIL** |
| main-s1 | 0.1219 | 0.1683 | 0.9295 | PASS |
| main-s2 | 0.1276 | 0.1627 | 0.8798 | PASS |
| main-s3 | 0.1295 | 0.1782 | 0.9283 | PASS |
| main-s4 | 0.1279 | 0.1848 | 0.9443 | PASS |

Key frequencies are seed-dependent (ours: s0 {35,31,43,22,24}, s1…s4 their
own sets), as the sheet itself states ("Different seeds pick different key
frequencies"). frac50 final shares 0.916–0.969; wd0/random-label final
shares 0.136–0.166 (no concentration, consistent with no generalization).

### VERIFY (4) — the model computes with its key frequencies (ablations)

Sheet thresholds: keep-only key5 **stays above 0.95**; remove key5 **below
0.05**; remove 5 random others **above 0.95 in all 10 draws**; keep-only 5
random others **below 0.05** (reference: 0.997+ / 0.009–0.013 / 1.000 / 0.009).

| run | keep_key5 | rm_key5 | min rm_random5 (10 draws) | max keep_random5 | all four |
|---|---|---|---|---|---|
| main-s0 | **0.1456** | 0.0112 | **0.0844** | 0.0137 | **FAIL** (2 of 4) |
| main-s1 | 0.9997 | 0.0103 | 1.0000 | 0.0103 | PASS |
| main-s2 | 0.9998 | 0.0138 | 1.0000 | 0.0138 | PASS |
| main-s3 | 1.0000 | 0.0108 | 1.0000 | 0.0125 | PASS |
| main-s4 | 0.9991 | 0.0112 | 1.0000 | 0.0112 | PASS |

Controls are not required to meet these thresholds; observed values are all
near chance for wd0/random-label runs (no learned rule to ablate) and all
four thresholds also hold for the frac50 runs (keep 1.0, rm ≤ 0.0104,
min-rm-random 1.0, max-keep-random ≤ 0.0104).

### Cross-checks against independently recorded gates

`evidence/part4-summary.json` `main_gates` booleans and
`evidence/independent/independent-part4-final-audit.json` agree with the
recomputed rows above row-by-row (seed 0: 2/7 pass; seeds 1–4: 7/7 pass;
`all_main_gates_pass: false`). The exact seed0 replay reproduced all 101
log rows and all 3 checkpoint arrays bitwise; 147 original artifacts
unchanged.

## 3. Setup compliance vs scientific pass — separation

- **Setup/protocol compliance: PASS.** Every explicit DO requirement
  (architecture, split, optimizer, budgets, logging, controls, Fourier
  analysis, ablations, evidence layout) is met, and the exact seed0 replay
  proves deterministic reproducibility of the pipeline.
- **Scientific gates: PARTIAL FAIL.**
  - Main seed 0 **fails**: grok_within_10000, delay>10 (undefined),
    final top-5 share>0.8 (0.3964), keep_only_key5>0.95 (0.1456),
    remove_random_other5>0.95 (min 0.0844). Passes only remove_key5<0.05
    and keep_only_random_other5<0.05.
  - Main seeds 1–4 **pass all seven gates**.
  - Controls behave as the sheet requires (no wd0 generalization; fast 50%
    generalization; random labels at chance).
- No claim of complete course satisfaction is made; final verification
  (defend at Week 9) and the wrap-up push remain user-level actions.

## 4. Corrections

**No supported correction to the implementation has been identified.** All
protocol-frozen setup choices conform to the sheet; the sheet does not
specify initialization/RNG details, so the unspecified choices are not
violations. The seed-0 outcome is a legitimate observed result of the
specified program; the implementation must not be modified merely to force
target metrics, and no gate was or will be relaxed. Root cause of the
seed-0 failure remains unresolved (init-dependent optimization is a
hypothesis, unproven).

## 5. Errata (factual documentation notes; frozen originals untouched)

- **E1 (existing).** `grok.py` module docstring describes the invalidated
  symmetric variant ("no positional embeddings, no causal mask", "62,305");
  the actual frozen code/protocol have learned positional embeddings (192
  params), a causal mask, and 62,497 parameters. Governing erratum:
  `week8/diagnostics/seed0-investigation-01/ERRATA-grok-docstring.md`.
- **E2 (new, this report).** In earlier summaries the phrase "every seed
  memorizes by step 200" should be read as "memorization step ≤ 200":
  main seeds 0–4 and wd0-s0/s2 memorized at exactly 200, but wd0-s1
  memorized at **step 100** (first logged step with train acc ≥ 0.99) —
  still consistent with the sheet's reference statement; recorded here so
  the exact value is preserved.
- **E3 (new, this report).** Rev-1 of the seed0 root-cause report
  overstated causality ("censored mid-transition", unlucky init as "the
  remaining cause"); these were already withdrawn in rev 2. The
  conservative status stands: no confirmed implementation defect, root
  cause unresolved.
- No other factual errors were found in the previously verified artifacts
  during this crosscheck; all previously reported numbers re-verified
  exactly. Prior final artifacts and packaged archives are preserved
  unchanged.

## 6. Unresolved status (concise)

- **Seed 0 main run does not satisfy the sheet's grok-within-10000 /
  delay>10 / Fourier-share / keep-key5 / remove-random expectations within
  the mandated 10000-step budget. Unresolved why.** No implementation bug
  confirmed; deterministic reproducibility proven (bitwise replay);
  init-dependent optimization hypothesis plausible but unproven; whether
  seed 0 would grok beyond the mandated budget is unknown and untested
  (no extension authorized).
- Seeds 1–4 and all controls conform to the sheet's scientific
  expectations within the specified budget.
- Pending user-level items: Week-9 live demonstration of the figures
  ("final verification"), and the wrap-up push (nothing was pushed).
