# TASK_STATUS — week8/language-models (revised 24-page sheet)

Agent: native WSL `pi` (provider `volcengine-plan`, model `glm-5-3-flash`).
Support snapshot: `/mnt/d/Keria/Course/AMAT5315/week8-support/revised-4e33bea92945/`
(learning-sheet.pdf SHA256 `4e33bea9…69cc35`, 24 pages). Old `week8/` results
(commit 24a860b) untouched.

## Completed

- Phase 0 scaffold (a1347fa): SPEC.md, PLAN.md, pyproject.toml, uv.lock,
  README, .gitignore; `uv sync` OK; JAX 0.6.2 CPU verified.
- SPEC/PLAN corrected per root preflight review (d348db2): Part 1 20-shuffle
  ±0.09 demoted to descriptive; Part 3 LS ISR redefined via H = K pinv(K)
  (h_jj own coefficient) + d=1/M=2 regression case; Part 4 factor-32
  denominator convention; Part 5 64-target scoring windows (starts 0, 64, …,
  one shared boundary token), common target stream tok[1:], structural
  causality control only; provenance moved to INPUTS.md +
  evidence/input-manifest.json; Part 6 literal fit spec (θ = logE/logA/logB/
  α/β, 3⁵ grid, δ-scaled Huber, L-BFGS-B bounds/tolerances).
- Phase 1 inputs (d348db2): all 5 pinned files verified (hashes, bytes,
  schemas) by `uv run python src/check_inputs.py` (exit 0). WSL curl exit-55
  failure preserved; Chinchilla CSV copied byte-exact from root's Windows
  urllib retrieval. Full SHAs API-confirmed for TinyStories +
  analyzing-chinchilla; two HF model-repo lookups timed out (recorded).
- Phase 2 RED (4bd3a8c): 5 test modules failed on missing `src.*`; log at
  evidence/test-logs/red-phase2.log; committed before implementation.
- Phase 3 GREEN + Part 1 (3818f47): 27/27 tests pass (incl. data-bound:
  split 24867/2763, ~2.05 chars/token, exactly 3 roundtrip exceptions,
  (29528, 256) table, 54/54 single-token words). Sabotage negative control:
  corrupted row 'cat' detected (cosine 0.6855 → 0.6681). Part 1 inference:
  table means 0.466/0.549/0.433 (gate ≥ 0.3 PASS), control seed-1730
  +0.030/−0.015/+0.009 (gate |·| ≤ 0.15 PASS); 20-shuffle means within
  [−0.089, +0.085] (descriptive); figure evidence/part1-relations.png.

## Preflight package

- READY_FOR_PREFLIGHT.json written (bindings: source snapshot, input hashes,
  test logs/results, Part 1 results, explicit not-yet-run list).

## Not started (deliberately — preflight boundary)

- All formal Parts 2–6 work (see READY_FOR_PREFLIGHT.json list).
