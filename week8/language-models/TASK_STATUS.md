# TASK_STATUS — week8/language-models (revised 24-page sheet)

Agent: native WSL `pi` (provider `volcengine-plan`, model `glm-5-3-flash`).
Support snapshot: `/mnt/d/Keria/Course/AMAT5315/week8-support/revised-4e33bea92945/`
(learning-sheet.pdf SHA256 `4e33bea9…69cc35`, 24 pages). Old `week8/` results
(commit 24a860b) untouched — verified again this phase (0 changed files).

## Completed

- Phase 0–4 (previous phases): scaffold, corrected SPEC/PLAN, verified pinned
  inputs, red-first tests, Part 1 inference (all mandatory gates PASS),
  READY_FOR_PREFLIGHT.
- Part 1 figure geometry corrected (equal aspect, re-rendered from saved
  numbers into `evidence/part1-relations-equal-aspect.png`; data unchanged;
  numbered pair key in README). Cosmetic QA pending: title overlap / xlabel
  clipping — deferred per instruction.
- **Part 2**: root-interrupt handled (multi-writer formal-01 invalidated with
  full provenance; worker stop receipt written; no live workers found). The
  pooled-chi2 rare-bin defect root identified is fixed with red-first tests
  (rare pool merged with the smallest adequate bin until ALL expected ≥ 5;
  single-group contexts contribute 0 stat / 0 df; metadata recorded).
  Driver rebuilt: atomic exclusive attempts (`--attempt`, exist_ok=False
  before settings write), SHA256+length verified caches, incremental
  crash-safe results JSON, sample/bad-sample/P-table saved for audit,
  bigram-only explicit 512³ enumeration, PPL-3% + monotonic-ladder gates
  added. `CONTROL_PART2_READY.json` published — formal run dispatched by the
  ROOT (agent runs no Part 2 formal attempt).
- **Part 3**: exact all-queries memory sweep driver (`src/memory_sweep.py`),
  frozen protocol (full-65536 prefix draws, seeds 101/102/103, own-normalized
  ISR, median aggregation, log-log capacity with never-crossed → None),
  chunked ReLU tiles (BLAS threads 8 predeclared), JSONL progress. Smoke done
  (d=16, M∈{8,32,128}, seed 101). Full sweep reserved for root dispatch.
- **Part 4**: 算 rendered from SimHei with provenance (font hash recorded,
  bitmap only committed, tofu checks passed). Formal 20-draw validation
  (seeds 531504..531523, showcase 531504 predeclared) — **all mandatory gates
  PASS**: outer R(1024)=3.89, ReLU R(1024)=2.82 (15% gates), ReLU² R(4096)=1.189>1,
  softmax median+showcase < 0.056 at every M. Every raw outcome retained.
- **Part 6**: literal-fit fitter (`src/scaling.py`: logE/logA/logB + α/β
  bounds, δ-scaled Huber with analytic gradient vs central-FD test at 1e-7,
  logsumexp safety, frozen 243-grid/bounds/tolerances, 400-seed bootstrap
  driver). Smoke on the real 240 filtered runs (3 starts): converges, own
  mean log residual −0.000656, printed params +0.0167, D*/N* ≈ 17.9 — smoke
  only. Full fit + bootstrap reserved for root dispatch.

## Test status

`uv run pytest tests/ -q` → **57 passed, 0 skipped**. Red-first commits
preserved for every task this phase (pooled-chi2, part2 driver, memory sweep,
part4, scaling).

## Pending (root dispatch / next phase)

- Part 2 formal-02 (root), Part 3 full sweep (root), Part 6 full fit + 400
  bootstrap (root).
- Part 3 MLP LM (16384-batch, with compressed-logit equivalence test) and
  Part 5 transformer — next phase; no model training yet.
- Cosmetic figure polish (Part 1 title/xlabel, glyph preview title).
