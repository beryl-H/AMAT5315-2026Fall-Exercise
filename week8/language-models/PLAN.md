# PLAN — Week 8 (revised), six-part language-model worksheet

Correctness contract: `SPEC.md`. All conventions below are frozen before any
formal result. Gates and commands are named per phase; commits happen after
each numbered phase in this order (spec → failing tests → implementation →
results), preserving meaningful history.

## Phase 0 — Scaffold (this commit)

- [x] `SPEC.md`, `PLAN.md`, `pyproject.toml`, `uv.lock`, `README.md` skeleton,
      `.gitignore`, `TASK_STATUS.md`.
- [x] `uv sync` resolves; `uv run python -c "import jax"` works on CPU.
- Environment: Python 3.12.3, numpy 2.2.6, jax/jaxlib 0.6.2 CPU, matplotlib
  3.10.3, pytest 8.4.2, sentencepiece 0.2.2, safetensors 0.8.0, scipy 1.17.1.

## Phase 1 — Pinned inputs + provenance

- [ ] Download the four pinned inputs (see SPEC table) via primary public URLs.
- [ ] Record in `data/MANIFEST.md`: resolved commit, URL, SHA256, bytes, schema
      (safetensors dtypes/shape; tokenizer.json type; SPM model; CSV columns).
- [ ] Data files stay in git-ignored `data/`; the manifest is committed.
- Verify: `uv run python src/check_inputs.py` prints the manifest table and
  exits nonzero on any hash/schema mismatch.

## Phase 2 — Failing tests first (input preparation + Part 1)

Tests derived from SPEC (not from implementation):

- `tests/test_corpus.py` — synthetic fixtures: split on `<|endoftext|>`, strip,
  drop empty, first-90%-of-whole-stories split (2 stories, 50/50 case; 9+1
  stories → 9/1; strip only surrounding whitespace; empties dropped).
- `tests/test_tokenizer.py` — real `tok512.model`: EOS-between-stories joining,
  no BOS, roundtrip on a small sentence; chars/token reported on a fixture.
- `tests/test_part1.py` — unit tests with a tiny synthetic table (4×3):
  unit-row scaling, offset/LOO-cosine math on a hand-computable case, control
  shuffle changes assignment but not the vector multiset, mean-of-9 aggregation.
- Data-bound checks (`tests/test_data_checks.py`, run only when `data/` is
  present): shapes/keys of the embedding table (29528 × 256), corpus split
  counts 24867/2763, ≈ 2.05 chars/token, exactly-3 roundtrip exceptions, all 54
  Part-1 words single-token.

Run with data present so the RED run shows real failures; log preserved at
`evidence/test-logs/red-phase2.log`; commit tests before implementation.

## Phase 3 — Input preparation + Part 1 implementation (GREEN)

- [ ] `src/corpus.py` (split/encode/roundtrip), `src/part1_embeddings.py`
      (table load, single-row lookup, LOO cosines, controls).
- [ ] `uv run pytest tests/ -q` green; log `evidence/test-logs/green-phase3.log`.
- [ ] Negative control for the Part 1 test bench: corrupt one embedding row and
      confirm the data-bound test catches it (sabotage check, recorded).
- [ ] Part 1 formal inference (`src/run_part1.py`, seed 1730 + 1731..1750):
      means, control means, 20-shuffle spread, figure (table above axis, control
      below), result JSON in `evidence/`. Gates from SPEC §Part 1 checked and
      reported with actual values.
- Commit implementation + evidence.

## Phase 4 — Preflight package

- [ ] `READY_FOR_PREFLIGHT.json`: source file list + SHA256s, input hashes,
      test commands + outcomes, Part 1 results, RNG table reference, explicit
      list of what has NOT been run yet (Parts 2–6 formal work).
- [ ] `TASK_STATUS.md` updated with actual status.
- STOP: concise report for independent audit before expensive formal steps.

## Phase 5+ — Formal parts (NOT in this phase; planned order)

5. Part 2 formal: k-selection (last 10% of train tokens; refit full train),
   n-gram ladder n = 1..6, explicit 512/512²/512³ normalization enumeration,
   bigram sampler + pooled chi² + bad-lag negative control, curse-of-
   dimensionality figure. Memory-efficient sparse counts only.
6. Part 3 formal: preflight small-M benchmark → bounded chunked-BLAS sweep
   (d 16/32/64, M 8..65536, stores outer/LS/ReLU^1,2,3, all stored keys
   queried), capacity interpolation, slope fits; then the 16384-batch MLP LM.
7. Part 4 formal: 算 bitmap rendering + provenance, 20-draw validation +
   predeclared showcase draw, four stores × four M.
8. Part 5 formal: transformer training (log/save/reload), held-out bpc,
   position-wise loss, causality negative control, sampler chi² suite,
   openings seed 1848, entropy z-scores.
9. Part 6 formal: drop-5 fit, grid multistart L-BFGS, 400 bootstrap refits,
   compute-optimal curves, printed-parameter residual comparison.
10. README with each evidence figure beside its producing command; final
    Chinese study guide; repo-root README link (only touch to existing files).

Each formal phase re-checks its SPEC gates and records actual values, including
failures. Checkpoints between phases allow independent audit.
