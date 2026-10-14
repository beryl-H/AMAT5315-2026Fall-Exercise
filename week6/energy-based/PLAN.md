# Completion status — 2026-10-14

- [x] Part 1: original dataset provenance, loaders, figures and independent verification.
- [x] Part 2: matched CD-1/CD-10 training, reconstruction, generation and conditional completion.
- [x] Part 3: 84-point classical Hopfield study and independent replay.
- [x] Part 4: 600 queries, 4200 outputs, library attention equality and classical comparison.
- [x] Final source/test review, 50 tests, figures and full-array acceptance.
- Final delivery receipts (backup checksum and remote commit) are recorded in the local delivery audit.

The original Part 1 implementation plan is preserved below.

# Part 1 implementation plan

1. **Setup**: `pyproject.toml` (numpy, matplotlib, jax[cpu], pytest),
   `uv sync`, `.gitignore` (inputs/, runs/, artifacts/, .venv/, __pycache__/).
2. **Downloads** (background, started first): MNIST 4 gz files + CIFAR-10
   binary tarball → `inputs/`. Log to `runs/download.log`. Record provenance
   (URL, size, SHA-256, date) via `scripts/provenance.py` →
   `evidence/provenance.json`.
3. **Tests first (TDD)**: `tests/test_loaders.py`
   - synthetic IDX encoder → loader roundtrip; invalid magic rejected;
     count mismatch image/label rejected; wrong dims rejected.
   - synthetic CIFAR batch: channel order (R plane first) verified by
     distinctive per-plane values; wrong record length rejected.
   - frequency function: normalization to 1, all classes present.
   - split separation: train/test arrays are independent objects with
     preserved original assignment.
4. **Implementation**: `src/ebm/data.py` (IDX + CIFAR loaders,
   `class_frequencies`), `src/ebm/viz.py` (`image_gallery` grayscale/RGB,
   `frequency_bars`), `scripts/download_data.py`, `scripts/provenance.py`,
   `scripts/make_evidence.py`.
5. **Evidence**: galleries (MNIST train/test, CIFAR train/test; first 5 per
   class, native pixels), frequency plots from all labels,
   `evidence/part1_summary.json`. Verify sheet numbers (MNIST 1 max / 5 min
   in both splits; CIFAR exact 0.1 per class).
6. **README**: provenance table, command↔figure pairs, explanation paragraphs.
7. **Commits** (scoped to `week6/energy-based/`): spec+plan, tests (red),
   implementation (green), evidence+README.
