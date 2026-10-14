# Week 6 — Energy-based models, Part 1 SPEC (dataset understanding)

Scope: **Part 1 only** this turn. Parts 2–4 are planned but not implemented.
All work confined to `week6/energy-based/`. Parent diffusion work untouched.

## Goal (from the learning sheet)

Image galleries with digit/object labels and class-frequency plots per
training/test split that identify what one sample represents and how the
training and test empirical distributions compare.

## Requirements

1. **Environment**: `uv`, CPU default. Independent `pyproject.toml`/`uv.lock`/
   `.venv` inside `week6/energy-based/`. `matplotlib` PNG figures, `pytest`
   tests. Numeric stack (JAX) installed for later parts.
2. **Data (exact source files, no substitutes)**:
   - MNIST: 4 gzip IDX files from
     `https://storage.googleapis.com/cvdf-datasets/mnist`
     (`{train,t10k}-{images-idx3-ubyte,labels-idx1-ubyte}.gz`).
   - CIFAR-10 binary tarball from `https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz`.
   - Provenance (URL, effective URL, size, SHA-256, fetch date) recorded in
     `evidence/provenance.json` and README.
   - `inputs/`, `runs/`, `artifacts/`, `.venv/` are git-ignored.
3. **Loaders (safe, reusable, tested)**:
   - IDX images: validate magic 2051, counts, 28×28 dims; uint8 0..255.
   - IDX labels: magic 2049, count matches images; uint8 0..9.
   - CIFAR binary records: 3073 bytes = 1 label + 1024 R + 1024 G + 1024 B;
     all 5 train batches + test batch read; no unsafe tar extraction
     (iterate expected members, read bytes only).
   - Pairing and split identity preserved (original train/test assignment).
4. **Visualization (reusable)**:
   - Grayscale/RGB gallery with optional labels, native pixels
     (`interpolation="nearest"`), correct class names.
   - Deterministic gallery selection: **first 5 stored examples per class per
     split** (documented to avoid cherry-picking).
   - Class-frequency bar charts from **all** labels of each split.
5. **Verification (Part 1 VERIFY items)**:
   - MNIST: 60000 train / 10000 test, 28×28 grayscale, uint8 0..255.
     Digit 1 most common and 5 least common in **both** splits.
   - CIFAR-10: 50000 train / 10000 test, 32×32 RGB;
     exactly 5000 train / 1000 test per class (fraction 0.1).
   - Counts/fractions printed; summary saved to `evidence/part1_summary.json`.
   - Tests include negative controls: wrong RGB channel order caught,
     mismatched MNIST counts caught, invalid IDX magic caught,
     class-frequency normalization correct, split separation preserved.
6. **README**: commands paired with figures; explanatory text on empirical
   distribution vs unknown population, gallery appearance vs frequency,
   pixel representation, balanced gallery ≠ balanced dataset, source identity.

## Acceptance

- `uv run pytest -q` green (loaders + viz + summary checks).
- `uv run python scripts/make_evidence.py` regenerates figures + summary.
- Real-data verification printed and consistent with the sheet's numbers.
- Commits scoped to `week6/energy-based/` only, order: spec/plan → tests →
  implementation → evidence.

## Future parts (settings recorded, NOT implemented now)

Part 2 RBM/CD reference settings: 128 hidden units, 20 epochs, batch 100,
plain gradient descent lr 0.1, seed 2026, matched initialization and data
order for CD-1/CD-10. Parts 3–4: classical & modern Hopfield per sheet.
