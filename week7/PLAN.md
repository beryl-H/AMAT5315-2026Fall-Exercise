# Week 7 implementation and acceptance plan

Goal: complete the three ordered parts in the selected local PDF, with raw
evidence, independent numerical audits and readable figures.
Architecture: sparse.py owns the convex 1D study; inpainting.py owns the DCT
solver/validated observations; study.py owns evaluation only; repaint.py owns
conditional sampling; figures.py reads saved evidence. The Week 6 source and
checkpoint remain read-only. Spec: SPEC.md. Global constraints are in SPEC.md.

- [x] Part 1: write tests for the DCT basis, signed L1 optimum, observation-only
  constraints and exact RNG stream; run failing tests; implement sparse.py;
  run the exact 120-trial command; independently recompute errors, LP residuals
  and basis/RNG; draw the requested representative trials; commit.
- [x] Part 2: test ties, unbounded outputs, observed restoration, masks, hidden
  MSE and real CLI; record initial failures; implement inpainting.py, inpaint,
  study.py and inpaint.design.toml. Verify official image identities; run batch
  and standalone commands; audit all 20 reconstructions against an independent
  SciPy loop and the fixed mean thresholds; generate/inspect three PNGs; commit.
- [x] Part 3: test skipped-level posterior/final step, noise matching, exact
  split ordering, t-1 network indexing and trace size with a tiny deterministic
  predictor. Implement repaint.py using existing Week 6 loader. Compare model
  outputs between old/new environments; run 20-image batch for four fixed
  seeds once, retaining each seed immediately. Independently replay all 298
  transitions, verify final conditioning and float64 ensemble identity. Generate
  four PNGs, inspect and commit.
- [x] Final: run full pytest/compile checks, audit raw artifacts/README links
  and input/checkpoint hashes, record failures/corrections, export code/figures
  and raw evidence with per-file hashes, then report exact delivery state.

Execution stays in this session; no subordinate agent or external model call
is required. Commands and measured results are appended to README/AUDIT.
