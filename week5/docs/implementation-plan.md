# Week 5 completion implementation plan

Goal: Complete lecture Parts 2-4 with actual Enzyme derivatives and audited evidence.
Architecture and authority: seismic-spec.md and ../seismic.design.toml.
Stack: Rust nightly-2026-09-05 + enzyme; Python 3.12, JAX, NumPy, matplotlib, pytest.
Execute inline in the named existing week5 folder, with task review after each stage.

- [x] 1. Freeze specification and toolchain. Re-run existing Part 1 tests; install the
  course enzyme-setup skill and pin an isolated compiler under ignored inputs/.
- [x] 2. Define forward tests in tests/test_seismic.py: hand-derived one-step fixture,
  boundary/CFL rejection, and lecture gather norms, peaks and diagonal displacement.
  Run pytest and save its missing-executable failure before adding Rust source.
- [x] 3. Implement Cargo.toml, build.rs, src/kernel.rs, src/lib.rs, src/main.rs and
  NPY I/O; smoke-test Enzyme cube (8,12,12); build negative control without autodiff
  must fail. Build/install forward CLI; pass independent solver tests; commit.
- [x] 4. Add derivative tests first: Born FD on reduced model, independent random
  weights, repeated receivers, velocity parameterization and zero boundary adjoint.
  Observe failing unsupported modes, then implement timestep JVP/VJP composition,
  full-history saved-state counters and recordings. Verify dot product and image.
- [x] 5. Add Treeverse tests first: N=6,budget=2 exact counts; lecture budget counts,
  descending unique grad coverage, saved-slot provenance, valid restores, budget
  enforcement and bitwise unchanged image after replay. Implement src/schedule.rs;
  run each reflector budget; audit all action files; commit.
- [x] 6. Run Marmousi Born and budget-5 reverse sequentially. Record actual runtime,
  image norm, six saved states and bytes; never allocate full Marmousi history.
- [x] 7. Create scripts/plot_seismic.py, scripts/verify_evidence.py, audit utilities,
  scripts/run_all.sh and pinned Python project files. Produce all lecture PNGs,
  machine-readable verification JSON and complete command/evidence README. Inspect
  all PNGs visually; record lecture viewer replacement if a matplotlib frame is used.
- [ ] 8. Run fresh end-to-end checks; confirm no ignored input/NPY files are tracked,
  each evidence file <5 MB, Git diff clean of private data; commit and push origin
  only within this exercise task; record failures and corrections in evidence log.
