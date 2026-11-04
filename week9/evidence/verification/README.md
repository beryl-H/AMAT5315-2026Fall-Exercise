# Verification receipts

Current acceptance is the full `audit-*.json` referenced by the parent README,
plus `build-bounded-tests.log` (20 Rust tests) and the Python console readback.
The complete fresh reproduction passed as recorded in `reproduction-smoke.log`.

Actual commands, with the local Rust/Python environments activated:

```bash
cargo +1.96.0 fmt --all --check
cargo +1.96.0 test --release --locked
cargo +1.96.0 build --release --locked
cargo +1.96.0 install --locked --path . --root ../week9-support/install
cargo +1.96.0 tree --locked -e features
python -m unittest discover -s tests -p test_python_audit.py -v
python verify_and_plot.py
bash reproduce.sh
```

Model commands are preserved in `reproduce.sh` and explained in the Chinese
guide. `run-final-models.log` is the actual final primary run output plus GNU
time's process statistics. Contraction work is a structural term count; BP
work counts documented multiplications. Neither is an exact wall-time model.

Historical failures remain visible:

- `build-first-pass.log`: Rust shape-inference error, corrected explicitly.
- `build-test-pass2.log`: early BP failures, corrected and superseded by the
  final 20-test pass.
- `run-promedus-path-failure.log`: vanished temporary executable, exit127;
  recovered using a persistent build cache and a copied executable.
- `run-final-models-oom.log`: independently reoptimized query phase exceeded
  memory, exit137. Query plans now reuse the scalar tree with proven bounds.
- `memory-probe-receipt.txt`: isolated all-one synthetic step probes; these
  are engineering diagnostics and not Promedus numerical acceptance.
- `probe-plan139.log`: plan-only cost/size diagnosis, no tensor evaluation.

Raw provider sessions, credentials, tool reasoning streams, compiler caches
and earlier local run archives are excluded from the public repository.
