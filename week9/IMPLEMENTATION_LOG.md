# Implementation activity

This is a chronological activity log; earlier checkpoint statements and hashes
are historical. Current accepted results and the executed binary hash are in
README.md and EVIDENCE_AUDIT.json. Paths under `../week9-support/` refer to the
retained local coordination archive. Selected public logs are copied into
`evidence/verification/`; private provider sessions are not part of this delivery.

- Read all 15 pages of week9-learning-sheet.pdf with pypdf and UTF-8 output.
- Downloaded official crates.io sources for tenferro 0.9.0 and omeco 0.2.6 into vendor-reference for API inspection (reference only, not dependencies).
- API correction: tenferro 0.9.0 uses CpuEngine::serial and TensorEinsumExt::einsum_subscripts, not older CpuBackend examples.
- API limitation: omeco 0.2.6 hard-codes SmallRng trial seeds to 42+trial_index; TreeSA CLI accepts seed 42 and records all trial seeds. No unsupported external seed is claimed.
- Wrote behavior tests before production modules. Initial cargo test could not run because Windows cargo was absent. Full executable red/green verification is pending WSL toolchain setup.
- No files deleted, no course documents modified, no commit/push performed by this worker.

- First executable compilation caught E0283 at tensor shape collect; explicit Vec<_> fixed it.
- `cargo +1.96.0 test --locked --test behavior`: 8 passed, 0 failed; complete output in ../week9-support/build-integration-pass3.log.
- Full unit suite at pass2: BP 4 passed and 3 failed; BP owner is repairing those separately. Do not interpret integration success as BP acceptance.
- `cargo +1.96.0 fmt --all` passed after installing rustfmt.
- `cargo +1.96.0 tree --locked -e features` saved ../week9-support/cargo-features.txt: CPU native enabled, no GPU/autodiff feature found.
- Real omeco binary topology is retained; private singleton labels are removed at earliest safe point, and saved search settings distinguish native optimizer work from executed/recounted work.

- ASIA cap representation correction: first sliced-factor runs are valid numerical checks but use a different cost graph. Final ASIA exact preserves original factors and appends one-hot caps; Promedus remains sliced. Cap regression RED missing-method failure is captured in ../week9-support/build-caps-red.log; final 9 integration tests passed in ../week9-support/build-caps-green.log. Earlier proposal to mask original tables was not adopted; cap topology is the final implementation.

- The first optimized build returned exit 0 in 11m47s, but the executable under /tmp was absent when the parent attempted its next run (exit 127); the later /tmp cache also lacked the earlier debug binary. The cause remains an environment warning, not a numerical result. No worker deletion was performed.
- Corrective action: interrupted a redundant temporary-target build, copied available partial artifacts without deleting them, and moved all subsequent build targets to persistent /home/keria_h/.cache/amat5315-week9-target. The recovery build uses CARGO_PROFILE_RELEASE_OPT_LEVEL=1, jobs=2, and copies the completed executable to persistent ../week9-support/release/infer with a SHA-256 receipt before exiting.
- Recovery shell script initially had CRLF line endings; converted that new script to LF. It then started successfully.
- A PowerShell-to-WSL inline wrapper expanded a shell exit-code variable before bash received it; this produced an integer-expression error after the interrupted test and started its next temporary build. Both stale wrapper/build processes were stopped. Subsequent build/test commands use literal LF shell script files with set -euo pipefail, avoiding inline shell-variable transport.

- Final persistent optimized build (opt-level=1, jobs=2) completed successfully. The executable was copied to persistent Windows storage before the process exited.
- Final `cargo +1.96.0 fmt --all --check` passed. Final `cargo +1.96.0 test --release --locked`: 8 BP unit tests and 10 integration tests passed, zero failures; see ../week9-support/build-final-release-tests.log. This includes real CLI rejection of partial ASIA queries without panic.
- Final locked release re-build and `cargo +1.96.0 install --locked --path . --root ../week9-support/install` both succeeded; see build-persistent-final-release.log and build-final-install.log.
- Final executable SHA-256: 15b337e8e51b9436f1a0b62b5896beda15315bccd02f0ac24d9ddbda9989bdfe. Initial persistent release, final release copy, and installed executable are checked independently from Windows.
- ASIA exact accepts all node queries in ID order only; subset requests return an explicit error. `evidence_quantity` distinguishes normalized-ASIA probability from the unnormalized MARKOV partition sum, and `evidence_partition_sum` preserves the numerical quantity while keeping the legacy evidence_mass field.

- Benchmark failure follow-up: the parent observed exit137 and high RSS in the original Promedus greedy run. Query102 plans had already been written; since the old CLI wrote query plans AFTER contraction, those files establish query102 completed. The precise failed query139 tree was not saved, so no claim is made that a later replan is byte-identical to the failed run.
- Isolated query102 steps823/824, with identical binary label geometry and all-one synthetic arrays, succeeded under a 1.5GiB address-space cap. Their measured peak virtual memory was436304/208660kB and RSS145176/169704kB. This does not support blaming those binary einsum operations or a mandatory union broadcast. Detailed receipt: ../week9-support/memory-probe-receipt.txt.
- A plan-only query139 replan (NO tensor contraction) produced work14006989794 and67108864 generated entries, i.e.512MiB for one f64 tensor. Saved under ../week9-support/promedus-q139-plan-only/; the observed failed phase cannot be pinpointed further without its missing original query plan.
- Corrective change: plan::retarget retains the scalar evidence tree's exact binary children and re-canonicalizes the requested open query indices. It does not clamp query values and does not reuse numeric results. For binary single queries, the generated-entry and arithmetic-work counts are at most twice the scalar tree's values. Query plans and independently recounted costs are now saved BEFORE execution, so future failures retain the actual attempted query plan.
- Retarget RED: missing helper produced E0425, captured in build-retarget-red.log. Final GREEN:8BP+12integration tests passed; includes topology/2x bounds, open marginals and T,L joint versus enumeration, and subnormal partition-scale checks. See build-bounded-tests.log. Exact tensor execution code was unchanged.
- New executable preserved separately as ../week9-support/release/infer-bounded; SHA2566eb67ebcca7717c1e1b4e4b040206bc078bf1c2e4fcdf51794b37ab82c4ee649. Previous15b... binary and failed-run evidence remain preserved. Full benchmark acceptance depends on the parent's new-namespace numerical run, not these synthetic/unit tests.

## Coordinator final acceptance

- The bounded executable above ran all final ASIA and Promedus jobs successfully
  (exit0), total wall time11.32s and peak RSS538788KiB. Installed and executed
  binaries have the same SHA-256.
- Promedus greedy/TreeSA log10 partition sums are -7.806746304582440 and
  -7.806746304582436. All three state-ordered query marginals pass the downloaded
  MAR comparison. BP remains unconverged at1000 sweeps, residual2.116264e-10,
  selected-query error0.38235292; these are retained outcomes.
- Final Rust tests:20 passed (8 BP +12 integration), zero failures. Python:4
  passed. Full independent audit:16 checks, zero failures, including actual
  query topology and dimension-bound recounts. All7 result figures were
  visually checked.
- `bash reproduce.sh` was also executed end-to-end into a new ignored directory;
  locked installation, all model runs and another full16-check audit passed.
  See `evidence/verification/reproduction-smoke.log`.
