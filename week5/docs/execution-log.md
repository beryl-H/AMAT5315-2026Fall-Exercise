# Week 5 continuation execution log

Task: continue the partially completed exercise in the user's existing Ubuntu
24.04 checkout, using the supplied `week5-learning-sheet.pdf`. All continuing
coursework stayed in `week5/`; Windows helper copies are in
`D:\Keria\Course\AMAT5315\week5-support`.

## Baseline and authorization

- Read the repository's AGENTS.md, existing Part 1 source/spec/tests, design TOML,
  lecture PDF, Git status/history and input provenance. The initial 13 Part 1
  tests passed. Existing commits, inputs and license were preserved.
- The terminal window was not available to native UI targeting. Work used
  `wsl.exe -d Ubuntu-24.04 --cd .../week5 --exec bash -lc ...` against the exact
  requested checkout. No commands were typed into an unrelated application.
- Normal source edits, installations, builds, plots and Git commits proceeded
  within the authorized exercise. No existing files/directories, Git branches,
  worktrees, environments or remote resources were deleted. No force push,
  hard reset or clean was performed.

## Files created

- Rust project: Cargo.toml, Cargo.lock, rust-toolchain.toml, build.rs; src/kernel.rs,
  lib.rs, main.rs, npy.rs, simulation.rs and schedule.rs.
- Independent tests: tests/test_seismic.py, test_derivatives.py,
  test_checkpoint.py and test_action_audit.py.
- Reproducibility: pyproject.toml, uv.lock, week5/.gitignore;
  scripts/env.sh, setup.sh, run_all.sh, download_inputs.py, audit_actions.py,
  plot_seismic.py, verify_evidence.py and write_inventory.py.
- Documentation: README.md, docs/seismic-spec.md, implementation-plan.md,
  execution-log.md, compiler-optimization.md and compiler-probe.rs.
- Evidence: all new forward/Born/adjoint/checkpoint/Marmousi JSON/PNG/action
  records and chronological build/test/diagnostic logs in README's exhaustive
  inventory. Numerical `.npy` files exist locally and remain Git-ignored.
- Course enzyme-setup skill installed in repository `.agents/skills/enzyme-setup/`.
  Existing MARMOUSI-LICENSE was retained and added to version control.
- Windows support: extracted lecture text, one inspected PDF page, source staging,
  result figure copies and verification/report copies. Local toolchains and
  environments live under ignored inputs/toolchain and .venv.

## Existing files modified

- scripts/compare_modes.py: changed the plotted title from force to energy
  derivative, since the graph reports dU/dr and physical force is -dU/dr.
- Existing Part 1 generated evidence in artifacts/ad/ was refreshed with the
  pinned environment. lj_ad.py, the original Part 1 tests, SPEC.md and the physical
  inputs were preserved.
- New source/docs evolved during implementation; Git commits retain the staged
  specification, failing-test and passing-implementation progression.
- A failed intermediate checkpoint log had an inaccurate passing filename. It
  was retained through a reversible rename to checkpoint-barrier-failure.txt;
  no log data was removed.

## Commands and validation

The complete numerical reproduction is `bash scripts/setup.sh`, then
`bash scripts/run_all.sh`. README lists every output beside its producer.
The material command families executed were:

```bash
source scripts/env.sh
rustup component add enzyme rustfmt --toolchain nightly-2026-09-05
inputs/toolchain/python-tools/bin/uv lock
inputs/toolchain/python-tools/bin/uv sync --locked
cargo fmt
cargo fmt --check
cargo build --release --locked
target/release/seismic --enzyme-smoke
python -m pytest -q
python scripts/run_ad.py
python scripts/compare_modes.py
python scripts/plot_graphs.py
python scripts/download_inputs.py
python scripts/plot_seismic.py
python scripts/verify_evidence.py
python scripts/write_inventory.py
bash -n scripts/env.sh scripts/setup.sh scripts/run_all.sh
git diff --check
git status --short
```

Forward/Born/full-history adjoint, four checkpoint budgets, Marmousi Born and
budget-5 adjoint use the exact CLI commands in README and scripts/run_all.sh.
The independent no-autodiff rustc build was required to fail. Test runs also
covered the exact two-step one-cell stencil, damping division, boundaries and CFL,
random-direction finite differences, independent random-weight transpose checks,
duplicate receivers, malformed data shape, complete saved-state accounting,
Treeverse work/budget/provenance and mutated action-audit failures.

The final test universe contains 13 original Part 1 tests and 24 new numerical,
CLI, source-edge and action-audit cases (37 total). Numerical acceptance is read
from arrays, not merely the solver's reported norms or planned counters.
Every PNG was opened for visual QA. All 21 action files were independently
replayed; all twelve audit violation counters were zero.

## Failures and corrective actions

| Observation | Action and final status |
| --- | --- |
| Windows restricted-token shell failed to enforce split writable roots | Used reviewed escalation for the named coursework; no destructive operation was requested. |
| Initial creation of .gitignore was auto-review rejected as a possible overwrite | Inspected the target, proved it absent, and used guarded creation. Resolved without deleting data. |
| WSL rg unavailable | Used available find and Python reads. |
| PDF text output hit a Windows GBK Unicode issue; fitz unavailable | Explicit UTF-8 output and installed pypdf extraction; bundled Poppler rendered the inspected page. |
| Staging docs parent directory absent | Created the new directory before the file patch. |
| New modes/executable/auditor absent | Saved expected red-test logs before adding implementations. |
| Direct recursive Treeverse work was excessive | Saved logs; replaced it with explicit owned Visit/Loop frames, preserving lecture order. A pure Rust probe with an exported schedule entry reproduces 8 vs 15 steps at O0/O3 through the ordinary driver, also with LTO. An earlier private-entry probe did not reproduce it. Exact pass and full production causation remain undetermined. |
| Far-tail source arithmetic produced NaN | Initial tail guard and regression fixed the reported path; later strengthened by the reviewed cases below. |
| Very large finite source frequency/amplitude overflowed, or a representable scaled tail was truncated | Three red regressions; multiply frequency by offset first, evaluate bounded wavelet before amplitude, and evaluate extreme scaled tails in log space. Three green regressions followed by a fresh complete workflow. |
| Recording-disabled rerun could retain old wavefield.npy | Reject before writing outputs; regression checks unchanged run.json. |
| Source correction caused tests to compare with older local Born/full arrays | Retained 4-failure/33-pass log; module-scoped fixture now generates both baseline arrays using the current executable, keeping strict bitwise equality. Tests no longer depend on ignored generated artifacts. |
| Depth-profile title clipped | Wrapped the title and regenerated/inspected the figure. Only the active source is marked in shot-0 snapshots. |
| Git whitespace check flagged pytest's raw diagnostic padding | Preserved log bytes; week5/.gitattributes disables whitespace checking only for raw evidence/*.txt. Source, Markdown and JSON remain checked. |

## Remaining limits

- The exported-entry probe demonstrates an optimization-related runtime defect;
  its precise pass and full connection to the earlier executor remain unconfirmed.
  The accepted explicit-frame solution passes numerical, budget and replay checks.
- JAX's optional CUDA backend is absent; CPU execution is verified.
- Byte counters measure saved restart states, not process RSS. Raw RTM is JᵀJm,
  so image equality with the perturbation is not an acceptance criterion.
- Week 6 account signup and payment verification require the user's account and
  the instructor's cost explanation; no cloud registration, payment method or
  GPU rental was changed during this exercise.
- A Chinese knowledge note is complete in this exercise's docs/. Two attempts to
  save it to ResearchOS were automatically rejected: the reviewer required fresh
  approval of the named note and did not accept the previously recorded ordinary
  knowledge-note authorization. No vault file or map was written. A concrete
  approval question was sent; course work continued independently.
- The combined commit/push request was automatically rejected before execution:
  the reviewer requires explicit approval of this payload to the configured
  public GitHub remote. Local preparation and commits can proceed; publishing
  remains pending that approval. No remote upload occurred in the rejected call.
