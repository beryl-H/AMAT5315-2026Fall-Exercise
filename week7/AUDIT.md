# Week 7 task record

Date: 2026-10-19, Asia/Shanghai. Outcome: all three selected-local-PDF parts
implemented and numerically verified in order. Student demonstration remains
the student's responsibility. Authoritative source and exact acceptance are
in SOURCE.md and SPEC.md. Work used the existing exercise repository, baseline
`af4bbe8622ac8e554a3961f4d7a172730802f6ca`, initially clean, with a new local
branch `week7-inverse-priors-20261019`. Codex implemented and independently
recomputed the results in this session; no external model/provider was called.

## Files

Created under week7/: SPEC.md, SOURCE.md, PLAN.md, AUDIT.md, README.md,
pyproject.toml, uv.lock, .gitignore; sparse.py, inpainting.py, repaint.py,
prepare_inputs.py, study.py, inpaint, inpaint.design.toml, figures.py; four
test modules; eight evidence PNGs and numerical provenance/audit JSONs.
Scientific raw NPZ/JSON files and reconstructed official inputs are retained
under ignored artifacts/ and inputs/. No old-week files were modified.

The Windows task support folder contains the extracted PDF text, local/online
source hash record, retained incompatible online download, implementation
staging, command logs, independent audit scripts/results, source-page render,
evidence copies, verified delivery export and learning guide. It contains no
credentials. No files, branches, environments or resources were deleted.

## Commands and actual checks

The working directory for Python commands was the repository's week7/.

| Command/action | Actual result and coverage |
| --- | --- |
| `wsl -d Ubuntu-24.04 ...`, read AGENTS.md / git status / checkpoint | Existing repo baseline verified; clean before work |
| `uv python install 3.11`; `uv sync --python 3.11` | Installed independent Python 3.11.17 environment; pinned scientific dependencies |
| Initial `pytest tests/test_sparse.py` | 4 expected assertion failures before implementation |
| `uv run pytest -q tests/test_sparse.py` | 4 passed; basis orientation, signed L1, empty observations and RNG stream |
| `uv run python sparse.py --size 128 --sparsity 5 --repeats 24 --seed 7101 --out artifacts` | 120 real trials; success counts 2/20/24/24/24 |
| Independent `verify-part1.py` | All random draws/raw errors and 120 alternative-LP objective checks passed; perturbed-hidden-value control rejected |
| Initial `pytest tests/test_inpainting.py` | 6 expected failures before implementation |
| `uv run python prepare_inputs.py` | Two original MNIST archive digests and all 20 exact row identities checked |
| `uv run python study.py --input inputs/images.npz --out artifacts` | DCT means .1895889244/.6017061911; observed error 0 |
| `uv run ./inpaint --method dct --input inputs/case.npz --k 20 --iterations 100 --out artifacts/case` | True standalone execution; exactly matches batch case |
| Independent `verify-part2.py` | Explicit dense DCT reconstruction, raw MSE, 400 hidden pixels and standalone equality passed; wrong denominator rejected |
| Initial `pytest tests/test_repaint.py` | 4 expected failures before implementation |
| `predict-compat.py` in each old/new environment | Same checkpoint/input; maximum prediction difference 3.70e-6 |
| `JAX_PLATFORMS=cpu uv run python study.py --input inputs/images.npz --checkpoint ../week6/checkpoints/mnist.npz --out artifacts` | 4 seeds x 20 images; 298 reverse attempts each; 38.60 s full process; exact checkpoint preserved |
| Independent `verify-part3.py` | All statistics, stored seed arrays, all 298 PRNG/algebra transitions and all recorded epsilon predictions passed |
| `uv run ./inpaint --method diffusion --input inputs/case.npz --checkpoint ../week6/checkpoints/mnist.npz --steps 100 --repeats 3 --seed 7201 --out artifacts/case-diffusion` | Actual standalone inference; observed error 0; batch shape is 1, so draws differ from full-study batch |
| `uv run python figures.py --part 1`, `--part 2`, `--part 3` | Eight figures generated and visually inspected; raw arrays unchanged |
| Final pytest / compile / links / export checks | Exact final results are in evidence/final-checks.json and Windows delivery receipt |

## Failures and corrections

- Sandbox WSL read was denied; authorized escalated execution accessed the
  existing distribution successfully. No repair/reset operation was performed.
- Initial PDF stdout extraction hit a Windows GBK UnicodeEncodeError; explicit
  UTF-8 output corrected extraction. The PDF was not changed.
- The web reader could not open the course site. Direct HTTPS retrieval worked
  and confirmed that both online lecture and image archive belong to a different
  version. Both were preserved; exact local-PDF MNIST rows were reconstructed
  from verified original data, and both fixed DCT mean thresholds matched.
- Read-only discovery initially tried nonexistent old module paths; the actual
  source is week6/src/diffusion/. No source was changed by discovery.
- The first independent Part 1 audit accidentally put a random choice on the
  left side of an assignment whose right side drew normals first. Python's
  evaluation order changed the verifier stream. Separate support/value draws
  corrected the verifier. Production stream, seed and experiment were unchanged.
- Initial fixed-inpainting plotting had uneven spacing because long text was
  attached to one narrow subplot. A separate note panel corrected layout.
- Initial JAX tests auto-detected a GPU without CUDA support and warned before
  using CPU. Subsequent checks explicitly selected `JAX_PLATFORMS=cpu`.
- Expected red tests were followed by passing production tests; they are
  development evidence, not unresolved implementation failures.
- The local command wrapper used the same JSON filename as the Part 3 audit
  report in the support folder. The tracked scientific audit was intact.
  Command receipts now use a `-command.json` suffix; the support audit was
  restored byte-for-byte from the tracked report. No experiment was rerun.

## Boundaries

No retraining, cloud resource action, secret access, destructive operation,
merge or remote publication is included in this local delivery. All data and
figures reflect fixed seeds and exact prescribed inputs. Diffusion improvement
is not an acceptance threshold: the large-hole result is worse and retained.
Four draws do not calibrate uncertainty or establish exact posterior samples.
The online/local lecture mismatch remains an explicit version warning.

Model source compatibility was measured on two fixed inputs; it is not a global
cross-version numerical equivalence proof. Full trace re-evaluation verifies
the actual first-case noise predictions used in this run. Automated tests and
plots do not substitute for the student's Week 8 explanation.
