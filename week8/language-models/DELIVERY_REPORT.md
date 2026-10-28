# Delivery audit — revised Week8

Execution is complete; scientific acceptance remains partial because the outer capacity slope fails one declared gate. See evidence/ACCEPTANCE.json for machine-readable status. No inference is presented as a demonstrated cause of that failure.

## Files created/modified

All work is under week8/language-models: SPEC/PLAN/README/TASK_STATUS,INPUTS,uv lock+project,src models/drivers/tests,portable tools,evidence JSON/PNGs/test logs,LEARNING_GUIDE_ZH.md andSAMPLES.md. Existing new-task scaffold/docs/modules were updated in place; original intermediate figures and failed attempts retained. Every one of876 baseline files outside the new task is unchanged. No user file or directory was deleted.

## Commands and checks actually executed

- Native WSL Pi: `pi --provider volcengine-plan --model glm-5-3-flash --mode json --session ... -p @pi-task-0N.md` (phases1–7). Prompts/raw sessions preserved locally; raw model reasoning is not published.
- Independent pytest from this task directory: `.venv/bin/python -m pytest tests -x -q`:95 passed. Earlier wrong-working-directory invocation failed on data/tok512.model; corrected cwd, no assertion changes.
- Part1 independent safetensor/vocabulary/LOO recomputation; split/SP token accounting; exact train/val stream checksums and all3 decode exceptions.
- Part2 sole exclusive `src/run_part2.py --attempt formal-02`: six losses match independent oracle within1e-12;P matrix and250000-token correct/incorrect sample arrays match bitwise; explicit512³ enumeration within1e-12; chi2 all expected>=5.
- Memory sole full sweep:630 unique records, all keys queried,3 fixed draws;943.80s,peak~797MiB; independent median/crossing/slope checks and18 direct small-case sums.7/8 gates pass; outer slope fails.
- Pi-authored frozen neural driver: 5/10-step coordinator smoke proves positive gradients, parameter/Adam updates and bitwise state reload; then one formal4000-step MLP and one6000-step Transformer. Actual elapsed247.63s/504.75s including scoring. No restart, extra epochs, seed change or hyperparameter sweep.
- Initial/mid/final NPZ checks verify0/2000/4000 and0/3000/6000 steps, finitefloat32 parameters/m/v andPRNG state. Source hashes unchanged throughout jobs.
- `tools/audit_neural.py`: full1,078,995-target freshf64-logsumexp accumulation from saved weights, both scores match producer to<2e-6 nats/token; all64 position curves;8 causal pairs;4x100000 samples plus greedy and bad-top-p controls;60-token fixed-seed stories;2x128x128-token entropy tests usingsequence-level SE.
- Part4 independent320-cell ISR recomputation, maximum absolute difference3.55e-15. Exact same fixed showcase reconstructed for corrected figure; no20-draw experiment rerun.
- Scaling:243 L-BFGS-B starts+400 single-attempt fixed bootstrap fits; all643 optimizer statuses success, every end parameter/resample recorded; independent objective,stationarity,6ND=C and percentile-interval checks.
- Visual review: all12 final figures checked; corrected angle aspect, label overlap, glyph transpose, dot-area upper clipping and optimal-allocation x-axis. Earlier figures retained.

## Failures and corrections

See evidence/failures/failure-register.json andpart2-formal-01-invalidated.json. Genuine remaining failure: outer slope.87127 not in[.9,1.1]. Independent small-case sums agree; sampling variability remains a hypothesis. The original3 seeds and gate are retained. Part2 multiple-writer custody and pooled-bin issues were fixed before cited formal-02. NN implementation defects were caught and fixed before any formal neural training. Pi quota error was preserved and never converted to a successful model result based on exit0.

## Warnings and limits

Pinned Matplotlib/PyParsing emits deprecation warnings. The old glyph-preview title warns about a missing CJK glyph; the actual mask is correct and final preview usesASCIIU+7B97. A degenerate deterministic entropy unit fixture has zero standard error and emits a divide-by-zero warning; formal entropy SEs are nonzero and saved. min-p p=.00150984 is close to, but above, the fixed.001 gate; retained without resampling. Scaling optimizer termination usesrelative function reduction, with best projectedgradient infinity2.25e-6, not declared1e-12gtol reached. Bootstrap intervals reflect the chosen published digitized sample and filter, not all possible language-model regimes. Passing code tests/negative controls does not imply every scientific criterion passes. Optional attention maps show read locations, not causal explanations; the optional hidden-unit audit was not required/executed.

## Local/public custody

Public: source, tests, prompts/protocol summaries, compact numerical evidence and figures. Local only: corpus/downloads,raw count arrays,all initial/mid/final optimizer checkpoints,failed-attempt directories andraw Pi sessions. Windows delivery mirror/manifest and Git publication receipt are produced after final checks. The updated worksheet explicitly requests pushing source and evidence; no course submission,message or PR review is made.

Publication formatting check: two imported Windows JSON receipts used CRLF; git diff --check flagged trailingCR. Public copies were normalized toLF; original receipts remain in the support snapshot. No numerical/model changes.

## Subsequent user-authorized MiniMax review

Added compact connection/review receipts, the original advisory answer, coordinator adjudication and an evidence index under evidence/minimax-followup-20261028. README and TASK_STATUS now link the follow-up. The first broad review was truncated and rejected; the narrowed review completed. Existing-record reconstruction and a two-memory algebraic fixture were checked independently. No training, random draws, code changes or scientific acceptance changes occurred. See the follow-up index for commands, evidence limits and the corrected own-term caveat.
