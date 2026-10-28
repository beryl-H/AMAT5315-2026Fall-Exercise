# Authorized MiniMax follow-up (2026-10-28)

After the original delivery at `a00bcc932c5807b41251341ca20739fca0c42671`, the user explicitly authorized switching this task to native WSL Pi with the existing independent MiniMax credentials. This directory publishes compact receipts and the advisory review. It does not change the original experiment or its acceptance decision.

## Provider and completion checks

- Native Ubuntu-24.04 WSL Pi used `minimax-direct / MiniMax-M3.1-Flash` at `https://api.minimax.cn/v1`, independently of the exhausted Volcengine quota.
- The minimal connection test returned exactly `MINIMAX_WSL_OK` with a nonempty final answer and `stopReason=stop`.
- The first broad review exhausted its output budget (`stopReason=length`) and was rejected, although the process exited zero. Its receipt remains visible.
- A single narrowed outer-product review completed with `stopReason=stop`. Its input was bound to the original commit and frozen source hashes. No automatic retry, tool access, alternate-provider fallback, or global Pi default change occurred.
- Credentials entered only the child environment through standard Pi `${TASK_PROVIDER_KEY}` configuration. No credentials, raw model reasoning, sessions, or provider configurations are published here. Reported model identity and token usage are client observations; zero configured cost fields mean unknown pricing, not free usage.

## Findings and coordinator adjudication

The review did not demonstrate a concrete implementation defect explaining the remaining slope failure. It identified finite-draw/interpolation sensitivity as a possibility, not an established cause. Model agreement does not turn the scientific failure into a pass.

The coordinator independently reconstructed all three outer-product curves and capacity crossings from the original 126 outer records (within 630 total records). Capacities for dimensions 16/32/64 remain 20.00949/34.93582/66.95666; the prescribed slope remains **0.871271425**, outside **[0.9,1.1]**. No seeds, draw counts, aggregation rule, fit interval, or threshold changed.

One caveat in the model answer was corrected: for unit keys, `off(K K^T) V = F - V` up to roundoff. The zero-diagonal product already is interference; directly computing ISR from it does not require restoring the own term. A deterministic two-memory, one-dimensional fixture confirms the identity. The original model answer is retained alongside this correction, rather than silently edited.

## Evidence map

| File | Meaning |
|---|---|
| `connection-receipt.json` | Accepted minimal native WSL Pi connection test |
| `truncated-review-receipt.json` | Preserved rejected broad review |
| `outer-review-receipt.json` | Accepted narrowed review, completion and usage metadata |
| `initial-packet-binding.json`, `narrow-packet-binding.json` | Frozen input/source bindings |
| `outer-review-answer.md` | Original advisory answer, including the caveat corrected above |
| `adjudication.json` | Coordinator's independent arithmetic checks and decision |
| `followup-verification.json` | Historical local follow-up verification at the original commit |
| `publication-verification.json` | Fresh publication checks and unchanged source/data boundary |

The checks in this follow-up are receipt/JSON validation, source binding, existing-record numerical reconstruction, an exact algebraic fixture, and credential/raw-file scanning. The original 95-test result remains part of the original delivery; this documentation-only follow-up adds no model implementation and does not rerun training. Added formal training, random draws, full sweeps, and model-source edits: **all zero**. Scientific acceptance remains partial because of the declared failed slope gate.

## Actual command form

The task-local controller loaded the existing credential into the child environment and invoked native WSL Pi with this form (placeholders contain no secret):

```bash
PI_CODING_AGENT_DIR=<task-local-config> pi --offline \
  --provider minimax-direct --model MiniMax-M3.1-Flash --thinking off \
  --mode json --no-tools --no-extensions --no-skills --no-prompt-templates \
  --no-themes --no-context-files --no-session --print -- <frozen-input>
```

The original local controllers were `minimax_followup.py`, `adjudicate_minimax_followup.py`, and `verify_minimax_followup.py` in the task support directory. Publication independently recalculated all 126 existing outer records, checked packet/receipt hashes and the unchanged raw-progress hash, checked the two-memory identity, verified frozen model sources and 876 old files, parsed all new JSON, and scanned the selected public text against the exact existing credential. This is publication verification, not another scientific experiment or a rerun of the original 95-test suite.
