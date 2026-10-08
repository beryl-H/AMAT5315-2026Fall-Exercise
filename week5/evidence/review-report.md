# Independent source review and follow-up

A read-only reviewer inspected the actual kernel, validated model/CLI, NPY I/O,
JVP/VJP composition, full-history reverse, explicit-frame Treeverse and independent
tests/action auditor. It did not alter files or rerun already successful suites.

The required reflector/Marmousi paths had no blocking findings: the restart state
contains both pressure fields; `grad` borrows an actual saved input; replay calls
primal only; Enzyme activity annotations match the intended state/velocity
directions; velocity shadow accumulation, duplicate receiver injection and output
buffer ownership are consistent with the transpose test.

Concrete review findings and parent corrections:

1. Very remote finite Ricker peak times produced NaN via polynomial overflow
   times an underflowed exponential. The parent recorded red/green regressions.
2. A recording-disabled rerun could leave a stale wavefield array. The parent
   rejects that reuse before modifying outputs and tests preservation of run.json.
3. Finite source_frequency=1e308 at t=t0 formed infinity times zero; finite
   source_amplitude=1e308 at theta=2 overflowed before the exponential. The parent
   confirmed both with failed CLI fixtures and added a third scaled-tail fixture
   at theta=30. Corrections multiply frequency by offset before PI, form the
   bounded wavelet before amplitude, and use log-space scaling for extreme tails.
   All three cases passed before the fresh full workflow.

The reviewer initially attributed the direct recursive scheduler failure to LLVM
signature rewriting, before sufficient runtime evidence. An initial private-entry
probe was correct at O0/O3. The reviewer then supplied the exact exported schedule
entry, and the parent reproduced O0=8 calls / correct bases versus O3=15 calls /
all-zero bases using the ordinary rustc driver, also with LTO. This demonstrates
an optimization-related runtime defect in that probe. The precise pass and full
production causal link remain unconfirmed. User-facing updates distinguish these
stages. The explicit-frame solution is accepted on actual work, storage, numerical
and action-audit evidence.

The final evidence report separately reads actual arrays and recomputes every
action file's live-slot provenance, counters and reverse coverage. The runtime
executor additionally asserts actual map size equals planned saved_states after
each operation. Final test totals and numerical outcomes are in final-tests.txt
and artifacts/verification.json.
