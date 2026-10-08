# Recursive schedule diagnosis: evidence and limits

The first direct recursive Treeverse executor produced 28,680 primal calls per
shot for every reflector budget. Budget-1 was correct; budgets 3/5/10 failed the
independent work-count tests, although the final image and saved-state peaks were
correct. The N=6, delta=2 case likewise produced 15 calls rather than 8.

The failure and intermediate barrier attempts remain in
`evidence/checkpoint-recompute-failure.txt` and
`evidence/checkpoint-barrier-failure.txt`. Adding function-entry diagnostics
yielded the correct nonzero replay bases; `checkpoint-debug-trace.txt` records
that observation. This is diagnostic evidence, not a causal proof.

A reviewer initially inferred an LLVM optimization defect from disassembly and
optimized LLVM IR: a replay-base scalar argument disappeared from a recursive
signature, while literal zero appeared before a `black_box`. That interpretation
was reported too confidently. Signature rewriting or zero initialization alone
does not establish an incorrect runtime result.

The first executable pure Rust probe called `visit` directly from `main`. This
private-entry version is retained in `compiler-private-probe.rs`; it contains no
Enzyme, FFI or unsafe code. Both O0 and O3 gave the correct eight calls. Its output
is retained in `evidence/compiler-optimization-probe.txt`.

The reviewer then supplied the exact original public `schedule` entry:
`#[unsafe(no_mangle)] pub fn schedule(e,d,t) -> Vec<(u8,usize)>`, calling `visit`,
with `main` calling `schedule`. This version is in `compiler-probe.rs`. Apart from
the required unsafe export-name attribute, it has no unsafe operations, Enzyme or
FFI. The parent built and executed it with the ordinary rustc driver:

```bash
source scripts/env.sh
for opt in 0 3; do
  rustc --edition=2024 --crate-name checkpoint_probe --crate-type bin \
    -C opt-level="$opt" -C panic=abort docs/compiler-probe.rs \
    -o "inputs/compiler-probe-public-o$opt"
  "inputs/compiler-probe-public-o$opt" 6
done
rustc --edition=2024 --crate-name checkpoint_probe --crate-type bin \
  -C opt-level=3 -C lto=fat -C panic=abort docs/compiler-probe.rs \
  -o inputs/compiler-probe-public-lto
inputs/compiler-probe-public-lto 6
```

The actual outputs, in `evidence/compiler-public-probe.txt`, are:

```text
O0:           calls=8;  restore bases=[0, 3, 3, 0, 1]
O3:           calls=15; restore bases=[0, 0, 0, 0, 0]
O3 + fat LTO: calls=15; restore bases=[0, 0, 0, 0, 0]
```

This is an optimization-related runtime counterexample under the pinned compiler,
including its ordinary final-link pipeline. The earlier private-entry probe alone
was insufficient to establish it. The precise compiler pass and the complete
causal connection to the original production executor have not been established;
the report distinguishes demonstrated behavior from inference about its cause.

The production solution separates action construction from execution. Owned
`Visit {b,s,e,d,t}` and `Loop {b,s,e,d,t}` frames evaluate the lecture's recursive
construction in depth-first order. The loop continuation is pushed before the
child so a LIFO stack visits the child first. All local parameters are preserved
explicitly. Saved wavefields are owned by the executor's slot map and are distinct
from these tiny scheduling frames, replay workspace, adjoints and derivative
scratch.

The solution is supported by runtime invariants, independent action audits,
exact N=6/budget-2 and reflector work counts, the delta+1 saved-state limit,
descending unique reverse steps, and bitwise identical images. These are the
acceptance claims, regardless of the undetermined precise cause of the earlier method.
