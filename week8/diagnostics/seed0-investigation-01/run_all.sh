#!/usr/bin/env bash
# Run the full seed0-investigation-01 diagnostic suite in order.
# Read-only w.r.t. frozen sources and formal artifacts; writes only under
# diagnostics/seed0-investigation-01/results/.
set -u
cd "$(dirname "$0")/../.."   # week8/
PY=.venv/bin/python
export JAX_PLATFORMS=cpu
D=diagnostics/seed0-investigation-01
rc=0
for s in diag01_adamw_crosscheck diag02_grad_finite_diff diag03_split_determinism \
         diag04_init_streams diag05_seed0_trajectory diag06_checkpoint_rescore \
         diag07_negative_controls; do
  echo "=== $s ==="
  $PY "$D/$s.py" || rc=1
done
echo "SUITE_EXIT=$rc"
exit $rc
