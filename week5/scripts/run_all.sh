#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
mkdir -p evidence
python scripts/download_inputs.py | tee evidence/input-fingerprints.txt
cargo fmt --check 2>&1 | tee evidence/format-check.txt
cargo build --release --locked 2>&1 | tee evidence/build-final.txt
target/release/seismic --enzyme-smoke | tee evidence/enzyme-smoke.txt
if rustc --edition=2024 --crate-type=staticlib -C panic=abort src/kernel.rs \
    -o inputs/enzyme-disabled-control.a > evidence/enzyme-disabled-control.txt 2>&1; then
  printf 'ERROR: Enzyme compiled without the required autodiff flag\n' >&2
  exit 1
fi
python -m pytest -q 2>&1 | tee evidence/final-tests.txt
python scripts/run_ad.py | tee evidence/part1-derivatives.txt
python scripts/compare_modes.py | tee evidence/part1-modes.txt
python scripts/plot_graphs.py | tee evidence/part1-graphs.txt
target/release/seismic --experiment inputs/reflector.json --mode forward \
  --every 3 --out artifacts/forward | tee evidence/forward-run.txt
target/release/seismic --experiment inputs/reflector.json --mode born \
  --out artifacts/born | tee evidence/born-run.txt
target/release/seismic --experiment inputs/reflector.json --mode adjoint \
  --data artifacts/born/born_data.npy --every 3 --storage full \
  --out artifacts/adjoint | tee evidence/adjoint-run.txt
for budget in 1 3 5 10; do
  target/release/seismic --experiment inputs/reflector.json --mode adjoint \
    --data artifacts/born/born_data.npy --storage treeverse --checkpoints "$budget" \
    --out "artifacts/checkpoint-$budget" | tee "evidence/checkpoint-$budget-run.txt"
done
target/release/seismic --experiment inputs/marmousi.json --mode born \
  --out artifacts/marmousi-born | tee evidence/marmousi-born-run.txt
target/release/seismic --experiment inputs/marmousi.json --mode adjoint \
  --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 \
  --out artifacts/marmousi-image | tee evidence/marmousi-image-run.txt
python scripts/plot_seismic.py | tee evidence/plots.txt
python scripts/verify_evidence.py | tee evidence/verification.txt
python scripts/write_inventory.py
