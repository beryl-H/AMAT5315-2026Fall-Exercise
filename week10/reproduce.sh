#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cargo test --locked
cargo install --locked --path . --force
python verify_and_plot.py --bin "$(command -v thermo)" --phase small
python study.py --bin "$(command -v thermo)" --phase all --workers 2
python verify_and_plot.py --bin "$(command -v thermo)" --phase all
# Existing complete run directories are read back; incomplete ones cause an error.
# No existing artifacts are deleted. For a completely fresh computation, copy source
# to a new directory and execute this script there.
