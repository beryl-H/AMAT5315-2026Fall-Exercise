#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
mkdir -p inputs/toolchain
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o inputs/toolchain/rustup-init.sh
  RUSTUP_INIT_SKIP_PATH_CHECK=yes sh inputs/toolchain/rustup-init.sh -y --no-modify-path \
    --profile minimal --default-toolchain nightly-2026-09-05 --component enzyme
fi
rustup component add enzyme rustfmt --toolchain nightly-2026-09-05
if [[ ! -x inputs/toolchain/python-tools/bin/python ]]; then
  python3 -m venv inputs/toolchain/python-tools
fi
inputs/toolchain/python-tools/bin/python -m pip install --disable-pip-version-check uv==0.12.23
inputs/toolchain/python-tools/bin/uv sync --locked
python scripts/download_inputs.py
