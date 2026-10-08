#!/usr/bin/env bash
# Source from week5. Toolchain remains isolated from system Rust.
export CARGO_HOME="$PWD/inputs/toolchain/cargo"
export RUSTUP_HOME="$PWD/inputs/toolchain/rustup"
export PATH="$CARGO_HOME/bin:/home/keria_h/.venvs/amat5315/bin:$PATH"
