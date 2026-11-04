#!/usr/bin/env bash
# Preserve checked-in evidence: every reproduction gets a new directory.
set -euo pipefail
export CARGO_PROFILE_RELEASE_OPT_LEVEL=1
export CARGO_BUILD_JOBS=2
cd "$(dirname "$0")"
cargo +1.96.0 install --locked --path .
infercmd="${CARGO_HOME:-$HOME/.cargo}/bin/infer"
runroot="reproductions/$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$runroot/promedus"
cp asia.uai "$runroot/asia.uai"
cp promedus/Promedus_14.uai promedus/Promedus_14.uai.evid promedus/Promedus_14.uai.PR promedus/Promedus_14.uai.MAR "$runroot/promedus/"
for label in prior AD ADX; do
  "$infercmd" --model asia.uai --evidence "$label" --out "$runroot/artifacts/$label"
done
"$infercmd" --model asia.uai --evidence AD --order random --seed 42 --out "$runroot/runs/random"
"$infercmd" --model asia.uai --evidence AD --order greedy --out "$runroot/runs/greedy"
"$infercmd" --model asia.uai --evidence AD --analyze "$runroot/runs/random/order.json" --out "$runroot/runs/analysis"
"$infercmd" --model asia.uai --evidence AD --plan "$runroot/runs/greedy/order.json" --out "$runroot/runs/replay"
for label in prior AD ADX; do
  "$infercmd" --model asia.uai --evidence "$label" --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out "$runroot/artifacts/bp-$label"
done
"$infercmd" --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --order greedy --out "$runroot/runs/promedus-greedy"
"$infercmd" --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --order treesa --seed 42 --trials 4 --iterations 50 --out "$runroot/runs/promedus-treesa"
"$infercmd" --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out "$runroot/artifacts/bp-promedus"
python verify_and_plot.py --root "$runroot"
printf 'Reproduction evidence: %s\n' "$runroot"
