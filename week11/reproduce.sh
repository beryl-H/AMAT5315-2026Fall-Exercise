#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

phase=${1:-all}
case "$phase" in
    all|pred|gadgets|magnet|queens|plots) ;;
    *) echo "Usage: bash reproduce.sh {all|pred|gadgets|magnet|queens|plots}" >&2; exit 2 ;;
esac
if [[ "$phase" == all || "$phase" == pred ]]; then
    test "$(pred --version)" = 'pred 0.6.0'
    mkdir -p runs/pred
    pred path Factoring SpinGlass | tee runs/pred/path.txt
    for target in 15 6 7; do
        width=2
        if [[ "$target" == 15 ]]; then width=3; fi
        stem="runs/pred/factor_${target}"
        pred create Factoring --target "$target" --m 2 --n "$width" -o "$stem.json"
        pred reduce "$stem.json" --to SpinGlass -o "$stem.bundle.json"
        pred solve "$stem.bundle.json" --json 2> "$stem.solve.stderr" | tee "$stem.solve.json"
        bits=$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["solution"])))' "$stem.solve.json")
        test -n "$bits"
        pred evaluate "$stem.json" --config "$bits" --json | tee "$stem.evaluate.json"
    done
fi
if [[ "$phase" == all || "$phase" == gadgets ]]; then
    tnsample --mode gadgets --seed 2026 --out runs/gadgets
fi
