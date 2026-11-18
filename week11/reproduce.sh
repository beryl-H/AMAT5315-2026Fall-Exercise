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
if [[ "$phase" == all || "$phase" == magnet ]]; then
    tnsample --mode count --l 10 --p-af 0 --beta 2.0 --seed 2026 --out runs/count_p0
    bundle=runs/pred/factor_15.bundle.json
    tnsample --mode count --input "$bundle" --beta 2.0 --seed 2026 --out runs/count_f15
    tnsample --mode sample --input "$bundle" --beta 2.0 --samples 2000 --seed 2026 --out runs/sample_f15
    state=$(python verify.py ground-config)
    pred extract "$bundle" --config "$state" --json | tee runs/sample_f15/extract.json
    for k in 2 3 4 5 6 7 8; do
        target=$(( ((1 << k) - 1) * ((1 << k) - 1) ))
        stem="runs/pred/scale_$k"
        pred create Factoring --target "$target" --m "$k" --n "$k" -o "$stem.json"
        pred reduce "$stem.json" --to SpinGlass -o "$stem.bundle.json"
        for order in greedy treesa; do
            tnsample --mode plan --order "$order" --input "$stem.bundle.json" \
                --seed 2026 --out "runs/plan_k${k}_$order"
        done
    done
fi
