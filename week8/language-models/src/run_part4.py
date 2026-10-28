"""Part 4 formal run: recall the 算 picture among M random memories with the
outer-product, ReLU, ReLU^2 and softmax (beta=8) stores.

Protocol (SPEC, frozen): picture value = 1024 flattened pixels (+1 ink / -1
background, norm 32); the other M-1 values are random +-1 pictures; all keys
random unit vectors in d=64 (independent per draw); the stored picture is
memory 0 and the query is its own key. 20 validation draws (seeds
531504..531523); the predeclared showcase draw is seed 531504 (never picked
for prettiness). ISR per cell: R = ||recall - w_own*picture|| / (|w_own|*32).
Reported: raw showcase R for all 16 cells and 20-draw medians (actual ratios,
every outcome retained).

Gates (mandatory): softmax R < 0.1 at every M (median AND showcase);
ReLU^2 at M=4096 R > 1; outer at M=1024 within 15% of 4; ReLU at M=1024
within 15% of 2.83. ReLU^2 at M=1024 ~ 0.60 is a descriptive reference only.
Figure: rows M, columns stores; dot area = max(0, (pixel+1)/2) of the
own-normalized recall (recall / w_own); cell marked red when showcase R < 1.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.part4_glyph import render_mask  # noqa: E402

D = 64
BETA = 8.0
M_VALUES = [64, 256, 1024, 4096]
STORES = ["outer", "relu", "relu2", "softmax"]
SEEDS = list(range(531504, 531524))
SHOWCASE_SEED = 531504


def softmax_weights(K, x, beta):
    s = beta * (K @ x)
    s = s - s.max()  # stable shift
    w = np.exp(s)
    return w / w.sum()


def recall_part4(K, V, x, store, beta):
    """Returns (recall vector, own weight w_own) for the query x = stored key."""
    if store == "softmax":
        w = softmax_weights(K, x, beta)
        return w @ V, float(w[0])
    if store == "outer":
        w = K @ x  # own coeff k.k = 1
    elif store == "relu":
        w = np.maximum(K @ x, 0.0)
    elif store == "relu2":
        w = np.maximum(K @ x, 0.0) ** 2
    else:
        raise ValueError(store)
    return w @ V, 1.0


def picture_R(recall, w_own, picture):
    return float(
        np.linalg.norm(recall - w_own * picture)
        / (abs(w_own) * np.linalg.norm(picture))
    )


def picture_ISR_gate_eval(medians, showcase):
    """Mandatory gates from the sheet; reference values are descriptive only."""
    gates = {
        "outer_1024_within_15pct_of_4": abs(medians[("outer", 1024)] - 4.0) / 4.0 <= 0.15,
        "relu_1024_within_15pct_of_2.83": abs(medians[("relu", 1024)] - 2.83) / 2.83 <= 0.15,
        "relu2_4096_gt_1": medians[("relu2", 4096)] > 1.0,
        "softmax_allM_lt_0.1": all(medians[("softmax", M)] < 0.1 for M in M_VALUES)
        and all(showcase[("softmax", M)] < 0.1 for M in M_VALUES),
    }
    return gates


def draw_case(mask_flat, M, seed, rng_offset=0):
    rng = np.random.default_rng(seed + rng_offset)
    K = rng.normal(size=(M, D))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.choice([-1.0, 1.0], size=(M, 1024))
    V[0] = mask_flat  # memory 0 is the stored picture
    return K, V


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="formal-01")
    args = ap.parse_args(argv)

    art_root = os.path.join(HERE, "artifacts", "part4")
    attempt_dir = os.path.join(art_root, args.attempt)
    if os.path.exists(attempt_dir):
        print(f"attempt exists: {attempt_dir}")
        return 2
    os.makedirs(attempt_dir)  # atomic exclusive attempt

    font = None
    for cand in ["/mnt/c/Windows/Fonts/simhei.ttf", "/mnt/c/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(cand):
            font = cand
            break
    mask, prov = render_mask(font, out_dir=os.path.join(HERE, "evidence"))
    mask_flat = mask.ravel().astype(np.float64)
    assert np.linalg.norm(mask_flat) == 32.0

    settings = {
        "attempt": args.attempt, "M_values": M_VALUES, "stores": STORES,
        "beta": BETA, "d": D, "seeds": SEEDS, "showcase_seed": SHOWCASE_SEED,
        "glyph": {k: prov[k] for k in ("character", "font_path", "font_sha256",
                                       "rasterization", "ink_fraction")},
        "source_hashes": {f: hashlib.sha256(open(os.path.join(HERE, "src", f), "rb").read()).hexdigest()
                          for f in ["part4_glyph.py", "run_part4.py"]},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(attempt_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    raw = {}       # (store, M, seed) -> R
    for seed in SEEDS:
        for M in M_VALUES:
            K, V = draw_case(mask_flat, M, seed)
            x = K[0]
            for store in STORES:
                r, w_own = recall_part4(K, V, x, store, BETA)
                raw[(store, M, seed)] = picture_R(r, w_own, mask_flat)

    medians = {(s, M): float(np.median([raw[(s, M, sd)] for sd in SEEDS]))
               for s in STORES for M in M_VALUES}
    showcase = {(s, M): raw[(s, M, SHOWCASE_SEED)] for s in STORES for M in M_VALUES}
    gates = picture_ISR_gate_eval(medians, showcase)

    results = {
        "settings": settings,
        "showcase_raw_ISR": {f"{s}|M={M}": showcase[(s, M)] for s in STORES for M in M_VALUES},
        "medians_20draws": {f"{s}|M={M}": medians[(s, M)] for s in STORES for M in M_VALUES},
        "all_raw_ISR": {f"{s}|M={M}|seed={sd}": raw[(s, M, sd)]
                        for s in STORES for M in M_VALUES for sd in SEEDS},
        "gates": gates,
        "gates_all_pass": all(gates.values()),
        "descriptive_reference": {"relu2_M1024": medians[("relu2", 1024)],
                                  "sheet_reference": 0.60},
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(attempt_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(HERE, "evidence", "part4-results.json"), "w") as f:
        json.dump({"attempt": args.attempt, **results}, f, indent=2)

    # figure: rows M, columns stores; showcase draw, own-normalized recall
    fig, axes = plt.subplots(len(M_VALUES), len(STORES), figsize=(13, 11))
    for i, M in enumerate(M_VALUES):
        K, V = draw_case(mask_flat, M, SHOWCASE_SEED)
        x = K[0]
        for j, store in enumerate(STORES):
            r, w_own = recall_part4(K, V, x, store, BETA)
            norm_recall = r / w_own  # true own-normalized recall
            ax = axes[i, j]
            area = np.clip(np.maximum(0.0, (norm_recall + 1.0) / 2.0), 0.0, 1.0)
            ii, jj = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
            ax.scatter(ii.ravel(), jj.ravel(), s=area.ravel() * 12,
                       c="k", marker="s", linewidths=0)
            readable = showcase[(store, M)] < 1.0
            ax.set_title(f"{store} M={M}\nR={showcase[(store, M)]:.3f}",
                         color=("red" if readable else "k"), fontsize=9)
            ax.set_xlim(-1, 32); ax.set_ylim(32, -1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("red" if readable else "0.7")
    from matplotlib.font_manager import FontProperties

    fig.suptitle(f"算 recalled among random memories (showcase seed {SHOWCASE_SEED}; "
                 "dot area = max(0,(pixel+1)/2) of own-normalized recall; red cell/frame = R<1)",
                 fontproperties=FontProperties(fname=font, size=13))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "evidence", "part4-recall.png"), dpi=140)

    print(json.dumps({"medians": results["medians_20draws"],
                      "gates": gates,
                      "all_pass": results["gates_all_pass"]}, indent=1))
    return 0 if results["gates_all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
