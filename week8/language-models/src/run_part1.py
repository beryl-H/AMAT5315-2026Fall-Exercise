"""Part 1 formal inference: relations as directions in potion-base-8M.

Protocol (SPEC): unit-normalize rows; per-pair LOO cosine against the mean
offset of the other 8 pairs; relation mean = mean of 9; control = rows shuffled
by seed 1730 (mandatory gate |mean| <= 0.15); 20 further shuffles (seeds
1731..1750) reported descriptively (reference bound ±0.09, not a gate).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.part1_embeddings import (  # noqa: E402
    load_embeddings,
    load_word_ids,
    pair_loo_cosines,
    relation_mean,
    shuffle_rows,
    unit_rows,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELATIONS = {
    "country_capital": [
        ("iran", "tehran"), ("turkey", "ankara"), ("russia", "moscow"),
        ("china", "beijing"), ("france", "paris"), ("sweden", "stockholm"),
        ("japan", "tokyo"), ("peru", "lima"), ("kenya", "nairobi"),
    ],
    "male_female": [
        ("king", "queen"), ("man", "woman"), ("he", "she"),
        ("brother", "sister"), ("father", "mother"), ("son", "daughter"),
        ("uncle", "aunt"), ("boy", "girl"), ("husband", "wife"),
    ],
    "adj_comparative": [
        ("big", "bigger"), ("small", "smaller"), ("fast", "faster"),
        ("dark", "darker"), ("cold", "colder"), ("old", "older"),
        ("strong", "stronger"), ("long", "longer"), ("tall", "taller"),
    ],
}


def relation_pair_cosines(U, ids, pairs):
    idx = [(ids[a], ids[b]) for a, b in pairs]
    return pair_loo_cosines(U, idx)


def main():
    E = load_embeddings(os.path.join(HERE, "data", "model.safetensors")).astype(np.float64)
    assert E.shape == (29528, 256)
    words = sorted({w for ps in RELATIONS.values() for p in ps for w in p})
    ids = load_word_ids(os.path.join(HERE, "data", "tokenizer.json"), words)

    U = unit_rows(E)
    Uc = unit_rows(shuffle_rows(E, 1730))  # mandatory control, seed 1730

    results = {"protocol": {"shuffle_seed": 1730, "twenty_shuffle_seeds": list(range(1731, 1751)),
                            "loo": True, "unit_row_normalization": True},
               "relations": {}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (name, pairs) in zip(axes, RELATIONS.items()):
        cos_t = relation_pair_cosines(U, ids, pairs)
        cos_c = relation_pair_cosines(Uc, ids, pairs)
        mean_t, mean_c = relation_mean(cos_t), relation_mean(cos_c)
        shuf_means = []
        for seed in range(1731, 1751):
            Us = unit_rows(shuffle_rows(E, seed))
            shuf_means.append(relation_mean(relation_pair_cosines(Us, ids, pairs)))

        weakest = pairs[int(np.argmin(cos_t))][0] + "→" + pairs[int(np.argmin(cos_t))][1]
        results["relations"][name] = {
            "table_mean_cosine": mean_t,
            "control_seed1730_mean_cosine": mean_c,
            "twenty_shuffle_means": shuf_means,
            "twenty_shuffle_mean_min": float(min(shuf_means)),
            "twenty_shuffle_mean_max": float(max(shuf_means)),
            "table_pair_cosines": {f"{a}->{b}": float(c) for (a, b), c in zip(pairs, cos_t)},
            "weakest_pair": weakest,
        }

        # figure: unit arrows, angle = arccos(cosine); table above, control below
        for cos, sign, color in [(cos_t, +1, "tab:blue"), (cos_c, -1, "0.55")]:
            for i, c in enumerate(cos):
                ang = np.arccos(np.clip(c, -1, 1))
                x, y = np.cos(ang), sign * np.sin(ang)
                ax.annotate("", xy=(x, y), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.1))
        ax.annotate("", xy=(1.0, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="k", lw=2.5))
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-1.15, 1.15)
        ax.set_title(f"{name}\n table {mean_t:.2f} | control {mean_c:.2f}")
        ax.set_xlabel("cosine with mean of other 8 offsets")
        ax.set_yticks([])

    fig.suptitle("potion-base-8M relations as directions (upper, blue: table; lower, grey: shuffled control)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "evidence", "part1-relations.png"), dpi=150)

    gates = {}
    for name, r in results["relations"].items():
        gates[f"{name}/table_mean_ge_0.3"] = r["table_mean_cosine"] >= 0.3
        gates[f"{name}/control_abs_le_0.15"] = abs(r["control_seed1730_mean_cosine"]) <= 0.15
    results["gates"] = gates
    results["gates_all_pass"] = all(gates.values())

    with open(os.path.join(HERE, "evidence", "part1-results.json"), "w") as f:
        json.dump(results, f, indent=2)

    for name, r in results["relations"].items():
        print(f"{name}: table mean {r['table_mean_cosine']:.4f} (gate >= 0.3), "
              f"control {r['control_seed1730_mean_cosine']:+.4f} (gate |.| <= 0.15), "
              f"20-shuffle means in [{r['twenty_shuffle_mean_min']:+.3f}, {r['twenty_shuffle_mean_max']:+.3f}], "
              f"weakest {r['weakest_pair']}")
    print("gates:", "ALL PASS" if results["gates_all_pass"] else f"FAIL {gates}")
    return 0 if results["gates_all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
