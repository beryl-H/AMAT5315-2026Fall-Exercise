"""diag04: Initialization stream integrity for main seeds 0..4.

Recomputes init_params(jax.random.PRNGKey(seed)) and compares BITWISE with
checkpoint-init.npz of each formal run. Verifies:
  - saved init == exact seeded init (no contamination, no seed mixing);
  - per-seed inits differ pairwise (distinct RNG streams);
  - init param norms and init top-5 Fourier shares vs the sheet reference
    (norm ~28, share ~0.13).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jnp

import grok  # frozen source, read-only

ROOT = Path(__file__).resolve().parents[2]
ATT = ROOT / "artifacts" / "part4" / "formal-decoder-02"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    per_seed = {}
    for seed in range(5):
        saved = grok.load_params(ATT / f"main-s{seed}" / "checkpoint-init.npz")
        saved = grok._detree(saved)
        fresh = grok.init_params(jax.random.PRNGKey(seed))
        bitwise_equal = all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(jax.tree_util.tree_leaves(saved),
                            jax.tree_util.tree_leaves(fresh)))
        E = np.asarray(saved["emb"])
        ks, share = grok.top5(E)
        per_seed[seed] = {"bitwise_equal_to_recomputed_init": bool(bitwise_equal),
                          "param_norm": grok.param_norm(saved),
                          "init_top5": ks, "init_top5_share": share}
    pairwise_distinct = all(
        not np.array_equal(np.asarray(grok.load_params(ATT / f"main-s{i}" / "checkpoint-init.npz")["emb"]),
                           np.asarray(grok.load_params(ATT / f"main-s{j}" / "checkpoint-init.npz")["emb"]))
        for i in range(5) for j in range(i + 1, 5))
    norms = [per_seed[s]["param_norm"] for s in range(5)]
    shares = [per_seed[s]["init_top5_share"] for s in range(5)]
    result = {"diagnostic": "diag04_init_streams", "per_seed": per_seed,
              "pairwise_distinct_inits": bool(pairwise_distinct),
              "init_norm_range": [float(min(norms)), float(max(norms))],
              "init_share_range": [float(min(shares)), float(max(shares))],
              "sheet_reference": {"init_norm": "~28", "init_share": "~0.13"},
              "pass": bool(all(per_seed[s]["bitwise_equal_to_recomputed_init"]
                               for s in range(5)) and pairwise_distinct
                           and all(27.0 <= n <= 29.0 for n in norms)
                           and all(0.10 <= sh <= 0.16 for sh in shares))}
    (OUT / "diag04_init_streams.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
