"""Part 2 formal run: n-gram ladder n = 1..6 on TinyStories.

Protocol (SPEC, frozen): sentinel-EOS stream, targets = actual corpus tokens;
k in {1e-4,1e-3,0.01,0.1,1} chosen per n on the last 10% of training tokens
(counts fitted on the inner 90%), then refit on the full training tokens;
validation scored once per (n, chosen k). Explicit chunked joint enumeration
for lengths 1..3. Bigram ancestral sampler 250000 tokens (seed 531501),
pooled chi-squared over the 20 most frequent training contexts, rare bins
pooled to expected >= 5; two-lag negative control with the same seed.

Formal attempt 1: artifacts/part2/formal-01/ (settings + source hashes written
before any compute; progress + arrays stay in artifacts; evidence/ gets the
post-processed summary + figures).
"""
import hashlib
import json
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.corpus import (  # noqa: E402
    chars_per_token, encode_stories, kselect_boundary, load_spm, split_stories,
    train_val_split,
)
from src.ngrams import (  # noqa: E402
    NgramCounts, ancestral_sample_bigram, bigram_dense_table, enumerate_joint_sum,
    pooled_chi2_bigram, score_nats, two_lag_sample_bigram, V,
)

ART = os.path.join(HERE, "artifacts", "part2", "formal-01")
EV = os.path.join(HERE, "evidence")
K_GRID = [1e-4, 1e-3, 0.01, 0.1, 1.0]
SEED_SAMPLER = 531501
N_TOKENS_SAMPLE = 250_000
TOP_CTX = 20
EMIN = 5.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_settings():
    os.makedirs(ART, exist_ok=True)
    settings = {
        "attempt": "formal-01",
        "protocol": {
            "sentinel": "single EOS before first story, context only, never a target",
            "targets": "stream[1:] = actual corpus tokens (inter-story EOS included)",
            "k_grid": K_GRID,
            "k_selection": "fit counts on inner 90% of train actual tokens, score last 10%; per-n best k; then refit full train",
            "history_padding": "n-1 EOS left-padded",
            "bpc_denominator": "Unicode chars of original story texts",
            "normalization_check": "explicit chunked enumeration of all V^L sequences, L=1..3, gate 1e-12",
            "sampler": f"bigram ancestral {N_TOKENS_SAMPLE} tokens seed {SEED_SAMPLER}; pooled chi2 over {TOP_CTX} most frequent training contexts, expected >= {EMIN}, df=sum(bins-1); two-lag negative control same seed",
        },
        "seeds": {"sampler": SEED_SAMPLER},
        "source_hashes": {f: sha256(os.path.join(HERE, "src", f))
                          for f in ["corpus.py", "ngrams.py", "run_part2.py"]},
        "data_hashes": {f: sha256(os.path.join(HERE, "data", f))
                        for f in ["TinyStoriesV2-GPT4-valid.txt", "tok512.model"]},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(ART, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    return settings


def load_or_encode(sp, stories, name):
    """Cache encoded streams in git-ignored artifacts with integrity hashes."""
    import numpy as np

    path = os.path.join(HERE, "artifacts", "part2", f"encoded-{name}.npy")
    meta_path = path + ".json"
    if os.path.exists(path) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        stream = np.load(path).tolist()
        assert len(stream) == meta["length"], "cache length mismatch"
        return stream
    stream = encode_stories(sp, stories)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(stream, dtype=np.int32))
    json.dump({"length": len(stream), "n_actual": len(stream) - 1,
               "stories": len(stories),
               "story_chars": sum(len(s) for s in stories)}, open(meta_path, "w"))
    return stream


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "progress.log"), "a") as f:
        f.write(line + "\n")


def main():
    settings = write_settings()
    sp = load_spm(os.path.join(HERE, "data", "tok512.model"))
    text = open(os.path.join(HERE, "data", "TinyStoriesV2-GPT4-valid.txt"), encoding="utf-8").read()
    train_stories, val_stories = train_val_split(split_stories(text))
    train = load_or_encode(sp, train_stories, "train")
    val = load_or_encode(sp, val_stories, "val")
    T_train, T_val = len(train) - 1, len(val) - 1
    val_chars = sum(len(s) for s in val_stories)
    log(f"train actual {T_train}, val actual {T_val}, val chars {val_chars}")
    assert T_train == 9667034 and T_val == 1078995 and val_chars == 2213915

    results = {"settings": settings, "models": {}, "normalization": {}, "sampler": {}}

    # --- k selection on last 10% of train tokens, per n ---
    inner = kselect_boundary(T_train)  # fit targets count
    select_nats = T_train - inner
    chosen_k = {}
    for n in range(1, 7):
        best = None
        for k in K_GRID:
            m = NgramCounts.fit(train, n, k, fit_targets=inner)
            nats = float(score_nats(m, train)[inner:].sum())
            log(f"k-select n={n} k={k}: selection nats/token {nats/select_nats:.4f}")
            if best is None or nats < best[1]:
                best = (k, nats)
        chosen_k[n] = best[0]
        log(f"n={n} chosen k = {best[0]}")
    results["k_selection"] = {str(n): chosen_k[n] for n in chosen_k}

    # --- refit full train, score validation ---
    for n in range(1, 7):
        m = NgramCounts.fit(train, n, chosen_k[n])
        nats = score_nats(m, val)
        total = float(nats.sum())
        N = len(nats)
        ppl = math.exp(total / N)
        bpc = total / math.log(2) / val_chars
        # unseen fraction of held-out n-grams (joint n-blocks absent from train)
        from src.corpus import ngram_windows

        ctx, tgt = ngram_windows(val, n)
        joints = (ctx << np.uint64(9)) | tgt.astype(np.uint64) if n > 1 else tgt.astype(np.uint64)
        if n == 1:
            unseen = float(np.isin(joints, m.joint_keys, invert=True).mean())
        else:
            idx = np.searchsorted(m.joint_keys, joints)
            idx = np.clip(idx, 0, len(m.joint_keys) - 1)
            unseen = float((m.joint_keys[idx] != joints).mean())
        results["models"][str(n)] = {
            "k": chosen_k[n], "nats_per_token": total / N, "perplexity": ppl,
            "bits_per_char": bpc, "unseen_fraction": unseen,
        }
        log(f"n={n}: loss {total/N:.4f} nats, ppl {ppl:.2f}, bpc {bpc:.4f}, unseen {100*unseen:.1f}%")

    # uniform row is exact
    results["models"]["uniform"] = {
        "nats_per_token": math.log(V), "perplexity": float(V),
        "bits_per_char": math.log(V) / math.log(2) * T_val / val_chars,
    }
    assert abs(results["models"]["uniform"]["bits_per_char"] - 4.386327) < 1e-5

    # --- explicit chunked joint enumeration (lengths 1..3) ---
    for n in (1, 2, 3):
        m = NgramCounts.fit(train, n, chosen_k[n])
        sums = {L: enumerate_joint_sum(m, L) for L in (1, 2, 3)}
        results["normalization"][str(n)] = sums
        log(f"normalization n={n}: " + ", ".join(f"L={L}: {s:.15f}" for L, s in sums.items()))

    # --- bigram sampler + pooled chi2 + two-lag negative control ---
    mb = NgramCounts.fit(train, 2, chosen_k[2])
    P = bigram_dense_table(mb)
    sample = ancestral_sample_bigram(P, N_TOKENS_SAMPLE, seed=SEED_SAMPLER)
    # 20 most frequent training contexts (by context count)
    top = np.argsort(-mb.ctx_counts)[:TOP_CTX]
    top_tokens = mb.ctx_keys[top]
    stat, df, p = pooled_chi2_bigram(np.array(sample), P, top_ctx=top_tokens, emin=EMIN)
    bad = two_lag_sample_bigram(P, N_TOKENS_SAMPLE, seed=SEED_SAMPLER)
    stat_b, df_b, p_b = pooled_chi2_bigram(np.array(bad), P, top_ctx=top_tokens, emin=EMIN)
    results["sampler"] = {
        "n_tokens": N_TOKENS_SAMPLE, "seed": SEED_SAMPLER, "top_contexts": top_tokens.tolist(),
        "pooled_chi2": {"stat": stat, "df": df, "p_value": p},
        "two_lag_control": {"stat": stat_b, "df": df_b, "p_value": p_b},
    }
    log(f"sampler pooled chi2: stat {stat:.1f} df {df} p {p:.4f}; two-lag control p {p_b:.3g}")

    # --- gates ---
    ref = {"uniform": 6.238, "1": 4.839, "2": 3.038, "3": 2.094}
    gates = {}
    for key, r in ref.items():
        gates[f"loss_{key}_within_0.03"] = abs(results["models"][key]["nats_per_token"] - r) <= 0.03
    bpcs = {int(k): v["bits_per_char"] for k, v in results["models"].items() if k.isdigit()}
    gates["best_at_n4"] = min(bpcs, key=bpcs.get) == 4
    gates["n4_1.310_within_0.03"] = abs(bpcs[4] - 1.310) <= 0.03
    gates["n5_1.46_within_0.03"] = abs(bpcs[5] - 1.46) <= 0.03
    gates["n6_1.79_within_0.03"] = abs(bpcs[6] - 1.79) <= 0.03
    uref = {3: 0.006, 4: 0.042, 5: 0.122, 6: 0.235}
    for n, u in uref.items():
        gates[f"unseen_n{n}_within_2pts"] = abs(results["models"][str(n)]["unseen_fraction"] - u) <= 0.02
    gates["normalization_1e-12"] = all(
        abs(s - 1.0) < 1e-12 for d in results["normalization"].values() for s in d.values())
    gates["sampler_p_gt_0.001"] = p > 0.001
    gates["two_lag_control_rejected"] = p_b < 0.001
    results["gates"] = gates
    results["gates_all_pass"] = all(gates.values())

    # --- figures ---
    ns = sorted(bpcs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ns, [bpcs[n] for n in ns], "o-")
    axes[0].set_xlabel("n"); axes[0].set_ylabel("held-out bits per character")
    axes[0].set_title("n-gram ladder")
    axes[1].plot(ns, [100 * results["models"][str(n)]["unseen_fraction"] for n in ns], "s-")
    axes[1].set_xlabel("n"); axes[1].set_ylabel("% held-out n-grams unseen in train")
    axes[1].set_title("curse of dimensionality")
    fig.tight_layout()
    fig.savefig(os.path.join(EV, "part2-ladder.png"), dpi=150)

    with open(os.path.join(EV, "part2-results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("gates: " + ("ALL PASS" if results["gates_all_pass"] else
                     str([g for g, ok in gates.items() if not ok])))
    return 0 if results["gates_all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
