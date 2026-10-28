"""Part 2 formal driver: n-gram ladder n = 1..6 on TinyStories.

Exclusivity: each attempt maps to artifacts/part2/<attempt>/ created with
exist_ok=False BEFORE any settings write (no attempt overwrites); the console
log lives OUTSIDE the attempt directory and is created before the run.
Persistence: chosen k, calibration losses and per-n results are appended to
the exclusive results JSON as computed (crash-safe), not only at the end.
Caches: encoded streams trusted only with matching SHA256 + expected length +
recorded input hashes. Sampler arrays and the bigram table are saved under the
attempt directory for independent audit.

Protocol (SPEC, frozen): sentinel-EOS stream, targets = actual corpus tokens;
k in {1e-4,1e-3,0.01,0.1,1} chosen per n on the last 10% of training tokens
(counts fitted on the inner 90%), then refit on the full training tokens;
validation scored once per (n, chosen k). Explicit chunked joint enumeration
for the bigram (all V^3 sequences, gate 1e-12). Bigram ancestral sampler
250000 tokens (seed 531501), pooled chi-squared over the 20 most frequent
training contexts (rare bins pooled and merged until ALL expected >= 5);
two-lag negative control, same seed.
"""
import argparse
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
    encode_stories, kselect_boundary, load_spm, split_stories, train_val_split,
)
from src.ngrams import (  # noqa: E402
    NgramCounts, ancestral_sample_bigram, bigram_dense_table, enumerate_joint_sum,
    pooled_chi2_bigram, score_nats, two_lag_sample_bigram, V,
)

EV = os.path.join(HERE, "evidence")
K_GRID = [1e-4, 1e-3, 0.01, 0.1, 1.0]
SEED_SAMPLER = 531501
N_TOKENS_SAMPLE = 250_000
TOP_CTX = 20
EMIN = 5.0

SHEET_REF = {  # nats/token, perplexity, bits/char from the sheet table
    "uniform": (6.238, 512.0, 4.386),
    "1": (4.839, 126.4, 3.403),
    "2": (3.038, 20.9, 2.136),
    "3": (2.094, 8.12, 1.473),
}


class AttemptBusy(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def prepare_attempt(art_root, attempt):
    """Atomically create artifacts/part2/<attempt>/; abort if it exists."""
    d = os.path.join(art_root, attempt)
    if os.path.exists(d):
        raise AttemptBusy(f"attempt directory already exists: {d}")
    os.makedirs(d)  # exist_ok=False: atomic guard
    return d, {}


def cache_meta_ok(path, meta, expected_n_actual):
    if not os.path.exists(path):
        return False
    if meta.get("sha256") != sha256(path):
        return False
    return meta.get("n_actual") == expected_n_actual


def load_or_encode(sp, stories, name, cache_dir, expected_n_actual, input_hashes):
    """Encode or load a verified cache; never trust an unverified file."""
    path = os.path.join(cache_dir, f"encoded-{name}.npy")
    meta_path = path + ".json"
    if os.path.exists(path) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if cache_meta_ok(path, meta, expected_n_actual) and \
                meta.get("input_hashes") == input_hashes:
            return np.load(path).tolist(), True
    stream = encode_stories(sp, stories)
    assert len(stream) - 1 == expected_n_actual, "encode length mismatch"
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, np.asarray(stream, dtype=np.int32))
    json.dump({"sha256": sha256(path), "length": len(stream),
               "n_actual": len(stream) - 1, "stories": len(stories),
               "input_hashes": input_hashes},
              open(meta_path, "w"))
    return stream, False


class ResultSink:
    """Crash-safe incremental JSON writer inside the exclusive attempt dir."""

    def __init__(self, attempt_dir):
        self.path = os.path.join(attempt_dir, "results.json")
        self.data = {}

    def update(self, **kv):
        self.data.update(kv)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)


def make_logger(console_path, attempt_dir):
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        for p in (console_path, os.path.join(attempt_dir, "progress.log")):
            with open(p, "a") as f:
                f.write(line + "\n")
    return log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="formal-02")
    args = ap.parse_args(argv)

    part2_art = os.path.join(HERE, "artifacts", "part2")
    os.makedirs(part2_art, exist_ok=True)
    console_path = os.path.join(part2_art, f"{args.attempt}.console.log")  # outside attempt dir
    open(console_path, "a").close()
    attempt_dir, _ = prepare_attempt(part2_art, args.attempt)
    log = make_logger(console_path, attempt_dir)

    settings = {
        "attempt": args.attempt,
        "protocol": {
            "sentinel": "single EOS before first story, context only, never a target",
            "targets": "stream[1:] = actual corpus tokens (inter-story EOS included)",
            "k_grid": K_GRID,
            "k_selection": "fit counts on inner 90% of train actual tokens, score last 10%; per-n best k; then refit full train",
            "history_padding": "n-1 EOS left-padded",
            "bpc_denominator": "Unicode chars of original story texts",
            "normalization_check": "explicit chunked enumeration of all V^3 bigram sequences, gate 1e-12",
            "sampler": f"bigram ancestral {N_TOKENS_SAMPLE} tokens seed {SEED_SAMPLER}; pooled chi2 over {TOP_CTX} most frequent training contexts, rare bins pooled and merged until all expected >= {EMIN}, df=sum(bins-1); two-lag negative control same seed",
        },
        "seeds": {"sampler": SEED_SAMPLER},
        "source_hashes": {f: sha256(os.path.join(HERE, "src", f))
                          for f in ["corpus.py", "ngrams.py", "run_part2.py"]},
        "data_hashes": {f: sha256(os.path.join(HERE, "data", f))
                        for f in ["TinyStoriesV2-GPT4-valid.txt", "tok512.model"]},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(attempt_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    # bound the code to what was hashed: refuse to run if files changed since
    for f, h in settings["source_hashes"].items():
        assert sha256(os.path.join(HERE, "src", f)) == h, f"source changed: {f}"

    sink = ResultSink(attempt_dir)
    sink.update(settings=settings)

    sp = load_spm(os.path.join(HERE, "data", "tok512.model"))
    text = open(os.path.join(HERE, "data", "TinyStoriesV2-GPT4-valid.txt"), encoding="utf-8").read()
    train_stories, val_stories = train_val_split(split_stories(text))
    input_hashes = {"corpus": settings["data_hashes"]["TinyStoriesV2-GPT4-valid.txt"],
                    "tokenizer": settings["data_hashes"]["tok512.model"]}
    # existing unhashed cache is superseded; oracle cross-check pending until
    # support/oracle-data is published by the root (never coerced)
    train, used_cache_tr = load_or_encode(sp, train_stories, "train",
                                          os.path.join(part2_art, "cache"),
                                          9667034, input_hashes)
    val, used_cache_va = load_or_encode(sp, val_stories, "val",
                                        os.path.join(part2_art, "cache"),
                                        1078995, input_hashes)
    T_train, T_val = len(train) - 1, len(val) - 1
    val_chars = sum(len(s) for s in val_stories)
    log(f"train actual {T_train} (cache={used_cache_tr}), val actual {T_val} "
        f"(cache={used_cache_va}), val chars {val_chars}")
    assert T_train == 9667034 and T_val == 1078995 and val_chars == 2213915
    sink.update(accounting={"train_actual": T_train, "val_actual": T_val,
                            "val_chars": val_chars,
                            "cache_used": [used_cache_tr, used_cache_va],
                            "oracle_crosscheck": "pending: support/oracle-data not yet published"})

    results = {"models": {}, "normalization": {}, "sampler": {}}

    # --- k selection on last 10% of train tokens, per n ---
    inner = kselect_boundary(T_train)
    select_nats = T_train - inner
    chosen_k = {}
    for n in range(1, 7):
        cal = {}
        for k in K_GRID:
            m = NgramCounts.fit(train, n, k, fit_targets=inner)
            nats = float(score_nats(m, train)[inner:].sum()) / select_nats
            cal[str(k)] = nats
            log(f"k-select n={n} k={k}: selection nats/token {nats:.4f}")
        best_k = min(cal, key=cal.get)
        chosen_k[n] = float(best_k)
        sink.update(k_selection={str(n): {"chosen": best_k, "calibration": cal}})
        log(f"n={n} chosen k = {best_k}")

    # --- refit full train, score validation ---
    from src.corpus import ngram_windows

    for n in range(1, 7):
        m = NgramCounts.fit(train, n, chosen_k[n])
        nats = score_nats(m, val)
        total = float(nats.sum())
        N = len(nats)
        res = {"k": chosen_k[n], "nats_per_token": total / N,
               "perplexity": math.exp(total / N),
               "bits_per_char": total / math.log(2) / val_chars}
        ctx, tgt = ngram_windows(val, n)
        joints = ((ctx << np.uint64(9)) | tgt.astype(np.uint64)) if n > 1 \
            else tgt.astype(np.uint64)
        idx = np.searchsorted(m.joint_keys, joints)
        idx = np.clip(idx, 0, len(m.joint_keys) - 1)
        res["unseen_fraction"] = float((m.joint_keys[idx] != joints).mean())
        results["models"][str(n)] = res
        sink.update(models=results["models"])
        log(f"n={n}: loss {res['nats_per_token']:.4f} nats, ppl {res['perplexity']:.2f}, "
            f"bpc {res['bits_per_char']:.4f}, unseen {100*res['unseen_fraction']:.1f}%")

    results["models"]["uniform"] = {
        "nats_per_token": math.log(V), "perplexity": float(V),
        "bits_per_char": math.log(V) / math.log(2) * T_val / val_chars,
    }
    sink.update(models=results["models"])
    assert abs(results["models"]["uniform"]["bits_per_char"] - 4.386327) < 1e-5

    # --- explicit chunked joint enumeration: bigram, all V^3 sequences ---
    mb = NgramCounts.fit(train, 2, chosen_k[2])
    sums = {L: enumerate_joint_sum(mb, L) for L in (1, 2, 3)}
    results["normalization"]["2"] = sums
    sink.update(normalization=results["normalization"])
    log("bigram normalization: " + ", ".join(f"L={L}: {s:.15f}" for L, s in sums.items()))

    # --- bigram sampler + pooled chi2 + two-lag negative control ---
    P = bigram_dense_table(mb)
    np.save(os.path.join(attempt_dir, "bigram-P.npy"), P)
    sample = np.array(ancestral_sample_bigram(P, N_TOKENS_SAMPLE, seed=SEED_SAMPLER))
    bad = np.array(two_lag_sample_bigram(P, N_TOKENS_SAMPLE, seed=SEED_SAMPLER))
    np.save(os.path.join(attempt_dir, "sample-seed531501.npy"), sample)
    np.save(os.path.join(attempt_dir, "sample-twolag-seed531501.npy"), bad)
    top = np.argsort(-mb.ctx_counts)[:TOP_CTX]
    top_tokens = mb.ctx_keys[top]
    stat, df, p, meta = pooled_chi2_bigram(sample, P, top_ctx=top_tokens,
                                           emin=EMIN, return_meta=True)
    stat_b, df_b, p_b, meta_b = pooled_chi2_bigram(bad, P, top_ctx=top_tokens,
                                                   emin=EMIN, return_meta=True)
    results["sampler"] = {
        "n_tokens": N_TOKENS_SAMPLE, "seed": SEED_SAMPLER,
        "top_contexts": top_tokens.tolist(),
        "pooled_chi2": {"stat": stat, "df": df, "p_value": p, "meta": meta},
        "two_lag_control": {"stat": stat_b, "df": df_b, "p_value": p_b, "meta": meta_b},
    }
    sink.update(sampler=results["sampler"])
    log(f"sampler pooled chi2: stat {stat:.1f} df {df} p {p:.6f} "
        f"(min expected {meta['minimum_expected']}); two-lag control p {p_b:.3g}")

    # --- gates ---
    gates = {}
    for key, (r_nat, r_ppl, r_bpc) in SHEET_REF.items():
        m = results["models"][key]
        gates[f"loss_{key}_within_0.03"] = abs(m["nats_per_token"] - r_nat) <= 0.03
        gates[f"ppl_{key}_within_3pct"] = abs(m["perplexity"] - r_ppl) / r_ppl <= 0.03
    bpcs = {int(k): v["bits_per_char"] for k, v in results["models"].items() if k.isdigit()}
    ppls = {int(k): v["perplexity"] for k, v in results["models"].items() if k.isdigit()}
    gates["ladder_monotonic_to_n4"] = all(ppls[n] < ppls[n - 1] for n in (2, 3, 4))
    gates["ladder_rises_after_n4"] = ppls[5] > ppls[4] and ppls[6] > ppls[4]
    gates["best_at_n4"] = min(bpcs, key=bpcs.get) == 4
    gates["n4_1.310_within_0.03"] = abs(bpcs[4] - 1.310) <= 0.03
    gates["n5_1.46_within_0.03"] = abs(bpcs[5] - 1.46) <= 0.03
    gates["n6_1.79_within_0.03"] = abs(bpcs[6] - 1.79) <= 0.03
    gates["ppl_n6_within_3pct_of_12.8"] = abs(ppls[6] - 12.8) / 12.8 <= 0.03
    uref = {3: 0.006, 4: 0.042, 5: 0.122, 6: 0.235}
    for n, u in uref.items():
        gates[f"unseen_n{n}_within_2pts"] = \
            abs(results["models"][str(n)]["unseen_fraction"] - u) <= 0.02
    gates["normalization_1e-12"] = all(abs(s - 1.0) < 1e-12 for s in sums.values())
    gates["sampler_p_gt_0.001"] = p > 0.001
    gates["two_lag_control_rejected"] = p_b < 0.001
    results["gates"] = gates
    results["gates_all_pass"] = all(gates.values())
    sink.update(models=results["models"], normalization=results["normalization"],
                sampler=results["sampler"], gates=gates,
                gates_all_pass=results["gates_all_pass"],
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # --- figure (from the exclusive attempt's own numbers) ---
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
    # copy the attempt results into evidence as the cited formal output
    with open(os.path.join(EV, "part2-results.json"), "w") as f:
        json.dump({"attempt": args.attempt, **sink.data}, f, indent=2)

    log("gates: " + ("ALL PASS" if results["gates_all_pass"] else
                     str([g for g, ok in gates.items() if not ok])))
    return 0 if results["gates_all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
