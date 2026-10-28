"""Part 1 formal run: counts, exact two-target example, three predictors,
figures. Deterministic; no seeds involved. Writes evidence/part1-summary.json
and three figures. Producing command:
    uv run python run_part1.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baselines as bl

ROOT = Path(__file__).resolve().parent
EVID = ROOT / "evidence"
EVID.mkdir(exist_ok=True)

CORPUS = ROOT / "data" / "input.txt"
EXPECTED_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def main() -> None:
    raw = CORPUS.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == EXPECTED_SHA, f"corpus fingerprint mismatch: {sha}"

    text = raw.decode("utf-8")
    vocab = bl.sorted_vocab(text)
    assert len(vocab) == 65
    ids = bl.encode(text, vocab)
    train, val = bl.train_validation_split(ids)

    two = bl.two_target_example()
    uni = bl.score_unigram(train, val)
    bi = bl.score_bigram(train, val)
    uniform = bl.score_uniform(n_scored=bi["n_scored"])

    print(f"corpus sha256        : {sha}")
    print(f"training characters  : {len(train):,}")
    print(f"validation characters: {len(val):,}")
    print(f"scored characters    : {bi['n_scored']:,}")
    print(f"vocabulary size      : {len(vocab)}")
    print(f"two-target example   : loss {two['loss']:.6f}  PPL {two['ppl']:.6f}")
    for name, res in [("uniform", uniform), ("unigram", uni), ("bigram", bi)]:
        print(f"{name:8s}            : loss {res['loss']:.6f}  PPL {res['ppl']:.3f}")

    # Fixed acceptance checks from the SPEC (report, not hide, any miss).
    checks = {
        "counts": (len(train), len(val), bi["n_scored"]) == (1_003_854, 111_540, 111_539),
        "uniform_ln65": abs(uniform["loss"] - np.log(65)) < 1e-12 and abs(uniform["ppl"] - 65) < 1e-12,
        "two_target": abs(two["loss"] - 1.0397208) < 1e-6 and abs(two["ppl"] - 2.8284271) < 1e-6,
        "unigram_1e-5": abs(uni["loss"] - 3.347305) < 1e-5,
        "bigram_1e-5": abs(bi["loss"] - 2.481889) < 1e-5,
        "monotone": uniform["loss"] > uni["loss"] > bi["loss"],
    }
    for k, ok in checks.items():
        print(f"check {k:15s}: {'PASS' if ok else 'FAIL'}")
    assert all(checks.values()), f"acceptance failure: {[k for k, v in checks.items() if not v]}"

    summary = {
        "corpus_sha256": sha,
        "counts": {"train": len(train), "validation": len(val), "scored": bi["n_scored"]},
        "vocab_size": len(vocab),
        "two_target": {k: two[k] for k in ("loss", "ppl")},
        "uniform": {k: uniform[k] for k in ("loss", "ppl")},
        "unigram": {k: uni[k] for k in ("loss", "ppl")},
        "bigram": {k: bi[k] for k in ("loss", "ppl")},
        "acceptance_checks": {k: bool(v) for k, v in checks.items()},
    }
    (EVID / "part1-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Figure 1: unigram probabilities with the uniform level.
    probs = bl.unigram_probs(train, 65)
    order = np.argsort(probs)[::-1]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(65), probs[order], color="#3a6ea5")
    ax.axhline(1 / 65, color="crimson", ls="--", lw=1, label=f"uniform 1/65 = {1/65:.4f}")
    ax.set_xticks(range(65), [vocab[i] for i in order])
    ax.set_ylabel("unigram probability (add-one)")
    ax.set_title("Part 1: unigram character probabilities (train only), sorted; space is the most frequent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVID / "part1-unigram.png", dpi=150)
    plt.close(fig)

    # Figure 2: bigram distribution after 't'.
    t_id = vocab.index("t")
    bi_probs = bl.bigram_probs(train, 65)[t_id]
    order = np.argsort(bi_probs)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([vocab[i] for i in order], bi_probs[order], color="#7a3a95")
    ax.set_ylabel("P(next | previous = 't')")
    ax.set_title("Part 1: bigram distribution after 't' (add-one, train only); 'h' takes ~0.34")
    fig.tight_layout()
    fig.savefig(EVID / "part1-bigram-after-t.png", dpi=150)
    plt.close(fig)
    print(f"share of 'h' after 't': {bi_probs[vocab.index('h')]:.3f}")

    # Figure 3: held-out loss and PPL of the three predictors.
    names = ["uniform", "unigram", "bigram"]
    losses = [uniform["loss"], uni["loss"], bi["loss"]]
    ppls = [uniform["ppl"], uni["ppl"], bi["ppl"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, losses, color=["#999999", "#3a6ea5", "#c46a1f"])
    axes[0].set_ylabel("held-out loss (nats / character)")
    axes[0].set_title("Token-weighted cross-entropy")
    for i, v in enumerate(losses):
        axes[0].text(i, v + 0.04, f"{v:.4f}", ha="center", fontsize=9)
    axes[1].bar(names, ppls, color=["#999999", "#3a6ea5", "#c46a1f"])
    axes[1].set_ylabel("held-out perplexity")
    axes[1].set_title("PPL = exp(loss)")
    for i, v in enumerate(ppls):
        axes[1].text(i, v + 0.5, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("Part 1: each step of context lowers the held-out loss")
    fig.tight_layout()
    fig.savefig(EVID / "part1-scores.png", dpi=150)
    plt.close(fig)

    print("wrote evidence/part1-summary.json and 3 figures")


if __name__ == "__main__":
    main()
