"""Part 1: exact corpus fingerprint, n-gram baselines and the block protocol.

Everything follows the learning sheet: sorted 65-character vocabulary,
first 1,003,854 characters for training, last 111,540 for validation,
validation scored in blocks of 128 inputs (871 full blocks + one final
block of 51 inputs) so that every validation character except the first
is scored exactly once. Losses are token-weighted nats.
"""
from __future__ import annotations

import math

import numpy as np

N_TRAIN = 1_003_854
BLOCK = 128


def sorted_vocab(text: str) -> list[str]:
    """The 65 corpus characters in sorted order; id = rank."""
    return sorted(set(text))


def encode(text: str, vocab: list[str]) -> np.ndarray:
    table = {c: i for i, c in enumerate(vocab)}
    return np.array([table[c] for c in text], dtype=np.int16)


def train_validation_split(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training = first 1,003,854 characters; validation = the rest."""
    return ids[:N_TRAIN], ids[N_TRAIN:]


def validation_blocks(val: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(inputs, targets) per block; targets never overlap, cover val[1:]."""
    blocks = []
    start = 0
    while start < len(val) - 1:
        n_tgts = min(BLOCK, len(val) - 1 - start)
        inputs = val[start:start + n_tgts]
        targets = val[start + 1:start + 1 + n_tgts]
        blocks.append((inputs, targets))
        start += n_tgts
    return blocks


def two_target_example() -> dict:
    """p = 1/2, 1/4: token-weighted loss ln(8)/2, PPL sqrt(8)."""
    nll = np.array([-math.log(0.5), -math.log(0.25)])
    loss = nll.sum() / 2  # token-weighted mean of the two targets
    return {"loss": loss, "ppl": math.exp(loss), "n_scored": 2}


def score_uniform(n_scored: int) -> dict:
    loss = math.log(65)
    return {"loss": loss, "ppl": math.exp(loss), "n_scored": n_scored}


def unigram_probs(train: np.ndarray, vocab_size: int) -> np.ndarray:
    """Add-one smoothing on training text only."""
    counts = np.bincount(train, minlength=vocab_size).astype(np.float64)
    return (counts + 1.0) / (counts.sum() + vocab_size)


def bigram_probs(train: np.ndarray, vocab_size: int) -> np.ndarray:
    """P(next | prev) with add-one smoothing, fitted on training text only."""
    counts = np.bincount(train[:-1] * vocab_size + train[1:],
                         minlength=vocab_size * vocab_size)
    counts = counts.reshape(vocab_size, vocab_size).astype(np.float64)
    return (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + vocab_size)


def _nll_table(probs: np.ndarray, prev: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    return -np.log(probs[prev, tgt])


def score_unigram(train: np.ndarray, val: np.ndarray) -> dict:
    probs = unigram_probs(train, 65)
    blocks = validation_blocks(val)
    nll = np.concatenate([-np.log(probs[tgt]) for _, tgt in blocks])
    loss = float(nll.sum() / nll.size)
    return {"loss": loss, "ppl": math.exp(loss), "n_scored": int(nll.size)}


def score_bigram(train: np.ndarray, val: np.ndarray) -> dict:
    probs = bigram_probs(train, 65)
    blocks = validation_blocks(val)
    parts = [_nll_table(probs, inp, tgt) for inp, tgt in blocks]
    nll = np.concatenate(parts)
    loss = float(nll.sum() / nll.size)
    return {"loss": loss, "ppl": math.exp(loss), "n_scored": int(nll.size)}
