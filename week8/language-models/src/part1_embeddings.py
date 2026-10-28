"""Part 1: potion-base-8M embedding table and relation-direction analysis."""
import json

import numpy as np
from safetensors import safe_open


def load_embeddings(path: str) -> np.ndarray:
    with safe_open(path, framework="np") as f:
        return f.get_tensor("embeddings")  # (29528, 256) float32


def load_word_ids(tokenizer_json_path: str, words) -> dict[str, int]:
    """Word -> single vocab row id via tokenizer.json. Whole-word lookup only
    (no mean pooling, no '##' continuation pieces, no fallback): an OOV or
    multi-token word raises KeyError and must be reported, not worked around."""
    with open(tokenizer_json_path, encoding="utf-8") as f:
        vocab = json.load(f)["model"]["vocab"]
    out = {}
    for w in words:
        if w not in vocab:
            raise KeyError(f"word not a single vocab token: {w!r}")
        out[w] = vocab[w]
    return out


def unit_rows(E: np.ndarray) -> np.ndarray:
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def pair_loo_cosines(U: np.ndarray, pairs) -> np.ndarray:
    """Leave-one-out cosine of each pair offset with the mean offset of the
    other pairs. U: unit rows; pairs: list of (a_idx, b_idx)."""
    offsets = np.array([U[b] - U[a] for a, b in pairs])
    n = len(offsets)
    out = np.empty(n)
    for i in range(n):
        others = np.delete(offsets, i, axis=0)
        m = others.mean(axis=0)
        out[i] = float(offsets[i] @ m / (np.linalg.norm(offsets[i]) * np.linalg.norm(m)))
    return out


def relation_mean(pair_cosines: np.ndarray) -> float:
    return float(pair_cosines.mean())


def shuffle_rows(E: np.ndarray, seed: int) -> np.ndarray:
    """Control copy: same rows, permuted assignment, seeded."""
    perm = np.random.default_rng(seed).permutation(E.shape[0])
    return E[perm]
