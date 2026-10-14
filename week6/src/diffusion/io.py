"""Serialization for samples.jsonl: index i and clipped, rounded pixels."""

from __future__ import annotations

import numpy as np


def clip_to_pm1(x):
    return np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)


def sample_rows(x) -> list[dict]:
    """One object per sample: i (index), x (flat row-major pixels in [-1,1],
    rounded to 3 decimals)."""
    x = clip_to_pm1(x)
    n = x.shape[0]
    flat = np.round(x.reshape(n, -1), 3)
    return [{"i": i, "x": flat[i].tolist()} for i in range(n)]