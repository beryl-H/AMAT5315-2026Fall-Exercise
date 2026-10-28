"""Pooled chi-squared rare-bin correction (root review): after pooling all
bins with expected < 5, the pooled total itself may still be < 5. The pool
must then be merged with the smallest adequate bin until ALL expected >= 5.
A context reduced to a single group contributes 0 statistic and 0 df (no
fake p-value). Metadata must record minimum expected, bins/df, context counts."""
import numpy as np
import pytest

from src.ngrams import pooled_chi2_bigram, V


def _fixture_P():
    """N=1000 draws from context 0; next-token support excludes token 0 so the
    stream has exactly 1000 context-0 transitions. P = [0, .7, .297,
    509 x (.003/509)]. Rare pool E = 3 (<5) under the old rule -> corrected to
    2 bins: 700 and 300."""
    P = np.zeros(V)
    P[1] = 0.7
    P[2] = 0.297
    P[3:] = 0.003 / (V - 3)
    return P.reshape(1, -1)


def test_rare_pool_merged_with_smallest_adequate_bin():
    P = _fixture_P()
    # sampled stream: exactly 1000 transitions from context 0 following P;
    # interleave context 0 with drawn next-tokens (support excludes 0, so the
    # interleaved zeros are the only zeros and each zero is a ctx-0 transition)
    rng = np.random.default_rng(0)
    u = rng.random(1000)
    nxt = np.searchsorted(np.cumsum(P[0]), u, side="right").astype(int)
    sampled = np.empty(2000, dtype=int)
    sampled[0::2] = 0
    sampled[1::2] = nxt
    stat, df, p, meta = pooled_chi2_bigram(
        sampled, P, top_ctx=[0], emin=5.0, return_meta=True
    )
    assert meta["contexts_used"] == 1
    assert meta["context_counts"][0] == 1000
    assert meta["bins"] == 2  # [700] and [300]; the E=3 pool merged into E=297
    assert meta["df"] == 1  # 2 bins - 1
    assert meta["minimum_expected"] == pytest.approx(300.0)
    # closed form after the corrected merge: bins are [700] and [300] where
    # the second group = token 2 plus the whole rare pool
    o1 = int((nxt == 1).sum())
    o2 = int((nxt == 2).sum())
    o_rest = 1000 - o1 - o2
    expected_stat = (o1 - 700.0) ** 2 / 700.0 + (o2 + o_rest - 300.0) ** 2 / 300.0
    assert stat == pytest.approx(expected_stat)
    assert 0.0 < p < 1.0


def test_single_group_context_contributes_zero():
    # degenerate: one-hot P -> after pooling only one group -> df 0, stat 0
    P = np.zeros((1, V))
    P[0, 7] = 1.0
    sampled = np.empty(1000, dtype=int)
    sampled[0::2] = 0
    sampled[1::2] = 7  # 500 ctx-0 transitions, all nexts = 7
    stat, df, p, meta = pooled_chi2_bigram(
        sampled, P, top_ctx=[0], emin=5.0, return_meta=True
    )
    assert stat == 0.0 and df == 0
    assert np.isnan(p)  # no fake p-value from an uninformative context
