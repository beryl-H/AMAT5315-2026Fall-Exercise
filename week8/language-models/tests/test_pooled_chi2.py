"""Pooled chi-squared rare-bin correction (root review): after pooling all
bins with expected < 5, the pooled total itself may still be < 5. The pool
must then be merged with the smallest adequate bin until ALL expected >= 5.
A context reduced to a single group contributes 0 statistic and 0 df (no
fake p-value). Metadata must record minimum expected, bins/df, context counts."""
import numpy as np
import pytest

from src.ngrams import pooled_chi2_bigram, V


def _fixture_P():
    """N=1000 draws; P = [.7, .297, 510 x (.003/510)]. Rare pool E = 3 (<5)
    under the old rule -> corrected to 2 bins: 700 and 300."""
    P = np.zeros(V)
    P[0] = 0.7
    P[1] = 0.297
    P[2:] = 0.003 / (V - 2)
    return P.reshape(1, -1)


def test_rare_pool_merged_with_smallest_adequate_bin():
    P = _fixture_P()
    # sampled stream: 1000 transitions from context 0 following P exactly
    rng = np.random.default_rng(0)
    u = rng.random(1000)
    cum = np.cumsum(P[0])
    nxt = np.searchsorted(cum, u, side="right").astype(int)
    sampled = np.concatenate([[0] * 1000, nxt])  # all contexts are 0
    stat, df, p, meta = pooled_chi2_bigram(
        sampled, P, top_ctx=[0], emin=5.0, return_meta=True
    )
    assert meta["contexts_used"] == 1
    assert meta["context_counts"][0] == 1000
    assert meta["bins"] == 2  # [700] and [300]; the E=3 pool merged into E=297
    assert meta["df"] == 1  # 2 bins - 1
    assert meta["minimum_expected"] == pytest.approx(300.0)
    # closed form: observed counts ~ (700, 300); stat = sum (O-E)^2/E over 2 bins
    o1 = int((nxt == 0).sum())
    o2 = 1000 - o1
    expected_stat = (o1 - 700.0) ** 2 / 700.0 + (o2 - 300.0) ** 2 / 300.0
    assert stat == pytest.approx(expected_stat)
    assert 0.0 < p < 1.0


def test_single_group_context_contributes_zero():
    # degenerate: one-hot P -> after pooling only one group -> df 0, stat 0
    P = np.zeros((1, V))
    P[0, 7] = 1.0
    sampled = np.array([0] * 500 + [7] * 500)
    stat, df, p, meta = pooled_chi2_bigram(
        sampled, P, top_ctx=[0], emin=5.0, return_meta=True
    )
    assert stat == 0.0 and df == 0
    assert np.isnan(p)  # no fake p-value from an uninformative context
