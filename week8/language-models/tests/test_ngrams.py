"""Part 2 n-gram engine tests: 9-bit packing, add-k closed form, explicit
chunked joint enumeration, ancestral sampler, pooled chi-squared."""
import numpy as np
import pytest

from src.ngrams import (
    NgramCounts,
    ancestral_sample_bigram,
    enumerate_joint_sum,
    pack_window,
    pooled_chi2_bigram,
    score_nats,
)

V = 512


def make_stream(n_tokens=4000, seed=7):
    rng = np.random.default_rng(seed)
    toks = rng.integers(0, 400, size=n_tokens)
    toks[1::2] = (toks[0::2] + 1) % 400  # deterministic alternation
    return [2] + toks.tolist()  # sentinel = EOS id 2


def test_pack_window_uses_64_bits_no_32bit_collision():
    win = np.array([[511] * 6], dtype=np.uint64)
    key = pack_window(win)[0]
    assert key == (1 << 54) - 1  # all six 9-bit slots full
    assert key > (1 << 32)  # would collide if 32-bit shifts were used
    assert key < (1 << 63)  # safely inside uint64
    # distinct streams -> distinct keys (roundtrip via digit extraction)
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 512, size=(1000, 6)).astype(np.uint64)
    keys = pack_window(toks)
    assert len(np.unique(keys)) == 1000


def test_add_k_closed_form_matches_manual_counts():
    stream = [2, 5, 6, 5, 6, 5, 7, 5, 6]
    m = NgramCounts.fit(stream, n=2, k=0.01)
    # manual bigram counts from actual tokens stream[1:] with sentinel context
    from collections import Counter

    pairs = Counter(zip(stream[:-1], stream[1:]))
    ctxs = Counter(stream[:-1])
    for (a, b), n_ab in pairs.items():
        p = m.logprob_joint(np.array([(a << 9) | b], dtype=np.uint64),
                            np.array([a], dtype=np.uint64))
        expect = (n_ab + 0.01) / (ctxs[a] + 0.01 * V)
        assert np.isclose(p[0], expect)
    # unseen context: uniform 1/V
    p = m.logprob_joint(np.array([(511 << 9) | 3], dtype=np.uint64),
                        np.array([511], dtype=np.uint64))
    assert np.isclose(p[0], 0.01 / (0.01 * V))


def test_explicit_enumeration_sums_to_one_small_vocab():
    # tiny-vocab model: pack 9 bits/token still, but V=4 grid enumerated
    stream = [2, 0, 1, 0, 2, 1, 1, 0, 3, 0, 1]
    for n in (1, 2, 3):
        m = NgramCounts.fit(stream, n=n, k=0.05, V=4)
        for L in (1, 2, 3):
            s = enumerate_joint_sum(m, L, chunk=7)  # small chunk forces chunking
            assert abs(s - 1.0) < 1e-12, (n, L, s)
        # closed form: dense conditional product equals chunked enumeration
        if n == 2:
            P = np.zeros((4, 4))
            for a in range(4):
                row = np.array([(a << 9) | b for b in range(4)], dtype=np.uint64)
                ctx = np.array([a] * 4, dtype=np.uint64)
                P[a] = m.logprob_joint(row, ctx)
            # explicit length-2 product: sum_{a,b} p(a|sentinel) p(b|a)
            p_first = m.logprob_joint(
                np.array([(2 << 9) | b for b in range(4)], dtype=np.uint64),
                np.array([2] * 4, dtype=np.uint64),
            )
            total = float(np.einsum("a,ab->", p_first, P))
            assert abs(total - 1.0) < 1e-12


def test_ancestral_sampler_matches_distribution_and_is_reproducible():
    rng = np.random.default_rng(0)
    P = np.full((V, V), 1.0 / V)
    P[5] = 0.0; P[5, 3] = 1.0  # context 5 -> token 3 always (one-hot row)
    s1 = np.array(ancestral_sample_bigram(P, 50, seed=531501))
    s2 = np.array(ancestral_sample_bigram(P, 50, seed=531501))
    assert np.array_equal(s1, s2)  # reproducible
    ctx5 = s1[:-1] == 5
    assert ctx5.any()
    assert (s1[1:][ctx5] == 3).all()  # one-hot row: context 5 -> token 3 always
    # distribution: near-uniform context -> roughly uniform next tokens
    P2 = np.full((V, V), 1.0 / V)
    s = np.array(ancestral_sample_bigram(P2, 40000, seed=1))
    trans = s[:-1][np.isin(s[:-1], np.arange(8))]  # contexts 0..7 all uniform rows
    nxt = s[1:][np.isin(s[:-1], np.arange(8))]
    frac_top8 = np.isin(nxt, np.arange(8)).mean()
    assert 0.015 < frac_top8 < 0.020  # 8/512 ~ 0.0156, loose 3-sigma band


def test_pooled_chi2_good_sampler_passes_bad_two_lag_fails():
    from src.ngrams import two_lag_sample_bigram

    # trainable structure: P[b|a] = 0.9 to (a+1)%64, else spread
    P = np.full((V, V), 0.1 / V)
    for a in range(64):
        P[a, (a + 1) % 64] += 0.9
    P /= P.sum(axis=1, keepdims=True)
    # generate a LONG sample with the true chain
    s = np.array(ancestral_sample_bigram(P, 200000, seed=531501))
    stat, df, p = pooled_chi2_bigram(s, P, top_ctx=np.arange(20), emin=5.0)
    assert p > 0.001
    # bad two-lag sampler: x_t drawn from the row of x_{t-2}
    s_bad = np.array(two_lag_sample_bigram(P, 200000, seed=531501))
    stat_b, df_b, p_b = pooled_chi2_bigram(s_bad, P, top_ctx=np.arange(20), emin=5.0)
    assert p_b < 1e-6
