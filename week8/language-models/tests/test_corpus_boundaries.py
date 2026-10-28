"""Boundary tests for the frozen corpus accounting convention (SPEC Part 2).

The single EOS prepended before the first story is a context sentinel, never a
corpus token or loss target. Actual corpus tokens have EOS between stories
only; scoring uses stream[1:]; n-gram histories are left-padded with (n-1) EOS.
"""
import os

import numpy as np
import pytest

from src.corpus import encode_stories, ngram_windows, kselect_boundary, chars_per_token

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
HAVE_DATA = os.path.exists(os.path.join(DATA, "tok512.model"))


class FakeSP:
    _eos = 2

    def eos_id(self):
        return self._eos

    def encode(self, s):
        return {"a": [10], "b": [11, 12], "c": [13]}[s]


def test_sentinel_is_context_only_inter_story_eos_are_actual_tokens():
    stream = encode_stories(FakeSP(), ["a", "b", "c"])
    # stream = [sentinel, 10, EOS, 11, 12, EOS, 13]
    assert stream == [2, 10, 2, 11, 12, 2, 13]
    actual = stream[1:]
    assert len(actual) == 6  # corpus tokens incl. 2 inter-story EOS
    assert stream[0] == 2 and 2 in actual  # sentinel separate; inter-story EOS actual
    assert stream.count(2) == 3  # 1 sentinel + 2 between stories


def test_ngram_windows_count_and_first_window_padding():
    stream = encode_stories(FakeSP(), ["a", "b", "c"])
    T = len(stream) - 1
    for n in (1, 2, 3, 6):
        ctx, tgt = ngram_windows(stream, n)
        assert len(ctx) == len(tgt) == T  # no first n-1 targets lost
        assert all(t != stream[0] or i > 0 for i, t in enumerate(tgt) for t in [tgt[i]])
        assert tgt[0] == stream[1]  # first target is the first actual token
    ctx2, _ = ngram_windows(stream, 2)
    assert ctx2[0] == stream[0]  # first bigram context is the sentinel
    ctx3, _ = ngram_windows(stream, 3)
    # first trigram context = EOS padding + sentinel (2 tokens, 9-bit packed)
    eos = FakeSP().eos_id()
    assert ctx3[0] == (eos << 9) | stream[0]
    ctx6, _ = ngram_windows(stream, 6)
    assert ctx6[0] == ((eos << 36) | (eos << 27) | (eos << 18) | (eos << 9) | stream[0])


def test_kselect_boundary_inner_fit_then_selection():
    stream = list(range(100, 111))  # 10 actual tokens after sentinel
    stream = [2] + stream
    b = kselect_boundary(len(stream) - 1, frac=0.9)
    assert b == 9  # inner fit targets = first 9 actual tokens, selection = 1
    b2 = kselect_boundary(9667034)
    assert b2 == 8700330  # floor(0.9 * 9667034)


@pytest.mark.skipif(not HAVE_DATA, reason="pinned tok512.model not downloaded")
def test_frozen_corpus_accounting_numbers():
    sp = __import__("src.corpus", fromlist=["load_spm"]).load_spm(os.path.join(DATA, "tok512.model"))
    text = open(os.path.join(DATA, "TinyStoriesV2-GPT4-valid.txt"), encoding="utf-8").read()
    from src.corpus import split_stories, train_val_split

    train, val = train_val_split(split_stories(text))
    s_tr = encode_stories(sp, train)
    s_va = encode_stories(sp, val)
    assert len(s_tr) - 1 == 9667034
    assert len(s_va) - 1 == 1078995
    tr_chars = sum(len(s) for s in train)
    va_chars = sum(len(s) for s in val)
    assert tr_chars == 19853989
    assert va_chars == 2213915
    # uniform validation bpc, exact frozen formula
    import math

    bpc = 1078995 * math.log(512) / 2213915
    assert abs(bpc - 4.386327) < 5e-7
    assert chars_per_token(sp, train) == pytest.approx(19853989 / 9667034, rel=1e-12)
