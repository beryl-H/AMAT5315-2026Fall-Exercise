"""Corpus split logic — derived from SPEC Part 2 input preparation.

Split on <|endoftext|>, strip surrounding whitespace, drop empty stories,
first 90% of whole stories (file order) = train, rest = validation.
"""
import numpy as np
import pytest

from src.corpus import encode_stories, split_stories, train_val_split

EOT = "<|endoftext|>"


def test_split_stories_separates_strips_and_drops_empty():
    text = f"  first story  {EOT}\n second story \n{EOT}{EOT}  {EOT}third{EOT}"
    assert split_stories(text) == ["first story", "second story", "third"]


def test_split_stories_no_eot_means_single_story():
    assert split_stories("only one story") == ["only one story"]


def test_train_val_split_first_90_percent_whole_stories():
    stories = [f"s{i}" for i in range(10)]
    train, val = train_val_split(stories)
    assert train == [f"s{i}" for i in range(9)]
    assert val == ["s9"]


def test_train_val_split_two_stories_gives_one_each():
    train, val = train_val_split(["a", "b"])
    assert train == ["a"] and val == ["b"]


def test_train_val_split_preserves_file_order():
    stories = ["x", "y", "z", "w", "v", "u", "t", "s", "r", "q"]
    train, val = train_val_split(stories)
    assert train == ["x", "y", "z", "w", "v", "u", "t", "s", "r"]
    assert val == ["q"]


def test_encode_stories_leading_eos_and_between_stories_no_bos():
    """Frozen convention: stream = [EOS] + s0 + [EOS] + s1 + ... (EOS before
    every story = 'EOS between stories' with a leading EOS as initial context);
    no BOS anywhere; the first stream token is context only."""
    from src.corpus import SPECIAL_EOS_ID

    class FakeSP:
        _eos = SPECIAL_EOS_ID

        def eos_id(self):
            return self._eos

        def encode(self, s):
            return {"a": [10], "b": [11, 12]}[s]

    stream = encode_stories(FakeSP(), ["a", "b"])
    assert stream == [SPECIAL_EOS_ID, 10, SPECIAL_EOS_ID, 11, 12]
    assert 1 not in stream  # no BOS
