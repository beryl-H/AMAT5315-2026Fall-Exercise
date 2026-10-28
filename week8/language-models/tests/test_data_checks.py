"""Data-bound checks against the pinned downloads (skipped when data/ absent).

Derived from SPEC: embedding shape/keys, 54 single-token Part-1 words, corpus
split 24867/2763, ~2.05 chars/token, exactly 3 roundtrip exceptions, no BOS.
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
HAVE_DATA = os.path.exists(os.path.join(DATA, "model.safetensors"))

pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="pinned data not downloaded")

from src.corpus import (  # noqa: E402
    chars_per_token,
    load_spm,
    roundtrip_exceptions,
    split_stories,
    train_val_split,
)
from src.part1_embeddings import load_embeddings, load_word_ids  # noqa: E402

PAIRS = {
    "country_capital": [
        ("iran", "tehran"), ("turkey", "ankara"), ("russia", "moscow"),
        ("china", "beijing"), ("france", "paris"), ("sweden", "stockholm"),
        ("japan", "tokyo"), ("peru", "lima"), ("kenya", "nairobi"),
    ],
    "male_female": [
        ("king", "queen"), ("man", "woman"), ("he", "she"),
        ("brother", "sister"), ("father", "mother"), ("son", "daughter"),
        ("uncle", "aunt"), ("boy", "girl"), ("husband", "wife"),
    ],
    "adj_comparative": [
        ("big", "bigger"), ("small", "smaller"), ("fast", "faster"),
        ("dark", "darker"), ("cold", "colder"), ("old", "older"),
        ("strong", "stronger"), ("long", "longer"), ("tall", "taller"),
    ],
}
ALL_WORDS = sorted({w for ps in PAIRS.values() for p in ps for w in p})


def test_embedding_table_shape_and_dtype():
    E = load_embeddings(os.path.join(DATA, "model.safetensors"))
    assert E.shape == (29528, 256)
    assert E.dtype == np.float32


def test_all_54_part1_words_are_single_vocab_tokens():
    ids = load_word_ids(os.path.join(DATA, "tokenizer.json"), ALL_WORDS)
    assert len(ids) == len(ALL_WORDS) == 54
    for word, i in ids.items():
        assert 0 <= i < 29528


def test_corpus_split_counts_24867_2763():
    text = open(os.path.join(DATA, "TinyStoriesV2-GPT4-valid.txt"), encoding="utf-8").read()
    stories = split_stories(text)
    train, val = train_val_split(stories)
    assert len(train) == 24867
    assert len(val) == 2763


def test_chars_per_token_about_2_05():
    text = open(os.path.join(DATA, "TinyStoriesV2-GPT4-valid.txt"), encoding="utf-8").read()
    stories = split_stories(text)
    train, val = train_val_split(stories)
    sp = load_spm(os.path.join(DATA, "tok512.model"))
    cpt_train = chars_per_token(sp, train)
    assert 2.0 < cpt_train < 2.1, f"chars/token {cpt_train} outside 2.0-2.1"


def test_exactly_3_roundtrip_exceptions_on_validation():
    text = open(os.path.join(DATA, "TinyStoriesV2-GPT4-valid.txt"), encoding="utf-8").read()
    stories = split_stories(text)
    train, val = train_val_split(stories)
    sp = load_spm(os.path.join(DATA, "tok512.model"))
    exc = roundtrip_exceptions(sp, val)
    assert len(exc) == 3, f"expected 3 roundtrip exceptions, got {len(exc)}: indices {exc}"
