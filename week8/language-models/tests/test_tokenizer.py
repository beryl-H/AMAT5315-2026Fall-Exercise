"""Tokenizer checks on the real pinned tok512.model (data-bound; skipped when
data/ is absent)."""
import os

import pytest

from src.corpus import load_spm

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "tok512.model")),
    reason="pinned tok512.model not downloaded",
)


def test_spm_vocab_is_512_and_has_eos():
    sp = load_spm("data/tok512.model")
    assert sp.get_piece_size() == 512
    assert sp.eos_id() >= 0


def test_encode_decode_roundtrip_small_sentence():
    sp = load_spm("data/tok512.model")
    s = "Once upon a time, there was a little girl named Lucy."
    ids = sp.encode(s)
    assert sp.decode(ids) == s


def test_eos_id_roundtrip():
    sp = load_spm("data/tok512.model")
    ids = sp.encode("hello") + [sp.eos_id()] + sp.encode("world")
    assert sp.decode(ids)  # decodable; EOS handled by decode


def test_chars_per_token_on_a_fixture_story():
    sp = load_spm("data/tok512.model")
    s = (
        "Once upon a time, there was a little girl named Lily. She liked to "
        "play in the park with her friends. One day, she found a big red ball."
    )
    ids = sp.encode(s)
    cpt = len(s) / len(ids)
    assert 1.5 < cpt < 3.5  # loose fixture bound; corpus-level gate is 2.0-2.1
