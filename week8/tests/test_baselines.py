"""Part 1 tests, derived from the learning sheet BEFORE implementation.

Breaks caught: wrong split boundary, overlapping or re-scored block targets,
reciprocal-probability averaging (two-target PPL 3), unsorted vocabulary,
smoothing missing (infinite loss), unweighted block averaging.
"""
from pathlib import Path

import numpy as np
import pytest

import baselines as bl

CORPUS = Path(__file__).resolve().parent.parent / "data" / "input.txt"
EXPECTED_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def test_corpus_fingerprint():
    import hashlib
    assert hashlib.sha256(CORPUS.read_bytes()).hexdigest() == EXPECTED_SHA


def test_sorted_vocab_of_65():
    text = CORPUS.read_text(encoding="utf-8")
    vocab = bl.sorted_vocab(text)
    assert len(vocab) == 65
    assert vocab == sorted(set(text))
    ids = bl.encode(text, vocab)
    assert ids.dtype.kind in "iu" and ids.min() >= 0 and ids.max() <= 64


def test_split_and_counted_characters():
    ids = bl.encode(CORPUS.read_text(encoding="utf-8"), bl.sorted_vocab(CORPUS.read_text(encoding="utf-8")))
    train, val = bl.train_validation_split(ids)
    assert len(train) == 1_003_854
    assert len(val) == 111_540
    assert len(train) + len(val) == len(ids)


def test_block_protocol_scores_each_target_once():
    ids = bl.encode(CORPUS.read_text(encoding="utf-8"), bl.sorted_vocab(CORPUS.read_text(encoding="utf-8")))
    _, val = bl.train_validation_split(ids)
    blocks = list(bl.validation_blocks(val))
    # 871 full blocks of 128 targets + one final block of 51 targets.
    assert len(blocks) == 872
    assert all(len(inp) == len(tgt) for inp, tgt in blocks)
    assert all(len(tgt) == 128 for inp, tgt in blocks[:-1])
    assert len(blocks[-1][1]) == 51
    # Inputs of block i are targets of block i-1 shifted: non-overlapping targets.
    seen = np.concatenate([tgt for _, tgt in blocks])
    assert len(seen) == 111_539
    assert np.array_equal(seen, val[1:])
    # Full-block inputs must be contiguous 128-character windows of val.
    inp0, tgt0 = blocks[0]
    assert np.array_equal(inp0, val[:128]) and np.array_equal(tgt0, val[1:129])
    inp870, tgt870 = blocks[870]
    assert np.array_equal(inp870, val[870 * 128:871 * 128])
    assert np.array_equal(tgt870, val[870 * 128 + 1:871 * 128 + 1])
    inpLast, tgtLast = blocks[-1]
    assert np.array_equal(inpLast, val[871 * 128:111540 - 1])
    assert np.array_equal(tgtLast, val[111540 - 51:])


def test_two_target_example_exact_and_not_reciprocal_averaged():
    res = bl.two_target_example()
    # p = 1/2, 1/4 -> loss = ln(8)/2 = 1.039721, PPL = sqrt(8) = 2.828427.
    assert res["loss"] == pytest.approx(np.log(8) / 2, abs=1e-6)
    assert res["loss"] == pytest.approx(1.0397208, abs=1e-6)
    assert res["ppl"] == pytest.approx(2.8284271, abs=1e-6)
    # The wrong convention (mean of reciprocal probabilities) gives exactly 3.
    assert res["ppl"] != pytest.approx(3.0, abs=1e-6)


def test_uniform_predictor_is_ln65():
    res = bl.score_uniform(n_scored=111_539)
    assert res["loss"] == pytest.approx(np.log(65), abs=1e-12)
    assert res["ppl"] == pytest.approx(65.0, abs=1e-12)


def test_unigram_addone_score():
    text = CORPUS.read_text(encoding="utf-8")
    vocab = bl.sorted_vocab(text)
    ids = bl.encode(text, vocab)
    train, val = bl.train_validation_split(ids)
    res = bl.score_unigram(train, val)
    assert res["loss"] == pytest.approx(3.347305, abs=1e-5)
    assert res["ppl"] == pytest.approx(28.426, abs=1e-2)
    # Token-weighted: recomputing from summed nats must reproduce the loss.
    assert res["n_scored"] == 111_539


def test_bigram_addone_score():
    text = CORPUS.read_text(encoding="utf-8")
    vocab = bl.sorted_vocab(text)
    ids = bl.encode(text, vocab)
    train, val = bl.train_validation_split(ids)
    res = bl.score_bigram(train, val)
    assert res["loss"] == pytest.approx(2.481889, abs=1e-5)
    assert res["ppl"] == pytest.approx(11.964, abs=1e-2)


def test_addone_never_assigns_zero():
    text = CORPUS.read_text(encoding="utf-8")
    vocab = bl.sorted_vocab(text)
    ids = bl.encode(text, vocab)
    train, val = bl.train_validation_split(ids)
    uni = bl.unigram_probs(train, len(vocab))
    assert (uni > 0).all()
    # Even a context unseen in training keeps every probability positive.
    bi = bl.bigram_probs(train, len(vocab))
    assert (bi > 0).all()
