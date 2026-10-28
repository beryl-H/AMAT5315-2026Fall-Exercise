"""Part 1 embedding-math unit tests on a tiny synthetic table (no data)."""
import numpy as np
import pytest

from src.part1_embeddings import (
    load_word_ids,
    pair_loo_cosines,
    relation_mean,
    shuffle_rows,
    unit_rows,
)


def _tiny_table():
    # rows: a0, b0, a1, b1 (hand-computable 2-pair case)
    E = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 2.0],
        ]
    )
    return E, [(0, 1), (2, 3)]


def test_unit_rows_scales_every_row_to_length_one():
    E = np.array([[3.0, 4.0], [0.0, -2.0]])
    U = unit_rows(E)
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0)
    assert np.allclose(U[0], [0.6, 0.8])


def test_pair_loo_cosines_matches_direct_formula_two_pairs():
    E, pairs = _tiny_table()
    U = unit_rows(E)
    got = pair_loo_cosines(U, pairs)
    o0 = U[1] - U[0]
    o1 = U[3] - U[2]
    cos01 = float(o0 @ o1 / (np.linalg.norm(o0) * np.linalg.norm(o1)))
    # LOO: pair 0 vs mean of others (here just o1), pair 1 vs o0
    assert np.allclose(got, [cos01, cos01])


def test_identical_offsets_give_loo_cosine_one():
    # All 9 pairs identical -> unit offsets all equal -> every LOO cosine is
    # exactly 1 (degenerate hand case exercising the LOO aggregation).
    u = np.array([1.0, 0.0])
    v = np.array([1.0, 1.0])
    E = np.vstack([np.tile(u, (9, 1)), np.tile(v, (9, 1))])
    U = unit_rows(E)
    pairs = [(i, 9 + i) for i in range(9)]
    got = pair_loo_cosines(U, pairs)
    assert np.allclose(got, 1.0, atol=1e-12)


def test_relation_mean_is_mean_of_nine():
    cos = np.array([0.5, -0.2, 0.3, 0.9, 0.1, 0.0, 0.4, -0.1, 0.7])
    assert relation_mean(cos) == pytest.approx(cos.mean())


def test_shuffle_rows_changes_assignment_keeps_multiset_reproducible():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(16, 4))
    S1 = shuffle_rows(E, 1730)
    S2 = shuffle_rows(E, 1730)
    S3 = shuffle_rows(E, 1731)
    assert np.array_equal(S1, S2)  # seed reproducibility
    assert not np.array_equal(S1, E)
    assert np.allclose(np.sort(S1, axis=0), np.sort(E, axis=0))  # same rows
    assert not np.allclose(S1, S3)  # different seed, different shuffle


def test_load_word_ids_maps_words_to_single_vocab_rows(tmp_path):
    import json

    tok = {
        "model": {
            "type": "WordPiece",
            "vocab": {"cat": 3943, "the": 1002, "sat": 1944},
        }
    }
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps(tok))
    ids = load_word_ids(str(p), ["cat", "the", "sat"])
    assert ids == {"cat": 3943, "the": 1002, "sat": 1944}


def test_load_word_ids_reports_oov_instead_of_fallback(tmp_path):
    import json

    tok = {"model": {"type": "WordPiece", "vocab": {"cat": 3943}}}
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps(tok))
    with pytest.raises(KeyError):
        load_word_ids(str(p), ["cat", "zzzunknownword"])
