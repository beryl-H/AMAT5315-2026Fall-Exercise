"""Part 2 driver regression tests (audit part2-review-01), written BEFORE the fixes.

Catches: eval dropping/altering scored targets (dummy windows, off-by-one on
the 51-char tail), swapped-target scoring, generation that does not use the
reloaded checkpoint, unpadded recompiling generation, checkpoint roundtrip
corruption of the layer list, missing attempt-directory preflight.
"""
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import baselines as bl
import transformer as tf
import run_part2 as rp

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def corpus():
    text = (ROOT / "data" / "input.txt").read_text(encoding="utf-8")
    vocab = bl.sorted_vocab(text)
    ids = bl.encode(text, vocab)
    return vocab, ids


@pytest.fixture(scope="module")
def params():
    # Nonuniform on purpose: scale embeddings so scores are not trivial.
    p = tf.init_params(jax.random.PRNGKey(1337))
    p["tok_emb"] = p["tok_emb"] * 3.0 + 0.5
    p["layers"][0]["wq"] = p["layers"][0]["wq"] * 2.0
    return p


def test_eval_matches_independent_next_token_scoring_111539(params, corpus):
    _, ids = corpus
    _, VAL = bl.train_validation_split(ids)
    res = rp.score_validation(params, VAL)
    assert res["n_scored"] == 111_539
    # Independent reference: direct next-token log-probs from raw VAL.
    n_full = 871
    lp_parts = []
    for i in range(n_full):
        inp, tgt = VAL[128 * i:128 * i + 128], VAL[128 * i + 1:128 * i + 129]
        logits = tf.forward(params, jnp.asarray(inp))
        lp_parts.append(-jax.nn.log_softmax(logits)[np.arange(128), np.asarray(tgt)])
    inp, tgt = VAL[111488:111539], VAL[111489:111540]
    logits = tf.forward(params, jnp.asarray(inp))
    lp_parts.append(-jax.nn.log_softmax(logits)[np.arange(51), np.asarray(tgt)])
    ref = tf.combine_nll(lp_parts)
    assert ref["n_scored"] == 111_539
    assert res["loss"] == pytest.approx(ref["loss"], abs=1e-6)


def test_eval_includes_the_51_char_tail(params, corpus):
    # The final partial block must contribute its 51 targets.
    _, ids = corpus
    _, VAL = bl.train_validation_split(ids)
    res = rp.score_validation(params, VAL)
    res_no_tail = rp.score_validation(params, VAL[:-51])
    assert res["n_scored"] - res_no_tail["n_scored"] == 51
    assert res["loss"] != pytest.approx(res_no_tail["loss"], abs=1e-12)


def test_eval_rejects_swapped_targets(params, corpus):
    # Negative control: scoring with swapped targets must NOT reproduce the
    # driver's score (the test detects target handling defects).
    _, ids = corpus
    _, VAL = bl.train_validation_split(ids)
    good = rp.score_validation(params, VAL[:2000])
    swapped = rp.score_validation(params, VAL[:2000], sabotage="swap_targets")
    assert good["n_scored"] == swapped["n_scored"]
    assert abs(good["loss"] - swapped["loss"]) > 1e-4


def test_padded_generation_matches_simple_reference(params, corpus):
    vocab, _ = corpus
    prompt = jnp.asarray(bl.encode("ROMEO:\n", vocab))
    ref = tf.generate(params, prompt, n_new=10, temperature=1.0,
                      key=jax.random.PRNGKey(7))
    out = rp.generate_padded(params, prompt, n_new=10, temperature=1.0,
                             key=jax.random.PRNGKey(7))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_checkpoint_roundtrip_preserves_layer_list(params, tmp_path):
    path = tmp_path / "weights.npz"
    rp.save_params(params, path)
    loaded = rp.load_params(path)
    assert isinstance(loaded["layers"], list) and len(loaded["layers"]) == 2
    assert tf.count_params(loaded) == tf.count_params(params)
    for a, b in zip(jax.tree_util.tree_leaves(params),
                    jax.tree_util.tree_leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_checkpoint_hash_verification(params, tmp_path):
    path = tmp_path / "weights.npz"
    rp.save_params(params, path)
    ok, sha = rp.verify_saved_weights(params, path)
    assert ok and len(sha) == 64
    ok2, _ = rp.verify_saved_weights(tf.init_params(jax.random.PRNGKey(1)), path)
    assert not ok2  # different tensors must not verify


def test_negative_control_probe_needs_mask_false(params, corpus):
    _, ids = corpus
    p = tf.init_params(jax.random.PRNGKey(1337))
    p["pos_emb"] = p["pos_emb"].at[64:].multiply(50.0)  # suffix-dependent logits
    pairs = tf.find_prefix_pairs(np.asarray(ids), n_pairs=2)
    assert tf.prefix_probe(p, np.asarray(ids), pairs, mask=True) == 0.0
    assert tf.prefix_probe(p, np.asarray(ids), pairs, mask=False) > 0.0


def test_attempt_directory_preflight(tmp_path):
    d1 = rp.prepare_attempt(tmp_path, "attempt-x")
    assert d1.is_dir()
    (d1 / "manifest.json").write_text("{}")
    with pytest.raises(AssertionError):
        rp.prepare_attempt(tmp_path, "attempt-x")  # must refuse to overwrite
