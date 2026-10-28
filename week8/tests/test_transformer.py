"""Part 2 tests, written BEFORE the implementation (sheet-derived).

Breaks caught: wrong parameter count (missing LayerNorms/head bias), mask not
applied (probe nonzero), rows not normalized, wrong schedule (no warmup or
wrong endpoint), coupled instead of decoupled weight decay, per-batch
(perplexity) averaging instead of token weighting.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import transformer as tf

CORPUS = Path(__file__).resolve().parent.parent / "data" / "input.txt"


@pytest.fixture(scope="module")
def ids():
    return np.frombuffer(CORPUS.read_bytes(), dtype=np.uint8).astype(np.int64)


def test_parameter_count_is_exactly_116673():
    params = tf.init_params(jax.random.PRNGKey(1337))
    assert tf.count_params(params) == 116673


def test_causal_attention_upper_triangle_exactly_zero_and_rows_sum_one():
    params = tf.init_params(jax.random.PRNGKey(0))
    tokens = jnp.array(np.random.default_rng(0).integers(0, 65, size=128))
    attn = tf.attention_weights(params, tokens)  # [layers, heads, T, T]
    assert attn.shape == (2, 4, 128, 128)
    upper = attn.sum(axis=1)  # zero above diagonal for every head
    for l in range(2):
        for h in range(4):
            above = jnp.triu(attn[l, h], k=1)
            assert float(jnp.max(jnp.abs(above))) == 0.0  # exactly zero
            row_sums = attn[l, h].sum(axis=-1)
            np.testing.assert_allclose(np.asarray(row_sums), 1.0, atol=1e-5)


def test_prefix_invariance_probe_is_exactly_zero(ids):
    params = tf.init_params(jax.random.PRNGKey(1337))
    pairs = tf.find_prefix_pairs(ids, n_pairs=8)
    assert len(pairs) == 8
    for a, b in pairs:
        w_a, w_b = ids[a:a + 128], ids[b:b + 128]
        assert np.array_equal(w_a[:64], w_b[:64])
        assert not np.array_equal(w_a, w_b)
    probe = tf.prefix_probe(params, ids, pairs)
    assert probe == 0.0  # exactly zero, per the sheet


def test_unmasked_forward_breaks_prefix_invariance(ids):
    # Negative control at the unit level: removing the mask must let later
    # characters change earlier logits, so any probe larger than zero detects
    # leakage.
    params = tf.init_params(jax.random.PRNGKey(1337))
    pairs = tf.find_prefix_pairs(ids, n_pairs=8)
    probe = tf.prefix_probe(params, ids, pairs, mask=False)
    assert probe > 0.0


def test_learning_rate_schedule_warmup_cosine():
    # 3e-3 warmup over 100 steps, cosine decay to 3e-4 at step 3000.
    assert tf.lr_schedule(0) == pytest.approx(0.0, abs=1e-12)
    assert tf.lr_schedule(50) == pytest.approx(1.5e-3, rel=1e-9)
    assert tf.lr_schedule(100) == pytest.approx(3e-3, rel=1e-9)
    mid = 3e-4 + 0.5 * (3e-3 - 3e-4) * (1 + np.cos(np.pi * 0.5))
    assert tf.lr_schedule(1550) == pytest.approx(mid, rel=1e-9)
    assert tf.lr_schedule(3000) == pytest.approx(3e-4, rel=1e-9)
    assert tf.lr_schedule(4000) == pytest.approx(3e-4, rel=1e-9)  # held after end


def test_adamw_decoupled_weight_decay_math():
    # Zero gradient: pure decoupled decay p <- p - lr*wd*p.
    state = tf.adamw_init({"w": jnp.array(1.0)})
    params, state = tf.adamw_step({"w": jnp.array(1.0)},
                                  {"w": jnp.array(0.0)}, state,
                                  lr=0.1, weight_decay=0.1, step=1)
    assert float(params["w"]) == pytest.approx(0.99, abs=1e-9)
    # One more step with unit gradient, checked against a manual update.
    p1 = params["w"]
    params, _ = tf.adamw_step(params, {"w": jnp.array(1.0)}, state,
                              lr=0.1, weight_decay=0.1, step=2)
    m = 0.9 * 0.0 + 0.1 * 1.0
    v = 0.999 * 0.0 + 0.001 * 1.0
    m_hat = m / (1 - 0.9 ** 2)
    v_hat = v / (1 - 0.999 ** 2)
    expected = p1 - 0.1 * 0.1 * p1 - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
    assert float(params["w"]) == pytest.approx(float(expected), abs=1e-7)


def test_token_weighted_loss_combination():
    a = jnp.full((3,), 1.0)  # 3 targets at loss 1
    b = jnp.full((1,), 2.0)  # 1 target at loss 2
    res = tf.combine_nll([a, b])
    assert res["loss"] == pytest.approx(1.25, abs=1e-9)  # not 1.5
    assert res["n_scored"] == 4


def test_window_sampling_deterministic_and_in_range(ids):
    k1 = jax.random.PRNGKey(1337)
    k2 = jax.random.PRNGKey(1337)
    w1 = tf.sample_windows(k1, ids[:1_003_854], batch=16)
    w2 = tf.sample_windows(k2, ids[:1_003_854], batch=16)
    # 129 = 128 model inputs + the next character as target, as in scoring.
    assert w1.shape == (16, 129)
    np.testing.assert_array_equal(np.asarray(w1), np.asarray(w2))
    assert w1.min() >= 0 and w1.max() <= 64


def test_generation_deterministic_and_valid():
    params = tf.init_params(jax.random.PRNGKey(1337))
    prompt = jnp.array([ord(c) for c in "ROMEO:\n"])
    out1 = tf.generate(params, prompt, n_new=20, temperature=1.0, key=jax.random.PRNGKey(7))
    out2 = tf.generate(params, prompt, n_new=20, temperature=1.0, key=jax.random.PRNGKey(7))
    assert out1.shape == (20 + 7,)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))
    assert out1.min() >= 0 and out1.max() <= 64
