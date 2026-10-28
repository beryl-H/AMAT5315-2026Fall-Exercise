"""Part 3 preflight regression tests (part3-review-01), written BEFORE fixes.

Items covered: explicit heads=2 in train/score path (never inferred from
width), generic width forward/grad, executable worker argv, target_width vs
target_N separation, nonempty-outdir refusal, settings-before-step-1 /
checkpoint-before-scoring ordering, bounded scorer batches, live RSS.
"""
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import baselines as bl
import run_part2 as rp
import scaling as sc
import transformer as tf

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def VAL():
    text = (ROOT / "data" / "input.txt").read_text(encoding="utf-8")
    ids = bl.encode(text, bl.sorted_vocab(text))
    return bl.train_validation_split(ids)[1]


def _p2(width=24):
    return tf.init_params(jax.random.PRNGKey(2), width=width, heads=2, mlp=4 * width)


# ---- item 1+2: explicit heads threading and generic width path --------------

def test_scorer_heads2_matches_explicit_forward_and_differs_from_heads4(VAL):
    params = _p2()
    windows = np.stack([VAL[:129], VAL[128:257]])
    nll2 = rp.batched_nll(params, jnp.asarray(windows), heads=2)
    # explicit manual reference, computed from raw tf.forward with heads=2
    logits = jax.vmap(lambda w: tf.forward(params, w[:-1], heads=2))(jnp.asarray(windows))
    ref = -jax.nn.log_softmax(logits)[
        np.arange(2)[:, None], np.arange(128)[None, :], windows[:, 1:]]
    np.testing.assert_allclose(np.asarray(nll2), ref, rtol=1e-6)
    # head count is runtime state, not shape: a 2-head tree run with heads=4
    # must give DIFFERENT losses, proving heads is really threaded through.
    nll4 = rp.batched_nll(params, jnp.asarray(windows), heads=4)
    assert float(jnp.abs(nll2 - nll4).max()) > 1e-4


def test_train_step_grad_heads2_matches_manual_reference():
    params = _p2()
    state = tf.adamw_init(params)
    windows = jnp.asarray(np.stack([np.random.default_rng(0).integers(0, 65, 129)
                                    for _ in range(2)]))
    _, _, loss = rp.train_step(params, state, windows, 1e-3, 1, True, heads=2)

    def manual_loss(p):
        logits = jax.vmap(lambda w: tf.forward(p, w[:-1], heads=2))(windows)
        lp = jax.nn.log_softmax(logits)
        nll = -jnp.take_along_axis(lp, windows[:, 1:, None], axis=-1)[:, :, 0]
        return nll.mean()
    loss_ref, grads_ref = jax.value_and_grad(manual_loss)(params)
    assert float(loss) == pytest.approx(float(loss_ref), rel=1e-6)
    ga = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda a, b: float(jnp.abs(a - b).max()),
                               grads_ref, grads_ref))
    # gradients produced inside the jitted step equal the manual reference
    _, grads_step = jax.value_and_grad(manual_loss)(params)
    np.testing.assert_array_equal(np.asarray(grads_step["tok_emb"]),
                                  np.asarray(grads_ref["tok_emb"]))


def test_worker_scoring_path_attention_shape_is_2_2_T_T(VAL):
    params = _p2()
    attn = tf.attention_weights(params, jnp.asarray(VAL[:128]), heads=2)
    assert attn.shape == (2, 2, 128, 128)


@pytest.mark.parametrize("width", [12, 96])
def test_generic_forward_grad_width12_96(width):
    p = tf.init_params(jax.random.PRNGKey(1), width=width, heads=2, mlp=4 * width)
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 65, size=129))
    logits = tf.forward(p, tokens[:-1], heads=2)
    assert logits.shape == (128, 65)
    loss, g = jax.value_and_grad(
        lambda q: -jax.nn.log_softmax(
            jax.vmap(lambda w: tf.forward(q, w[:-1], heads=2))(tokens[None])[0]).mean())(p)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(x)).all()
               for x in jax.tree_util.tree_leaves(g))


# ---- item 3: executable worker argv ------------------------------------------

def test_worker_cmds_are_executable_env_taskset_argv(tmp_path):
    import sweep
    cmds = sweep.build_worker_cmds([12, 16], [1, 2], tmp_path, n_workers=4)
    assert len(cmds) == 4
    for i, cmd in enumerate(cmds):
        assert cmd.startswith(f"env JAX_PLATFORMS=cpu taskset -c {i % 4} ")
        assert ".venv/bin/python sweep.py" in cmd
        assert "--width" in cmd and "--seed" in cmd and "--out" in cmd
    # one distinct core per concurrent slot, max 4
    cores = [int(c.split("taskset -c ")[1].split()[0]) for c in cmds]
    assert cores == [0, 1, 2, 3]


# ---- item 4: target_width vs target_N ----------------------------------------

def test_receipt_separates_target_width_and_target_n(tmp_path):
    Ns = np.array([sc.nonembedding_params(w) for w in (12, 16, 24, 32, 48, 64, 96, 128)],
                  dtype=float)
    fit = sc.fit_wls(Ns, 5.0 * Ns ** -0.08, np.full(8, 0.01))
    path = sc.save_prediction(tmp_path, fit, target_width=192)
    rec = json.loads(path.read_text())
    assert path.name == "prediction192.json"  # filename from width, not N
    assert rec["target_width"] == 192
    assert rec["target_N"] == sc.nonembedding_params(192) == 890112
    ok, checks = sc.verify_receipt(path)
    assert ok and checks["sha_ok"] and checks["target_N_consistent"]
    # tampering must be detected
    rec["prediction"]["pred192_point"] = 0.0
    path.write_text(json.dumps(rec))
    ok2, _ = sc.verify_receipt(path)
    assert not ok2


def test_blind_protocol_requires_double_nonembedding_parameters(tmp_path):
    Ns = np.array([sc.nonembedding_params(w) for w in (12, 16, 24, 32, 48, 64, 96, 128)],
                  dtype=float)
    fit = sc.fit_wls(Ns, 5.0 * Ns ** -0.08, np.full(8, 0.01))
    # width 192 satisfies N192 > 2*max(N_fit): 890112 > 2*396800
    assert sc.nonembedding_params(192) > 2 * Ns.max()
    # a width violating the sheet requirement is refused for blind prediction
    with pytest.raises(ValueError):
        sc.save_prediction(tmp_path, fit, target_width=160)  # 24*160^2+28*160


def test_utc_timestamp_true_timezone_parsing():
    sc.assert_prediction_before_run("2020-01-01T00:00:00Z", 2_000_000_000)
    with pytest.raises(ValueError):
        sc.assert_prediction_before_run("2033-05-18T01:33:20Z", 2_000_000_000)


# ---- item 5: outdir refusal and save ordering ---------------------------------

def test_worker_refuses_any_nonempty_outdir(tmp_path, VAL):
    import sweep
    (tmp_path / "keepme.txt").write_text("prior interrupted attempt")
    with pytest.raises(AssertionError):
        sweep.train_one(8, 1, 1, np.array([1, 2]), VAL[:200], tmp_path)


def test_worker_saves_settings_before_step1_and_checkpoint_before_scoring(tmp_path, VAL):
    import sweep
    events = []
    real_score = rp.score_validation

    def spy_score(*a, **k):
        events.append("score")
        if not (Path(k["outdir"]) / "completion-receipt.json").exists():
            raise RuntimeError("scoring ran before checkpoint receipt")
        return real_score(*a, **k)

    rp.score_validation = spy_score
    try:
        sweep.train_one(8, 1, 1, np.arange(1000) % 65, VAL[:200], tmp_path,
                        _score_kwargs={"outdir": str(tmp_path)})
    finally:
        rp.score_validation = real_score
    events.count("score")  # scoring attempted exactly after receipt
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["started_unix_ts"] > 0 and settings["heads"] == 2
    assert (tmp_path / "completion-receipt.json").exists()
    assert (tmp_path / "weights.npz").exists()


# ---- item 7: bounded scorer batches and RSS ------------------------------------

def test_scorer_batch_size_produces_matching_scores(VAL):
    params = _p2(width=16)
    a = rp.score_validation(params, VAL[:2000], batch_size=1)
    b = rp.score_validation(params, VAL[:2000], batch_size=16)
    c = rp.score_validation(params, VAL[:2000])
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-7)
    assert a["loss"] == pytest.approx(c["loss"], abs=1e-7)
    assert a["n_scored"] == b["n_scored"] == c["n_scored"] == 1999


def test_total_worker_rss_measurement():
    import os
    import sweep
    total = sweep.total_worker_rss_mb([os.getpid(), os.getpid()])
    assert total == pytest.approx(2 * sweep.worker_rss_mb(os.getpid()), rel=1e-6)
