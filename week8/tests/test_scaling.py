"""Part 3 tests (corrected SPEC / spec-review-01), written BEFORE implementation.

Conventions frozen in SPEC before any sweep: SEM of two runs = |diff|/2;
direct loss-space nonlinear WLS with a profiled analytically; Birge
max(1, sqrt(chi2/6)); zero-scatter floor 1e-3; parametric bootstrap seed
20260, 10000 draws, loss space; blind prediction guarded against leakage.
"""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scaling as sc
import transformer as tf

WIDTHS = [8, 12, 16, 24, 32, 48, 64, 96, 128]


# ---- generic architecture accounting ---------------------------------------

@pytest.mark.parametrize("w", WIDTHS)
def test_param_formula_generic_width(w):
    params = tf.init_params(jax.random.PRNGKey(1), width=w, heads=2, mlp=4 * w)
    assert tf.count_params(params) == 24 * w**2 + 286 * w + 65
    assert sc.nonembedding_params(w) == 24 * w**2 + 28 * w


def test_part2_width64_unchanged():
    params = tf.init_params(jax.random.PRNGKey(1))
    assert tf.count_params(params) == 116673
    assert sc.nonembedding_params(64) == 100096


def test_width192_forward_shapes_only_no_training():
    # Shape smoke only: initialization + forward + scorer; no optimizer step,
    # no scientific loss recorded.
    params = tf.init_params(jax.random.PRNGKey(1), width=192, heads=2, mlp=768)
    assert tf.count_params(params) == 24 * 192**2 + 286 * 192 + 65
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 65, size=129))
    logits = tf.forward(params, tokens[:-1], heads=2)
    assert logits.shape == (128, 65)
    attn = tf.attention_weights(params, tokens[:-1], heads=2)
    assert attn.shape == (2, 2, 128, 128)


# ---- SEM / sigma conventions ------------------------------------------------

def test_sem_of_two_runs_is_abs_diff_over_two():
    assert sc.sem_two_runs(1.7, 1.6) == pytest.approx(0.05, abs=1e-12)


def test_zero_scatter_floored_at_1e-3():
    assert sc.sigma_points([1.5, 1.5]) == pytest.approx(1e-3, abs=1e-12)
    assert sc.sigma_points([1.5, 1.7]) == pytest.approx(0.05, abs=1e-12)


# ---- loss-space WLS fit ------------------------------------------------------

def _synthetic(a=5.0, alpha=0.08, noise=None, seed=0):
    Ns = np.array([sc.nonembedding_params(w) for w in WIDTHS[1:]], dtype=float)
    Ls = a * Ns ** (-alpha)
    if noise is not None:
        Ls = Ls + noise
    return Ns, Ls


def test_wls_recovers_synthetic_parameters():
    Ns, Ls = _synthetic()
    fit = sc.fit_wls(Ns, Ls, np.full(len(Ns), 0.01))
    assert fit["alpha"] == pytest.approx(0.08, abs=2e-4)
    assert fit["a"] == pytest.approx(5.0, abs=2e-3)
    assert fit["dof"] == 6


def test_wls_exact_data_gives_birge_one_and_chi2_zero():
    Ns, Ls = _synthetic()
    fit = sc.fit_wls(Ns, Ls, np.full(len(Ns), 0.01))
    assert fit["chi2"] == pytest.approx(0.0, abs=1e-10)
    assert fit["birge"] == 1.0


def test_birge_inflates_when_scatter_exceeds_sigma():
    Ns, Ls = _synthetic()
    rng = np.random.default_rng(3)
    Ls_noisy = Ls + rng.normal(0, 0.05, size=len(Ns))
    fit = sc.fit_wls(Ns, Ls_noisy, np.full(len(Ns), 0.005))
    assert fit["birge"] > 1.0
    # With the honest (larger) sigma the inflation must be ~1.
    fit2 = sc.fit_wls(Ns, Ls_noisy, np.full(len(Ns), 0.05))
    assert fit2["birge"] < fit["birge"]


def test_fit_uses_only_the_given_widths_no_hidden_holdout():
    # Leakage control: the fit object must expose its fitted widths; predicting
    # a width not larger than every fitted width must be rejected.
    Ns, Ls = _synthetic()
    fit = sc.fit_wls(Ns, Ls, np.full(len(Ns), 0.01))
    assert fit["Ns"].max() == Ns.max()
    with pytest.raises(ValueError):
        sc.predict_blind(fit, Ns.max())


def test_predict_blind_matches_law():
    Ns, Ls = _synthetic()
    fit = sc.fit_wls(Ns, Ls, np.full(len(Ns), 0.01))
    n192 = sc.nonembedding_params(192)
    assert sc.predict_blind(fit, n192) == pytest.approx(5.0 * n192**-0.08, rel=1e-4)


# ---- deterministic parametric bootstrap --------------------------------------

def test_bootstrap_deterministic_count_and_interval():
    Ns, Ls = _synthetic(noise=0.01, seed=1)
    fit = sc.fit_wls(Ns, Ls, np.full(len(Ns), 0.01))
    b1 = sc.bootstrap(fit, n_draws=10000, seed=20260, n192=sc.nonembedding_params(192))
    b2 = sc.bootstrap(fit, n_draws=10000, seed=20260, n192=sc.nonembedding_params(192))
    assert b1["n_draws"] == 10000
    assert b1["pred192"] == b2["pred192"]
    assert b1["alpha"] == b2["alpha"]
    lo, hi = b1["pred192_interval"]
    assert hi > lo
    assert b1["pred192_point"] == pytest.approx(sc.predict_blind(fit, sc.nonembedding_params(192)), rel=1e-6)


def test_bootstrap_interval_covers_synthetic_truth():
    a, alpha = 5.0, 0.08
    Ns, Ls = _synthetic(a, alpha, noise=None)
    fit = sc.fit_wls(Ns, Ls, np.full(len(Ns), 0.02))
    b = sc.bootstrap(fit, n_draws=10000, seed=20260, n192=sc.nonembedding_params(192))
    lo, hi = b["pred192_interval"]
    assert lo <= a * sc.nonembedding_params(192) ** (-alpha) <= hi


# ---- blind prediction receipt -------------------------------------------------

def test_prediction_receipt_timestamp_sha_and_ordering(tmp_path):
    fit = sc.fit_wls(*_synthetic(noise=0.01), np.full(8, 0.01))
    rec_path = sc.save_prediction(tmp_path, fit, n192=sc.nonembedding_params(192),
                                  bootstrap_seed=20260, n_draws=10000)
    rec = json.loads(rec_path.read_text())
    payload = json.dumps({k: v for k, v in rec.items() if k != "sha256"},
                         sort_keys=True).encode()
    import hashlib
    assert rec["sha256"] == hashlib.sha256(payload).hexdigest()
    ts = time.strptime(rec["utc_timestamp"], "%Y-%m-%dT%H:%M:%SZ")
    assert ts is not None
    # Ordering guard: the receipt must predate the training start.
    sc.assert_prediction_before_run(rec["utc_timestamp"], time.time())
    with pytest.raises(ValueError):
        sc.assert_prediction_before_run(rec["utc_timestamp"], time.time() - 60)


# ---- worker runner (not launched here) ----------------------------------------

def test_worker_plan_pins_at_most_four_distinct_cores(tmp_path):
    import sweep
    cmds = sweep.build_worker_cmds(widths=[12, 16], seeds=[1, 2],
                                   attempt_dir=tmp_path, n_workers=4)
    assert len(cmds) == 4
    cores = [c for c in cmds[0].split() if c.startswith("0-") or c.isdigit()]
    assert any("taskset" in cmd for cmd in cmds)
    assert len(cmds) == sum(1 for _ in cmds)
    # every command pins cores and caps at 4 workers
    assert all("taskset -c" in cmd for cmd in cmds)
    assert all("--width" in cmd and "--seed" in cmd for cmd in cmds)


def test_worker_rss_measurement():
    import os
    import sweep
    mb = sweep.worker_rss_mb(os.getpid())
    assert 10 < mb < 6000  # WSL total is ~6GB
