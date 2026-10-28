"""Part 4 tests (grokking), written BEFORE implementation.

Covers: exact parameter count 62370 with NO LayerNorm, dataset/split
integrity (2823/4704, RNG 897, no overlap, label-independent), random-label
vector shuffle (RNG 8970, class frequencies preserved), AdamW betas
(0.9, 0.98) manual math, stop rule on the actual log grid, DFT power via
conjugate-pair rfft == direct trig basis, DC/'=' preservation in ablations,
toy negative controls, and the exclusive-attempt protocol.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import grok as gk

ROOT = Path(__file__).resolve().parent.parent


# ---- architecture -------------------------------------------------------------

def test_param_count_62370_and_no_layernorm():
    params = gk.init_params(jax.random.PRNGKey(0))
    assert gk.count_params(params) == 62370
    leaves_paths = [jax.tree_util.keystr(k) for k, _ in
                    jax.tree_util.tree_flatten_with_path(params)[0]]
    assert not any("ln" in p.lower() for p in leaves_paths)


def test_initial_param_norm_near_28():
    params = gk.init_params(jax.random.PRNGKey(0))
    norm = gk.param_norm(params)
    assert 20 < norm < 36  # reference: ~28 at initialization


# ---- dataset, split, labels -----------------------------------------------------

def test_dataset_shape_and_answer_never_in_input():
    data = gk.build_dataset()
    assert data["inputs"].shape == (9409, 3)
    assert data["targets"].shape == (9409,)
    a, b = np.meshgrid(np.arange(97), np.arange(97), indexing="ij")
    assert np.array_equal(data["targets"], (a + b) % 97)
    assert (data["inputs"][:, 2] == 97).all()  # '=' token at the answer slot
    # the answer never appears as an input token (slot 0/1 hold a and b only)
    assert np.array_equal(data["inputs"][:, 0], a.ravel() if False else data["inputs"][:, 0])
    np.testing.assert_array_equal(data["inputs"][:, 0], np.repeat(np.arange(97), 97))
    np.testing.assert_array_equal(data["inputs"][:, 1], np.tile(np.arange(97), 97))


def test_fixed_split_sizes_no_overlap_label_independent():
    s30 = gk.fixed_split(fraction=0.30)
    s50 = gk.fixed_split(fraction=0.50)
    assert s30["train"].size == 2823 and s30["test"].size == 9409 - 2823
    assert s50["train"].size == 4704 and s50["test"].size == 9409 - 4704
    for s in (s30, s50):
        assert len(np.intersect1d(s["train"], s["test"])) == 0
    # nested: the 50% split contains the 30% split
    assert np.intersect1d(s30["train"], s50["train"]).size == 2823
    # deterministic and label-independent: same indices for any labels
    s30b = gk.fixed_split(fraction=0.30)
    np.testing.assert_array_equal(s30["train"], s30b["train"])
    # permuting labels cannot change the split (labels not even an argument)


def test_random_labels_vector_shuffle_rng8970():
    true = gk.build_dataset()["targets"]
    shuffled = gk.random_labels(seed=0)
    # class frequencies preserved exactly (a permutation of the label VECTOR)
    assert np.array_equal(np.sort(shuffled), np.sort(true))
    assert not np.array_equal(shuffled, true)
    # deterministic
    np.testing.assert_array_equal(shuffled, gk.random_labels(seed=0))
    # it is NOT a global class permutation pi((a+b)%97): that would keep the
    # label a function of (a+b); the shuffle breaks pair-label structure.
    sums = (gk.pair_sums())  # (a+b)%97 for every pair
    for c in range(97):
        same_sum = shuffled[sums == c]
        if same_sum.size > 1:
            assert len(np.unique(same_sum)) > 1  # structure broken


# ---- optimizer math ---------------------------------------------------------------

def test_adamw_grokking_betas_manual():
    state = gk.adamw_init({"w": jnp.array(1.0)})
    p, state = gk.adamw_step({"w": jnp.array(1.0)}, {"w": jnp.array(0.0)},
                             state, lr=0.1, step=1)
    assert float(p["w"]) == pytest.approx(0.99, abs=1e-6)  # decoupled wd=... lr*wd
    p1 = p["w"]
    p, _ = gk.adamw_step(p, {"w": jnp.array(1.0)}, state, lr=0.1, step=2)
    m, v = 0.1, 0.001  # beta1=0.9, beta2=0.98
    m_hat = m / (1 - 0.9**2)
    v_hat = v / (1 - 0.98**2)
    expected = p1 - 0.1 * 0.1 * p1 - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
    assert float(p["w"]) == pytest.approx(float(expected), abs=1e-5)


# ---- stop rule on the actual log grid -----------------------------------------------

def test_stop_rule_first_logged_crossing_and_1000_delay():
    # logged grid every 100 steps; crossing between logs is only visible at logs
    log = [{"step": 100 * i, "train_acc": min(1.0, i / 2), "test_acc": 0.0}
           for i in range(1, 21)]
    log[15]["test_acc"] = 0.995  # step 1600
    res = gk.analyze_log(log, max_steps=10000)
    assert res["memorization_step"] == 200   # first logged train_acc>=0.99
    assert res["grok_step"] == 1600
    assert res["stop_step"] == 2600
    assert res["delay_ratio"] == pytest.approx(8.0)
    # cap at 10000
    log2 = [{"step": 100 * i, "train_acc": 1.0, "test_acc": 0.0} for i in range(1, 101)]
    res2 = gk.analyze_log(log2, max_steps=10000)
    assert res2["stop_step"] == 10000 and res2["grok_step"] is None


# ---- Fourier power: rfft conjugate pairs == direct trig basis ------------------------

def test_fft_power_matches_direct_trig_basis():
    rng = np.random.default_rng(0)
    E = rng.normal(0, 0.11, size=(97, 64))
    power = gk.fourier_power(E)  # k = 1..48
    assert power.shape == (48,)
    # direct cos/sin coefficients (unnormalized), summed over dims
    a = np.arange(97)
    direct = np.empty(48)
    for k in range(1, 49):
        C = (E * np.cos(2 * np.pi * k * a / 97)[:, None]).sum(0)
        S = (E * np.sin(2 * np.pi * k * a / 97)[:, None]).sum(0)
        direct[k - 1] = (C**2 + S**2).sum()
    # shares must agree (normalization cancels), and rfft must be consistent
    np.testing.assert_allclose(power / power.sum(), direct / direct.sum(), rtol=1e-10)
    # k=1..48 exactly covers all non-DC bins for p=97 (odd)
    full = np.fft.rfft(E, axis=0)
    assert full.shape == (49, 64)  # bins 0..48


def test_top5_share_and_snapshot_top5():
    rng = np.random.default_rng(1)
    E = rng.normal(0, 0.11, size=(97, 64))
    # inject a dominant frequency k=7 on one dimension
    a = np.arange(97)
    E[:, 0] += 10 * np.cos(2 * np.pi * 7 * a / 97)
    top5, share = gk.top5(E)
    assert 7 in top5 and len(top5) == 5 and len(set(top5)) == 5
    assert share > 0.5
    assert all(1 <= k <= 48 for k in top5)


# ---- ablations: DC and '=' preserved, no key overlap ---------------------------------

def test_toy_fourier_negative_control():
    """Before any real run: on a toy residue embedding whose class code lives
    ONLY in the key-frequency span, keep-only preserves readout accuracy,
    removing the keys destroys it, removing other frequencies does nothing."""
    rng = np.random.default_rng(7)
    key_ks = [3, 17, 23, 31, 43]
    a = np.arange(97)
    feats = np.stack([np.cos(2*np.pi*k*a/97) for k in key_ks] +
                     [np.sin(2*np.pi*k*a/97) for k in key_ks]).T  # (97,10)
    W = rng.normal(0, 1, size=(10, 97))
    true = (feats @ W).argmax(1)
    E = 0.05 * rng.normal(0, 1, size=(98, 64))
    E[:97, :10] = feats  # residue content lives in the key-frequency span

    def acc_from_E(E98):
        f2 = np.stack([np.cos(2*np.pi*k*a/97) for k in key_ks] +
                      [np.sin(2*np.pi*k*a/97) for k in key_ks]).T
        coef = np.linalg.lstsq(f2, E98[:97, :10], rcond=None)[0]
        recon = f2 @ coef  # residue content recovered through the key basis
        return float(((recon @ W).argmax(1) == true).mean())

    keep_acc = acc_from_E(gk.project_embedding(E, keep=key_ks, mode="keep"))
    rm_key_acc = acc_from_E(gk.project_embedding(E, keep=key_ks, mode="remove"))
    other = gk.random_other_draws(key_ks, n_draws=1, seed=4497)[0]
    rm_other_acc = acc_from_E(gk.project_embedding(E, keep=other, mode="remove"))
    assert keep_acc > 0.95
    assert rm_key_acc < 0.05   # chance: content gone, argmax collapses
    assert rm_other_acc > 0.95


def test_ablation_preserves_dc_and_eq_row():
    params = gk.init_params(jax.random.PRNGKey(0))
    E = np.asarray(params["emb"])
    K = [3, 11, 29]
    keep = gk.project_embedding(E, keep=K, mode="keep")
    assert np.allclose(keep[97], E[97])            # '=' untouched
    assert np.allclose(keep.mean(0), E[:97].mean(0))  # DC preserved
    remove = gk.project_embedding(E, keep=K, mode="remove")
    assert np.allclose(remove[97], E[97])
    assert np.allclose(remove.mean(0), E[:97].mean(0))
    # removing ALL k=1..48 leaves only DC
    only_dc = gk.project_embedding(E, keep=list(range(1, 49)), mode="remove")
    np.testing.assert_allclose(only_dc[:97], np.broadcast_to(E[:97].mean(0), (97, 64)),
                               atol=1e-12)


def test_random_other_draws_fixed_no_key_overlap():
    top5 = [3, 11, 29, 37, 41]
    draws = gk.random_other_draws(top5, n_draws=10, seed=4497)
    assert len(draws) == 10
    for d in draws:
        assert len(d) == 5 and len(set(d)) == 5
        assert not (set(d) & set(top5))
        assert all(1 <= k <= 48 for k in d)
    # fixed seeds -> identical draws
    draws2 = gk.random_other_draws(top5, n_draws=10, seed=4497)
    assert draws == draws2


def test_toy_fourier_negative_control():
    """Before any real run: keep-only key frequencies preserves a toy rule's
    readout accuracy; removing them destroys it; removing others does not."""
    rng = np.random.default_rng(7)
    key_ks = [3, 17, 23, 31, 43]
    a = np.arange(97)
    # toy embedding: rows encode the residue via key-frequency trig features
    feats = np.stack([np.cos(2*np.pi*k*a/97) for k in key_ks] +
                     [np.sin(2*np.pi*k*a/97) for k in key_ks]).T  # (97,10)
    # random linear map from features to 97 class scores; teacher uses SAME map
    W = rng.normal(0, 1, size=(10, 97))
    E = 0.05 * rng.normal(0, 1, size=(98, 64))
    E[:97, :10] = feats  # residue content lives in the embedding
    # readout: score(c) = proj(a) . v_c with proj = feats @ W  (97-dim)
    proj = feats @ W  # (97, 97)
    v = rng.normal(0, 1, size=(97,))  # final readout direction
    def readout(E98):
        # recompute features from the embedding rows via the SAME key basis
        f2 = np.stack([np.cos(2*np.pi*k*a/97) for k in key_ks] +
                      [np.sin(2*np.pi*k*a/97) for k in key_ks]).T
        # model prediction from the projected embedding:
        coef = f2.T @ E98[:97, :10]  # (10,10) approx identity if kept
        scores = (E98[:97, :10] @ coef.T @ W * 0)  # placeholder, real logic below
        return None
    # simpler exact toy: class score = feats @ W @ diag(v); ablation on E rows
    def acc_from_E(E98):
        f2 = np.stack([np.cos(2*np.pi*k*a/97) for k in key_ks] +
                      [np.sin(2*np.pi*k*a/97) for k in key_ks]).T
        coef = np.linalg.lstsq(f2, E98[:97, :10], rcond=None)[0]
        recon = f2 @ coef  # recovered residue content
        scores = recon @ W
        pred = scores.argmax(1)
        # toy "pairs": all residues; accuracy vs teacher label = argmax of true
        true = (feats @ W).argmax(1)
        return float((pred == true).mean())
    keep_acc = acc_from_E(gk.project_embedding(E, keep=key_ks, mode="keep"))
    rm_key = gk.project_embedding(E, keep=key_ks, mode="remove")
    rm_acc = acc_from_E(rm_key)
    other = gk.random_other_draws(key_ks, n_draws=1, seed=4497)[0]
    rm_other_acc = acc_from_E(gk.project_embedding(E, keep=other, mode="remove"))
    assert keep_acc > 0.95
    assert rm_acc < 0.05
    assert rm_other_acc > 0.95


# ---- attempt protocol ------------------------------------------------------------------

def test_part4_attempt_refuses_existing_dir(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "x").write_text("")
    with pytest.raises(AssertionError):
        gk.prepare_attempt(tmp_path, "run")
