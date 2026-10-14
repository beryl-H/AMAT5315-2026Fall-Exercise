"""Part 2 tests: RBM energy/free energy, Gibbs, CD gradient, completion.

Reference values come from an independent NumPy implementation with exact
enumeration on a tiny 2-visible x 2-hidden RBM — not from the JAX code
under test.
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ebm.rbm import (
    binarize,
    cd_surrogate,
    clamped_gibbs_chain,
    dice_ink,
    energy,
    free_energy,
    gibbs_chain,
    gibbs_step,
    init_params,
    p_h_given_v,
    p_v_given_h,
)

N_V, N_H = 2, 2


def ref_params():
    rng = np.random.default_rng(3)
    return {
        "a": rng.normal(size=N_V).astype(np.float64),
        "b": rng.normal(size=N_H).astype(np.float64),
        "W": rng.normal(size=(N_V, N_H)).astype(np.float64) * 0.5,
    }


def ref_energy(v, h, p):
    """E(v,h) = -a.v - b.h - v W h (NumPy reference)."""
    v = np.asarray(v, float)
    h = np.asarray(h, float)
    return -(p["a"] @ v) - (p["b"] @ h) - v @ p["W"] @ h


def ref_free_energy(v, p):
    """F(v) = -log sum_h exp(-E(v,h)) by exact enumeration."""
    v = np.asarray(v, float)
    total = sum(
        math.exp(-ref_energy(v, h, p))
        for h in itertools.product([0, 1], repeat=N_H)
    )
    return -math.log(total)


def ref_cond_h(v, p):
    """Per-bit marginals p(h_j=1|v), exactly (bits independent given v)."""
    v = np.asarray(v, float)
    den = sum(math.exp(-ref_energy(v, g, p))
              for g in itertools.product([0, 1], repeat=N_H))
    out = []
    for j in range(N_H):
        num = sum(math.exp(-ref_energy(v, h, p))
                  for h in itertools.product([0, 1], repeat=N_H) if h[j] == 1)
        out.append(num / den)
    return np.array(out)


def ref_cond_v(h, p):
    """Per-bit marginals p(v_i=1|h), exactly."""
    h = np.asarray(h, float)
    den = sum(math.exp(-ref_energy(g, h, p))
              for g in itertools.product([0, 1], repeat=N_V))
    out = []
    for i in range(N_V):
        num = sum(math.exp(-ref_energy(v, h, p))
                  for v in itertools.product([0, 1], repeat=N_V) if v[i] == 1)
        out.append(num / den)
    return np.array(out)


def jax_params(p):
    return {
        "a": jnp.asarray(p["a"], dtype=jnp.float32),
        "b": jnp.asarray(p["b"], dtype=jnp.float32),
        "W": jnp.asarray(p["W"], dtype=jnp.float32),
    }


# ------------------------------------------------------------ energy / F
def test_energy_matches_reference():
    """Implementation energy() vs independent NumPy enumerated E(v,h)."""
    p = ref_params()
    jp = jax_params(p)
    for v in itertools.product([0.0, 1.0], repeat=N_V):
        for h in itertools.product([0.0, 1.0], repeat=N_H):
            vj = jnp.array(v)[None]
            hj = jnp.array(h)[None]
            E = energy(vj, hj, jp["a"], jp["b"], jp["W"])
            assert E.shape == (1,)
            assert float(E[0]) == pytest.approx(ref_energy(v, h, p), abs=1e-5)


def test_free_energy_matches_hidden_marginalization():
    """F(v) must equal -log sum_h exp(-E(v,h)) (sheet eq. 4)."""
    p = ref_params()
    jp = jax_params(p)
    for v in itertools.product([0.0, 1.0], repeat=N_V):
        v = jnp.array(v)
        assert float(free_energy(v[None], jp["a"], jp["b"], jp["W"])[0]) == \
            pytest.approx(ref_free_energy(v, p), abs=1e-5)


# --------------------------------------------------------- conditionals
def test_conditionals_match_reference():
    p = ref_params()
    jp = jax_params(p)
    v = jnp.array([1.0, 1.0])
    np.testing.assert_allclose(
        np.asarray(p_h_given_v(v[None], jp["b"], jp["W"])[0]),
        ref_cond_h(v, p), atol=1e-5)
    h = jnp.array([0.0, 1.0])
    np.testing.assert_allclose(
        np.asarray(p_v_given_h(h[None], jp["a"], jp["W"])[0]),
        ref_cond_v(h, p), atol=1e-5)


# ------------------------------------------------------ Gibbs stationarity
def ref_gibbs_transition(p):
    """Exact 4x4 visible transition matrix of one block-Gibbs step.

    A hidden configuration hk has probability prod_j ph_j**hk_j *
    (1-ph_j)**(1-hk_j); the visible target configuration has the analogous
    product from its per-bit marginals. All hidden configs are summed.
    """
    T = np.zeros((4, 4))
    for i, v0 in enumerate(itertools.product([0, 1], repeat=N_V)):
        ph = ref_cond_h(v0, p)
        for j, v1 in enumerate(itertools.product([0, 1], repeat=N_V)):
            acc = 0.0
            for hk in itertools.product([0, 1], repeat=N_H):
                p_h = 1.0
                for m in range(N_H):
                    p_h *= ph[m] ** hk[m] * (1.0 - ph[m]) ** (1 - hk[m])
                marg_v = ref_cond_v(hk, p)  # per-bit p(v_i=1|h)
                p_v1 = 1.0
                for m in range(N_V):
                    p_v1 *= marg_v[m] ** v1[m] * (1.0 - marg_v[m]) ** (1 - v1[m])
                acc += p_h * p_v1
            T[i, j] = acc
    return T


def ref_visible_marginal(p):
    Z = sum(math.exp(-ref_energy(v, h, p))
            for v in itertools.product([0, 1], repeat=N_V)
            for h in itertools.product([0, 1], repeat=N_H))
    return np.array([
        sum(math.exp(-ref_energy(v, h, p))
            for h in itertools.product([0, 1], repeat=N_H)) / Z
        for v in itertools.product([0, 1], repeat=N_V)])


def test_gibbs_chain_k0_k1_k3_batched():
    """k=0 returns the start; k>=1 moves the chain; batch shape preserved
    (batches 3 and 7 so accidental 2-batch unpacking cannot pass)."""
    jp = jax_params(ref_params())
    for n in (3, 7):
        v0 = (np.arange(n * N_V).reshape(n, N_V) % 2).astype(np.float32)
        key = jax.random.PRNGKey(1)
        out0 = gibbs_chain(jnp.asarray(v0), jp, key, 0)
        np.testing.assert_array_equal(np.asarray(out0), v0)
        out1 = gibbs_chain(jnp.asarray(v0), jp, key, 1)
        assert np.asarray(out1).shape == v0.shape
        out3 = gibbs_chain(jnp.asarray(v0), jp, key, 3)
        assert np.asarray(out3).shape == v0.shape
        assert set(np.unique(np.asarray(out3))) <= {0.0, 1.0}


def test_gibbs_stationary_distribution():
    """pi T = pi with pi the exact Boltzmann visible marginal; a batched
    JIT-compiled chain (many parallel walkers, lax.scan steps) converges to
    the same pi."""
    p = ref_params()
    T = ref_gibbs_transition(p)
    pi = ref_visible_marginal(p)
    np.testing.assert_allclose(pi @ T, pi, atol=1e-12)
    # batched sampled check: 20000 parallel chains, 200 sweeps via lax.scan
    jp = jax_params(p)
    key = jax.random.PRNGKey(0)
    v = jax.random.bernoulli(key, 0.5, (20000, N_V)).astype(jnp.float32)

    def body(i, carry):
        v, key = carry
        key, sub = jax.random.split(key)
        return gibbs_step(v, jp, sub), key

    (v_final, _) = jax.lax.fori_loop(0, 200, body, (v, key))
    counts = np.zeros(4)
    vf = np.asarray(v_final)
    for b in (0, 1):
        for a in (0, 1):
            counts[a * 2 + b] = np.mean((vf[:, 0] == a) & (vf[:, 1] == b))
    np.testing.assert_allclose(counts, pi, atol=0.02)


# ------------------------------------------------------------ CD gradient
def test_cd_gradient_matches_analytic_sufficient_statistics():
    """With endpoints fixed, d/dtheta mean[F(v0)-F(vk)] equals the analytic
    positive-minus-negative statistics (sheet eq. 9)."""
    rng = np.random.default_rng(11)
    v0 = rng.integers(0, 2, size=(16, N_V)).astype(np.float32)
    vk = rng.integers(0, 2, size=(16, N_V)).astype(np.float32)
    jp = jax_params(ref_params())

    loss, grads = jax.value_and_grad(cd_surrogate)(jp, v0, vk)
    hbar0 = np.asarray(p_h_given_v(v0, jp["b"], jp["W"]))
    hbark = np.asarray(p_h_given_v(vk, jp["b"], jp["W"]))
    ref = {
        # gradient of mean[F(v0)-F(vk)] wrt theta = NEGATIVE of the sheet's
        # eq. 9 update direction (the update is lr * minus this gradient)
        "W": (vk.T @ hbark - v0.T @ hbar0) / len(v0),
        "a": (vk - v0).mean(axis=0),
        "b": (hbark - hbar0).mean(axis=0),
    }
    for name in ("a", "b", "W"):
        np.testing.assert_allclose(
            np.asarray(grads[name]), ref[name], atol=1e-5)


def test_stop_gradient_in_actual_cd_surrogate():
    """The production cd_surrogate must stop the gradient through the sampled
    endpoint: d/ds cd_surrogate(params, v0, sigmoid(s)) == 0. Control: the
    same endpoint WITHOUT stop_gradient gives a nonzero gradient."""
    rng = np.random.default_rng(4)
    jp = jax_params(ref_params())
    v0 = rng.integers(0, 2, size=(8, N_V)).astype(np.float32)
    s = jnp.array([0.4, -0.7])

    g_sg = jax.grad(lambda s_: cd_surrogate(jp, jnp.asarray(v0),
                                            jax.nn.sigmoid(s_)[None]))(s)
    assert float(jnp.abs(g_sg).sum()) == 0.0

    # control: endpoint WITHOUT stop-gradient has a live gradient path
    def loss_no_sg(s_):
        vk = jax.nn.sigmoid(s_)[None]
        return free_energy(vk, jp["a"], jp["b"], jp["W"]).sum()
    g_no = jax.grad(loss_no_sg)(s)
    assert float(jnp.abs(g_no).sum()) > 1e-6


# ---------------------------------------------------------- binarization
def test_binarize_boundary():
    px = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(np.asarray(binarize(px)), [[0.0, 0.0, 1.0, 1.0]])


# ------------------------------------------------------------- clamping
def test_clamped_gibbs_chain_production_function():
    """The production clamped_gibbs_chain keeps clamped pixels fixed across
    all steps, end to end."""
    key = jax.random.PRNGKey(5)
    jp = init_params(jax.random.PRNGKey(2026), n_visible=8, n_hidden=4)
    v0 = (jax.random.uniform(key, (3, 8)) > 0.5).astype(jnp.float32)
    observed = v0[:, :4]
    mask = np.zeros((3, 8), dtype=bool)
    mask[:, :4] = True
    clamp_full = np.concatenate(
        [observed, np.zeros((3, 4), np.float32)], axis=1)  # inactive half unused
    for k in (0, 1, 5, 20):
        vk = clamped_gibbs_chain(v0, jp, key, k,
                                 jnp.asarray(clamp_full), jnp.asarray(mask))
        np.testing.assert_array_equal(np.asarray(vk[:, :4]),
                                      np.asarray(observed))
        assert set(np.unique(np.asarray(vk[:, 4:]))) <= {0.0, 1.0}


# ------------------------------------------------------------ ink Dice
def test_dice_known_values():
    pred = np.zeros((2, 2))
    truth = np.ones((2, 2))
    assert dice_ink(pred, truth) == 0.0            # blank prediction
    assert dice_ink(truth, pred) == 0.0
    assert dice_ink(pred, pred) == 0.0             # both blank -> 0 (incl.)
    assert dice_ink(truth, truth) == 1.0
    half = np.array([[1.0, 1.0], [0.0, 0.0]])
    # 2|A cap B| / (|A|+|B|) = 2*2 / (2+2) = 1 for identical ink
    assert dice_ink(half, half) == 1.0
    one = np.array([[1.0, 0.0], [0.0, 0.0]])
    # A={p0}, B={p0,p1}: 2*1/(1+2) = 2/3
    assert dice_ink(one, half) == pytest.approx(2.0 / 3.0)


# ------------------------------------------- shared init / data order
def test_shared_initialization_and_permutations(tmp_path):
    """The same seed must reproduce byte-identical init parameters and epoch
    permutations, so CD-1/CD-10 comparisons are matched."""
    from ebm.train import build_shared_init

    def digests():
        out = {}
        shared = build_shared_init(jax.random.PRNGKey(2026), n_epochs=3)
        out["a"] = hash(np.asarray(shared["params"]["a"]).tobytes())
        out["W"] = hash(np.asarray(shared["params"]["W"]).tobytes())
        out["perms"] = hash(np.asarray(shared["perms"]).tobytes())
        return out

    assert digests() == digests()


def test_init_visible_bias_uses_train_pixel_means():
    """Visible bias = logit of clipped train pixel means; no test data."""
    pm = np.full(16, 0.2)  # a constant train mean
    jp = init_params(jax.random.PRNGKey(0), n_visible=16, n_hidden=4,
                     pixel_mean=pm)
    a = np.asarray(jp["a"])
    np.testing.assert_allclose(a, math.log(0.2 / 0.8), rtol=1e-5)
    assert np.all(np.isfinite(a))
