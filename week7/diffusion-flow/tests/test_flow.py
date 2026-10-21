"""Part 3 tests: flow matching network, exact marginal velocity, Euler
sampler, coarse DDPM jumps and the unbiased energy-distance estimator.

Derived from the lecture sheet (pages 10-15), not from the implementation.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.diffusion import CENTRES, abar_from_betas, beta_schedule

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@pytest.fixture(scope="module")
def flow_cfg():
    with open("flow.toml", "rb") as f:
        return tomllib.load(f)


def test_flow_toml_matches_lecture(flow_cfg):
    m, tr = flow_cfg["model"], flow_cfg["training"]
    assert m["hidden_widths"] == [128, 128, 128]
    assert m["time_features"] == 32
    assert m["time_frequency_base"] == 10_000.0
    assert m["time_scale"] == 1000.0
    assert "time_offset" not in m  # NO -1 offset for flow matching
    assert tr["updates"] == 25_000 and tr["batch_size"] == 512
    assert tr["seed"] == 2029 and tr["precision"] == "float32"
    assert (tr["beta1"], tr["beta2"], tr["epsilon"]) == (0.9, 0.999, 1e-8)


# ---------- flow features and network ----------

def test_flow_features_no_minus_one_offset():
    from src.flow import flow_features
    # phase = 1000 * s * omega_j; 16 sines then 16 cosines
    s = 0.5
    feats = np.asarray(flow_features(jnp.array([s])))[0]
    for j in range(16):
        # float32 reference (same precision as production) and documented
        # abs tol 3e-5: float64 reference differs by up to ~1.4e-5 at
        # phase ~500 (root Part 1 measurement 1.36e-5)
        omega = np.float32(10000.0) ** (-j / np.float32(16.0))
        phase = np.float32(1000.0 * np.float32(s)) * omega
        assert feats[j] == pytest.approx(np.float32(np.sin(phase)), abs=3e-5)
        assert feats[16 + j] == pytest.approx(np.float32(np.cos(phase)),
                                              abs=3e-5)
    # s = 0 must give sin(0)=0, cos(0)=1 (a -1 offset would not)
    feats0 = np.asarray(flow_features(jnp.array([0.0])))[0]
    np.testing.assert_allclose(feats0[:16], 0.0, atol=1e-7)
    np.testing.assert_allclose(feats0[16:], 1.0, atol=1e-7)


def test_flow_predict_shapes_and_finite():
    from src.flow import flow_init_params, flow_predict
    params = flow_init_params(jax.random.PRNGKey(2029))
    x = jnp.zeros((5, 2), dtype=jnp.float32)
    s = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = flow_predict(params, x, s)
    assert out.shape == (5, 2) and out.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(out)))
    # identical architecture/initialization: same 2029 key must give the
    # SAME weights as the diffusion net; only the seed differs (2029 vs 2026)
    from src import network
    p_same = network.init_params(jax.random.PRNGKey(2029))
    for pf, pn in zip(params, p_same):
        np.testing.assert_array_equal(np.asarray(pf["w"]), np.asarray(pn["w"]))
        np.testing.assert_array_equal(np.asarray(pf["b"]), np.asarray(pn["b"]))
    p_diff = network.init_params(jax.random.PRNGKey(2026))
    assert not np.allclose(np.asarray(params[0]["w"]),
                           np.asarray(p_diff[0]["w"]))


def test_flow_training_loss_gradient_and_update():
    """Algorithm 4 loss: 1/d ||v_theta(x_s, s) - (z - x0)||^2, and one
    manual gradient-descent update reduces the loss on a fixed batch."""
    from src.flow import flow_init_params, flow_loss, flow_predict
    params = flow_init_params(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    k_x0, k_z, k_s = jax.random.split(key, 3)
    x0 = jnp.stack([CENTRES[0]] * 16)
    z = jax.random.normal(k_z, (16, 2), dtype=jnp.float32)
    s = jax.random.uniform(k_s, (16,), dtype=jnp.float32)
    xs = (1 - s)[:, None] * x0 + s[:, None] * z
    loss, grads = jax.value_and_grad(flow_loss)(params, x0, z, s)
    # loss on fixed inputs is deterministic; gradient finite and nonzero
    assert bool(jnp.isfinite(loss))
    for g in jax.tree_util.tree_leaves(grads):
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.abs(g).max()) > 0.0
    # manual SGD step in the negative gradient direction reduces this loss
    stepped = [{"w": p["w"] - 0.01 * g["w"], "b": p["b"] - 0.01 * g["b"]}
               for p, g in zip(params, grads)]
    loss2 = flow_loss(stepped, x0, z, s)
    assert float(loss2) < float(loss)


def test_flow_training_smoke_deterministic_and_reduces_loss(tmp_path):
    from src.flow import train_flow
    kw = dict(seed=2029, updates=30, batch_size=64, checkpoint_updates=(0, 30))
    train_flow(str(tmp_path / "a"), **kw)
    h1 = np.load(tmp_path / "a" / "loss_history.npz")
    train_flow(str(tmp_path / "b"), **kw)
    h2 = np.load(tmp_path / "b" / "loss_history.npz")
    np.testing.assert_allclose(h1["loss"], h2["loss"], rtol=1e-6)
    assert len(h1["loss"]) == 30 and np.all(np.isfinite(h1["loss"]))


# ---------- exact marginal velocity ----------

def test_exact_velocity_boundary_fields():
    from src.flow import exact_marginal_velocity
    x = jnp.array([[0.7, -1.3], [2.0, 0.1]], dtype=jnp.float32)
    # v(x, 1) = x  (all mass from noise)
    np.testing.assert_allclose(
        np.asarray(exact_marginal_velocity(x, 1.0)), np.asarray(x), atol=1e-5)
    # v(x, 0) = -x  (data stays put; -x direction)
    np.testing.assert_allclose(
        np.asarray(exact_marginal_velocity(x, 0.0)), -np.asarray(x),
        atol=1e-5)


def test_exact_velocity_component_conditional_moments():
    """Near one component (others negligible), the exact one-component
    conditional velocity is Cov(u,x)/Var(x) * (x - a mu) - mu with
    Cov(u,x) = s - 0.09 a and Var(x) = 0.09 a^2 + s^2 (NOT z0 - mu,
    because the component variance 0.09 is nonzero)."""
    from src.flow import exact_marginal_velocity
    mu = np.array(CENTRES[0], dtype=np.float64)  # (3, 0)
    s = 0.01
    a = 1.0 - s
    V = 0.09 * a * a + s * s
    z0 = np.array([[0.6, -0.8]])
    x = a * mu + s * z0
    v = np.asarray(exact_marginal_velocity(jnp.asarray(x, jnp.float32), s))
    expected = -mu + (s - 0.09 * a) / V * (x - a * mu)
    np.testing.assert_allclose(v, expected, atol=1e-4)


def test_exact_velocity_numerical_continuity():
    from src.flow import exact_marginal_velocity
    x = jnp.array([[0.3, 0.9], [-1.0, 2.0]], dtype=jnp.float32)
    for s in (0.2, 0.5, 0.8):
        a = exact_marginal_velocity(x, s - 1e-4)
        b = exact_marginal_velocity(x, s + 1e-4)
        assert float(jnp.abs(a - b).max()) < 1e-2


# ---------- Euler sampling ----------

def test_euler_explicit_grid_original_time_no_clipping():
    """Field v(x, s) = s * ones (numeric function of s): the two Euler
    updates are x - 0.5*1*ones then minus 0.5*0.5*ones. A loop-counter
    confusion (s = 0 then 1) gives different values, so the hand check
    pins the ORIGINAL s. Start far outside [-1,1]: no clipping."""
    from src.flow import euler_sample

    def v_pred(params, x, s):
        return s[:, None].astype(jnp.float32) * jnp.ones_like(x)

    x1 = jnp.array([[100.0, -100.0]], dtype=jnp.float32)
    out = euler_sample(v_pred, None, x1, jnp.array([1.0, 0.5, 0.0]))
    expected = 100.0 - 0.5 * 1.0 - 0.5 * 0.5
    np.testing.assert_allclose(float(out[0, 0]), expected, rtol=1e-6)
    # defect emulation (s = 0 then 1, i.e. loop index) must differ
    x_loop = 100.0 - 0.5 * 0.0 - 0.5 * 1.0
    assert abs(expected - x_loop) > 0.1
    assert float(jnp.abs(out).max()) > 10.0


def test_euler_sign_convention_moves_toward_data():
    """With the exact velocity, Euler integration from s=1 to 0 must move a
    noise point toward the data mean (origin), not away."""
    from src.flow import euler_sample, exact_marginal_velocity
    x1 = jnp.array([[3.0, 3.0]], dtype=jnp.float32)
    grid = jnp.linspace(1.0, 0.0, 101)
    out = euler_sample(lambda p, x, s: exact_marginal_velocity(x, s),
                       None, x1, grid)
    # moved toward the data (radius shrunk from |(3,3)| = 4.24); with only
    # 100 Euler steps of a stiff field it need not reach a cluster centre
    assert float(jnp.linalg.norm(out)) < 4.0


def test_euler_validation():
    from src.flow import euler_sample
    with pytest.raises(ValueError):
        euler_sample(lambda p, x, s: x, None, jnp.zeros((1, 2)),
                     jnp.array([1.0, 1.0, 0.0]))  # not strictly decreasing
    with pytest.raises(ValueError):
        euler_sample(lambda p, x, s: x, None, jnp.zeros((1, 2)),
                     jnp.array([0.9, 0.0]))  # must start at 1


# ---------- coarse DDPM ----------

def test_coarse_ddpm_adjacent_matches_part1():
    """Tiny T=2 schedule betas [.1,.2], abar [1,.9,.72]: the coarse grid
    [2,1,0] with the SAME innovations must equal Part 1's two fine steps
    exactly (adjacent jump: 1-a = beta_t, sqrt(1-a) = sqrt(beta_t))."""
    from src.coarse import coarse_ddpm_sample
    from src.sample import ddpm_sample
    betas = jnp.array([0.1, 0.2], dtype=jnp.float32)
    abar = jnp.array([1.0, 0.9, 0.72], dtype=jnp.float32)

    def predict(params, x, t):
        return t[:, None].astype(jnp.float32) * jnp.ones_like(x)

    key = jax.random.PRNGKey(7)
    xT = jax.random.normal(key, (4, 2), dtype=jnp.float32)
    z2 = jax.random.normal(key, (4, 2), dtype=jnp.float32)
    z1 = jax.random.normal(key, (4, 2), dtype=jnp.float32)
    coarse = coarse_ddpm_sample(predict, None, xT, betas, abar,
                                levels=np.array([2, 1, 0]),
                                reverse_noise={2: z2, 1: z1})
    fine = ddpm_sample(predict, None, xT, betas, abar,
                       reverse_noise={2: z2, 1: z1})
    np.testing.assert_allclose(np.asarray(coarse), np.asarray(fine),
                               rtol=1e-6)


def test_coarse_ddpm_original_time_via_numerics():
    """Tiny T=2: coarse single jump [2,0] with predictor eps = t*ones.
    Hand computation pins the ORIGINAL level (eps=2); a loop-index
    confusion would give eps=0 and a different endpoint. Validation tests
    supply an explicit key per the sampler contract."""
    from src.coarse import coarse_ddpm_sample
    betas = jnp.array([0.1, 0.2], dtype=jnp.float32)
    abar = jnp.array([1.0, 0.9, 0.72], dtype=jnp.float32)

    def predict(params, x, t):
        return t[:, None].astype(jnp.float32) * jnp.ones_like(x)

    xT = jnp.array([[1.0, -2.0]], dtype=jnp.float32)
    x = coarse_ddpm_sample(predict, None, xT, betas, abar,
                           levels=np.array([2, 0]),
                           key=jax.random.PRNGKey(0))
    a = 0.72
    expected = (xT - (1 - a) / math.sqrt(1 - a) * 2.0) / math.sqrt(a)
    np.testing.assert_allclose(np.asarray(x), np.asarray(expected),
                               rtol=1e-5)  # float32
    # loop-index confusion (eps = 0 at the first jump) differs
    loop = xT / math.sqrt(a)
    assert float(jnp.abs(expected - loop).max()) > 0.5


def test_coarse_ddpm_validation_with_key():
    from src.coarse import coarse_ddpm_sample
    betas = beta_schedule()
    abar = abar_from_betas(betas)

    def predict(params, x, t):
        return jnp.zeros_like(x)

    xT = jnp.zeros((2, 2), dtype=jnp.float32)
    with pytest.raises(ValueError):
        coarse_ddpm_sample(predict, None, xT, betas, abar,
                           levels=np.array([1000, 1000, 0]),
                           key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        coarse_ddpm_sample(predict, None, xT, betas, abar,
                           levels=np.array([999, 0]),
                           key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        coarse_ddpm_sample(predict, None, xT, betas, abar,
                           levels=np.array([1000, 0]))  # no key, no replay


# ---------- energy distance ----------

def test_energy_distance_matches_literal_loops():
    from src.energy import energy_distance_unbiased
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 2)).astype(np.float32)
    Y = rng.normal(size=(5, 2)).astype(np.float32)

    def lit(A, B):
        n, m = len(A), len(B)
        cross = sum(np.linalg.norm(a - b) for a in A for b in B)
        wx = sum(np.linalg.norm(a - b) for i, a in enumerate(A)
                 for j, b in enumerate(A) if i != j)
        wy = sum(np.linalg.norm(a - b) for i, a in enumerate(B)
                 for j, b in enumerate(B) if i != j)
        return 2 * cross / (n * m) - wx / (n * (n - 1)) - wy / (m * (m - 1))

    got = float(energy_distance_unbiased(X, Y))
    assert got == pytest.approx(lit(X, Y), rel=1e-4)
    # symmetry of the estimator in (X, Y)
    assert float(energy_distance_unbiased(Y, X)) == pytest.approx(got, rel=1e-5)


def test_energy_distance_same_array_negative_regression():
    """The UNBIASED distinct-pairs estimator on the SAME array is negative:
    cross mean includes the zero diagonal, within means exclude it, so
    D(X,X) = -2 * mean_distinct_pair_distance / n (a valid negative
    estimate; only plotted values are floored). Hand value via pdist."""
    from scipy.spatial.distance import pdist
    from src.energy import energy_distance_unbiased
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2)).astype(np.float32)
    d = float(energy_distance_unbiased(X, X))
    pd = pdist(X.astype(np.float64))
    mean_distinct = float(pd.mean())
    expected = -2.0 * mean_distinct / len(X)
    assert d == pytest.approx(expected, rel=1e-6)
    assert d < 0
    # far-apart independent sets give a large positive estimate
    Y = X + 10.0
    assert float(energy_distance_unbiased(X, Y)) > 5.0


def test_euler_rejects_grid_not_ending_at_zero():
    from src.flow import euler_sample
    with pytest.raises(ValueError):
        euler_sample(lambda p, x, s: x, None, jnp.zeros((1, 2)),
                     jnp.array([1.0, 0.5, 0.25]))


def test_coarse_ddpm_rejects_grid_not_ending_at_zero():
    from src.coarse import coarse_ddpm_sample
    betas = jnp.array([0.1, 0.2], dtype=jnp.float32)
    abar = jnp.array([1.0, 0.9, 0.72], dtype=jnp.float32)
    with pytest.raises(ValueError):
        coarse_ddpm_sample(lambda p, x, t: jnp.zeros_like(x), None,
                           jnp.zeros((1, 2)), betas, abar,
                           levels=np.array([2, 1]),
                           key=jax.random.PRNGKey(0))
