"""Task 2 tests: training loop mechanics (smoke-scale, same code paths)."""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import train as train_mod
from src.diffusion import abar_from_betas, beta_schedule, sample_target


def test_cosine_learning_rate_endpoints():
    eta1 = float(train_mod.cosine_learning_rate(1, 25000, 1e-3, 0.0))
    eta_end = float(train_mod.cosine_learning_rate(25000, 25000, 1e-3, 0.0))
    eta_mid = float(train_mod.cosine_learning_rate(12500, 25000, 1e-3, 0.0))
    assert eta1 == pytest.approx(1e-3, rel=1e-4)  # ~lr0 at j=1
    assert eta_end == pytest.approx(0.0, abs=1e-8)
    assert eta_mid == pytest.approx(0.5e-3, rel=1e-3)  # midpoint halves lr


def test_batch_loss_with_zero_output_is_one():
    """With zeroed weights the prediction is 0, so loss = E[0.5||eps||^2] = 1."""
    params = train_mod.__dict__ and __import__("src.network", fromlist=["init_params"]).init_params(jax.random.PRNGKey(0))
    # zero all weights so predict outputs 0
    params = [{"w": jnp.zeros_like(p["w"]), "b": jnp.zeros_like(p["b"])}
              for p in params]
    key = jax.random.PRNGKey(3)
    x0 = sample_target(key, 200000)
    k1, k2 = jax.random.split(key)
    eps = jax.random.normal(k1, (200000, 2), dtype=jnp.float32)
    t = jax.random.randint(k2, (200000,), 1, 1001)
    abar = abar_from_betas(beta_schedule())
    loss = float(train_mod.batch_loss(params, x0, eps, t, abar))
    assert loss == pytest.approx(1.0, abs=0.02)


def test_smoke_train_run_saves_and_is_deterministic(tmp_path):
    run1 = str(tmp_path / "run1")
    run2 = str(tmp_path / "run2")
    kwargs = dict(seed=2026, updates=20, batch_size=64, t_levels=1000,
                  checkpoint_updates=(0, 5, 10, 20), loss_every=1)
    train_mod.train(run1, **kwargs)
    hist1 = np.load(os.path.join(run1, "loss_history.npz"))
    assert len(hist1["steps"]) == 20
    assert np.all(np.isfinite(hist1["loss"]))
    assert hist1["steps"][0] == 1 and hist1["steps"][-1] == 20
    for step in (0, 5, 10, 20):
        assert os.path.exists(os.path.join(run1, "checkpoints",
                                           f"params_{step:05d}.npz"))
    with open(os.path.join(run1, "metadata.json")) as f:
        meta = json.load(f)
    assert meta["seed"] == 2026 and meta["precision"] == "float32"
    assert meta["batch_size"] == 64
    assert meta["indexing"] == ("betas[t-1] (len T); abar[t] (len T+1); "
                                "features (t-1)*omega")
    # checkpoint arrays are finite
    ck = np.load(os.path.join(run1, "checkpoints", "params_00020.npz"))
    for name in ck.files:
        assert np.all(np.isfinite(ck[name]))
    # determinism: same seed reproduces the same loss history
    train_mod.train(run2, **kwargs)
    hist2 = np.load(os.path.join(run2, "loss_history.npz"))
    np.testing.assert_allclose(hist1["loss"], hist2["loss"], rtol=1e-6)


def test_smoke_training_reduces_loss():
    """A short run must already push the loss below the 1.0 start."""
    run = "/tmp/train_smoke_reduce"  # retained, never deleted
    train_mod.train(run, seed=2026, updates=200, batch_size=256,
                    checkpoint_updates=(200,))
    hist = np.load(os.path.join(run, "loss_history.npz"))
    assert hist["loss"][-1] < 0.9
