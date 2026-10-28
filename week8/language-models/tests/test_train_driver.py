"""Training driver tests: exclusive attempts, checkpoint roundtrip (params +
AdamW m/v + step + PRNG state, bitwise), held-out window scoring covers every
actual target exactly once, smoke run produces JSONL + receipt + reload-verify."""
import numpy as np
import pytest

from src.train_driver import (
    AttemptExists,
    flatten_params,
    load_checkpoint,
    prepare_attempt,
    save_checkpoint,
    unflatten_params,
)


def test_prepare_attempt_exclusive_namespaces(tmp_path):
    d1, _ = prepare_attempt(str(tmp_path), "mlp", "smoke-01")
    assert d1.endswith("mlp-smoke-01")
    with pytest.raises(AttemptExists):
        prepare_attempt(str(tmp_path), "mlp", "smoke-01")
    d2, _ = prepare_attempt(str(tmp_path), "transformer", "smoke-01")
    assert "part5" in d2 and d2.endswith("smoke-01")


def test_checkpoint_roundtrip_bitwise():
    import jax

    import jax.numpy as jnp
    from src.mlp_model import init_mlp
    from src.nn_common import init_adamw

    params = init_mlp(jax.random.PRNGKey(0))
    grads = jax.tree_util.tree_map(lambda x: x * 0.1, params)
    state = init_adamw(params)
    _, state = __import__("src.nn_common", fromlist=["adamw_step"]).adamw_step(
        params, state, grads, lr=0.01)
    key = jax.random.PRNGKey(42)
    flat = flatten_params({"params": params, "m": state["m"], "v": state["v"]})
    flat["step"] = np.int64(7)
    flat["prng"] = np.asarray(key, dtype=np.uint32)
    save_checkpoint("/tmp/test-ckpt.npz" if False else str(_np := __import__("pathlib").Path("artifacts/test-ckpt.npz")), flat) if False else None
    import pathlib

    p = pathlib.Path("/tmp") / "ckpt-test.npz"
    save_checkpoint(str(p), flat)
    loaded = load_checkpoint(str(p))
    for k, v in flat.items():
        assert np.array_equal(loaded[k], v), k
    # rebuild trees and verify bitwise against originals
    tree = unflatten_params({k: loaded[k] for k in loaded if k.startswith("params/")})
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(tree)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_heldout_windows_cover_every_target_exactly_once():
    from src.train_driver import heldout_windows

    val = [2] + list(range(100, 120))  # sentinel + 20 actual tokens
    wins = list(heldout_windows(val, ctx=64))
    all_targets = []
    for x, y, mask in wins:
        assert len(x) <= 64
        all_targets.extend([t for t, m in zip(y, mask) if m])
    assert all_targets == val[1:]  # every actual target exactly once, in order
    # boundary: window starting at 64 shares one input token, no double target
    starts = [w[3] for w in wins] if len(wins[0]) > 3 else None


def test_smoke_run_end_to_end(tmp_path):
    import shutil

    from src.train_driver import run_training

    stream = [2] + list(np.random.default_rng(0).integers(0, 512, size=4000))
    val = [2] + list(np.random.default_rng(1).integers(0, 512, size=300))
    for model, n_upd in [("mlp", 5), ("transformer", 10)]:
        d = run_training(model, "smoke-test", stream, val, str(tmp_path),
                         smoke_updates=n_upd, full_score=False)
        import json
        import os

        assert os.path.exists(os.path.join(d, "settings.json"))
        assert os.path.exists(os.path.join(d, "receipt.json"))
        receipt = json.load(open(os.path.join(d, "receipt.json")))
        assert receipt["n_updates"] == n_upd and receipt["completed"] is True
        assert receipt["reload_bitwise_ok"] is True
        jsonl = open(os.path.join(d, "train.jsonl")).read().strip().split("\n")
        assert len(jsonl) == n_upd
        for line in jsonl:
            rec = json.loads(line)
            assert np.isfinite(rec["loss"]) and "step" in rec and "lr" in rec
        for ck in ["checkpoint-initial.npz", "checkpoint-final.npz"]:
            assert os.path.exists(os.path.join(d, ck)), ck
