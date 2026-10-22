"""Part 5 formal study driver: four test images x four seeds 7201-7204.

NOT executed during the engineering checkpoint; refuses to overwrite a
completed run. Truth (`inputs/images.npz` key 'test') is used ONLY for
evaluation outside the sampler; the sampler receives observations (zero at
hidden) and the mask. Batch/key convention: for each seed, root =
PRNGKey(seed) is split into (k_initial, k_carry) inside repaint_sample and
all draws are full [B=4,C=3,32,32] arrays for that 4-image batch; the same
compiled core (module cache) is reused across seeds.
"""

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.repaint import OFFICIAL_LEVELS, make_hole_mask, repaint_events, \
    repaint_sample

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "artifacts/part5"
SEEDS = [7201, 7202, 7203, 7204]


def param_hash(weights):
    h = hashlib.sha256()
    for k in sorted(weights):
        h.update(k.encode())
        h.update(np.asarray(weights[k]).tobytes())
    return h.hexdigest()


def main():
    if RAW.exists() and any(RAW.iterdir()):
        raise SystemExit(f"refusing to overwrite populated run directory: "
                         f"{RAW} (use a fresh attempt directory)")
    RAW.mkdir(parents=True, exist_ok=True)
    from src.cifar_unet import load_cifar_unet
    weights = load_cifar_unet(BASE / "inputs/cifar-model/"
                              "diffusion_pytorch_model.safetensors",
                              BASE / "inputs/cifar-model/config.json")
    hash_before = param_hash(weights)

    data = np.load(BASE / "inputs/images.npz")
    truth = data["test"]                      # (4,3,32,32) in [-1,1]
    assert truth.shape == (4, 3, 32, 32)
    mask = np.asarray(make_hole_mask(truth.shape))
    obs = truth * mask                        # zero at hidden entries
    assert (obs[..., 6:26, 6:26] == 0).all()
    np.savez(RAW / "inputs_used.npz", truth=truth, observations=obs,
             mask=mask, test_ids=data["test_ids"], labels=data["labels"])

    from src.diffusion import abar_from_betas, beta_schedule
    betas = np.asarray(beta_schedule())
    abar = np.asarray(abar_from_betas(jnp.asarray(betas)))
    events = repaint_events(OFFICIAL_LEVELS, attempts=2)

    completions = np.zeros((4, 4, 3, 32, 32), dtype=np.float32)
    event_states = np.zeros((4, 99, 4, 3, 32, 32), dtype=np.float32)
    initial_states = np.zeros((4, 4, 3, 32, 32), dtype=np.float32)
    key_arrays = np.zeros((4, 2, 2), dtype=np.uint32)  # [seed, (init,carry)]
    t0 = time.perf_counter()
    for si, seed in enumerate(SEEDS):
        key = jax.random.PRNGKey(seed)
        k_init, k_carry = jax.random.split(key)
        key_arrays[si, 0] = np.asarray(k_init, dtype=np.uint32)
        key_arrays[si, 1] = np.asarray(k_carry, dtype=np.uint32)
        initial_states[si] = np.asarray(
            jax.random.normal(k_init, obs.shape, dtype=jnp.float32))
        x0, xs = repaint_sample(weights, obs, mask, betas, abar,
                                OFFICIAL_LEVELS, attempts=2, key=key,
                                record_states=True)
        completions[si] = np.asarray(x0)
        event_states[si] = np.asarray(xs)
        print(f"seed {seed} done", flush=True)
    wall = time.perf_counter() - t0
    np.save(RAW / "completions_4seeds.npy", completions)
    np.save(RAW / "event_states_4seeds.npy", event_states)
    np.save(RAW / "initial_states_4seeds.npy", initial_states)
    np.save(RAW / "keys_4seeds.npy", key_arrays)
    np.savez(RAW / "events.npz",
             t=np.array([e["t"] for e in events], dtype=np.int32),
             s=np.array([e["s"] for e in events], dtype=np.int32),
             is_retry=np.array([e["is_retry"] for e in events]),
             betas=betas, abar=abar, seeds=np.array(SEEDS))

    # metrics in float64 for stable mean/variance/MSE (sampler stays f32)
    assert np.all(np.isfinite(completions)) and np.all(np.isfinite(event_states))
    hidden = mask == 0
    mses = np.zeros((4, 4))
    obs_changes = np.zeros((4, 4))
    c64 = completions.astype(np.float64)
    t64 = truth.astype(np.float64)
    m64 = mask.astype(np.float64)
    for si in range(4):
        for ii in range(4):
            err = (c64[si, ii] - t64[ii])[hidden[ii]]
            mses[si, ii] = float((err ** 2).sum() / (400 * 3))
            obs_changes[si, ii] = float(
                np.abs((c64[si, ii] - t64[ii])[~hidden[ii]]).max())
    assert np.all(obs_changes < 1e-5), obs_changes.max()
    for ii in range(4):
        # observed values identical across seeds, zero variance outside hole
        assert np.all(c64[:, ii][:, ~hidden[ii]] ==
                      c64[0, ii][~hidden[ii]])
        assert float(c64[:, ii][:, ~hidden[ii]].var(axis=0).max()) == 0.0
        assert float(c64[:, ii][:, hidden[ii]].var(axis=0).max()) > 0.0
    mean_over_seeds = c64.mean(axis=0).astype(np.float32)  # population mean
    var_over_seeds = c64.var(axis=0, ddof=0).astype(np.float32)  # divisor 4
    assert np.all(var_over_seeds[..., ~hidden[0]] == 0.0)

    h_after = param_hash(weights)
    meta = {
        "seeds": SEEDS,
        "levels": OFFICIAL_LEVELS, "n_events": len(events),
        "n_retries": sum(1 for e in events if e["is_retry"]),
        "predictor_evaluations": len(events),
        "key_convention": ("PRNGKey(seed) split inside repaint_sample into "
                           "(k_initial, k_carry); carry advances per event "
                           "splitting into (k_f, k_xi, k_eta, k_next); all "
                           "draws full [4,3,32,32] batch arrays"),
        "variance_convention": ("pixelwise variance over the 4 seeds is the "
                                "population variance (divisor 4, ddof=0)"),
        "wall_time_s": wall,
        "weights_sha256_before": hash_before,
        "weights_sha256_after": h_after,
        "weights_unchanged": hash_before == h_after,
        "jax_version": jax.__version__,
        "hidden_mse_sum_sq_over_400x3": mses.tolist(),
        "hidden_mse_average": float(mses.mean()),
        "observed_max_changes": obs_changes.tolist(),
        "normal_draws_per_seed": "298 drawn (1 initial + 99*3); "
                                 "248 used (1 initial + 99*2 + 49 f); "
                                 "50 unused f draws",
        "model_evaluations_per_image": 99,
        "forward_renoises": 49,
        "n_hidden_pixels_per_image": int(hidden[0].sum() // 3),
    }
    # file provenance hashes
    import hashlib
    def fsha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 20), b""):
                h.update(ch)
        return h.hexdigest()
    meta["file_hashes"] = {
        "config": fsha(BASE / "inputs/cifar-model/config.json"),
        "weights": fsha(BASE / "inputs/cifar-model/"
                        "diffusion_pytorch_model.safetensors"),
        "images_npz": fsha(BASE / "inputs/images.npz"),
        "repaint_source": fsha(BASE / "src/repaint.py"),
    }
    meta["versions"] = {"python": __import__("sys").version.split()[0],
                        "numpy": np.__version__,
                        "jax": jax.__version__,
                        "jaxlib": __import__("jaxlib").__version__}
    meta["expected_weights_hash"] = ("c35165b672140a1afa6b293b46553e81fd8d5"
                                     "6c9f4872fc8b9c1826afd1f6c04")
    assert h_after == meta["expected_weights_hash"], \
        f"weights hash changed: {h_after}"
    np.savez(RAW / "metrics.npz", mses=mses, mean=mean_over_seeds,
             variance=var_over_seeds)
    summary_keys = ("hidden_mse_average", "wall_time_s",
                    "weights_unchanged", "predictor_evaluations",
                    "n_retries")
    assert all(k in meta for k in summary_keys)
    assert meta["weights_unchanged"] and meta["weights_sha256_after"] == \
        meta["expected_weights_hash"]
    print(json.dumps({k: meta[k] for k in summary_keys}, indent=1))
    print("observed_max_changes:", meta["observed_max_changes"])
    # metadata written LAST, after all raw/metric writes and checks succeed
    with open(RAW / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
