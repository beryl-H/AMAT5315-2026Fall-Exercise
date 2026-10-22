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
    if (RAW / "run_meta.json").exists():
        raise SystemExit(f"refusing to overwrite completed run: {RAW}")
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

    betas = np.asarray(__import__("src.diffusion", fromlist=["beta_schedule"])
                       .beta_schedule())
    from src.diffusion import abar_from_betas
    abar = np.asarray(abar_from_betas(jnp.asarray(betas)))
    events = repaint_events(OFFICIAL_LEVELS, attempts=2)

    completions = np.zeros((4, 4, 3, 32, 32), dtype=np.float32)
    states_seed0 = None
    t0 = time.perf_counter()
    for si, seed in enumerate(SEEDS):
        key = jax.random.PRNGKey(seed)
        x0, xs = repaint_sample(weights, obs, mask, betas, abar,
                                OFFICIAL_LEVELS, attempts=2, key=key,
                                record_states=True)
        completions[si] = np.asarray(x0)
        if si == 0:
            states_seed0 = np.asarray(xs)     # (99,4,3,32,32) for replay
        print(f"seed {seed} done", flush=True)
    wall = time.perf_counter() - t0
    np.save(RAW / "completions_4seeds.npy", completions)
    np.save(RAW / "event_states_seed7201.npy", states_seed0)

    # metrics (evaluation OUTSIDE the sampler)
    hidden = mask == 0
    n_hidden = int(hidden.sum())             # 400*3*4 images per batch
    mses = np.zeros((4, 4))
    max_obs_change = 0.0
    for si in range(4):
        for ii in range(4):
            err = (completions[si, ii] - truth[ii])[hidden[ii]]
            mses[si, ii] = float((err ** 2).sum() / (400 * 3))
            max_obs_change = max(max_obs_change, float(
                np.abs((completions[si, ii] - truth[ii])[~hidden[ii]]).max()))
    mean_over_seeds = completions.mean(axis=0)               # population mean
    var_over_seeds = completions.var(axis=0, ddof=0)         # population /4

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
        "max_observed_change": max_obs_change,
        "n_hidden_pixels_per_image": int(hidden[0].sum() // 3),
    }
    with open(RAW / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    np.savez(RAW / "metrics.npz", mses=mses, mean=mean_over_seeds,
             variance=var_over_seeds)
    print(json.dumps({k: meta[k] for k in
                      ("hidden_mse_average", "max_observed_change",
                       "wall_time_s", "weights_unchanged",
                       "predictor_evaluations", "n_retries")}, indent=1))


if __name__ == "__main__":
    main()
