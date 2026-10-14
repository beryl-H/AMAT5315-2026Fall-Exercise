"""Part 2 training driver: CD-1 and CD-10 on MNIST with matched init/order.

Writes git-ignored artifacts under artifacts/part2/ (shared init, per-run
parameters at all 21 recon checks, per-run metrics) and a compact tracked
metrics file evidence/part2_training_metrics.json after both runs finish.
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ebm.data import load_mnist_pair
from ebm.rbm import binarize
from ebm.train import (build_shared_init, epoch_batches, make_epoch_fn,
                       recon_hamming, sha256_array)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts/part2"
SETTINGS = dict(n_hidden=128, n_epochs=20, batch_size=100, lr=0.1, seed=2026,
                binarize_threshold=128, n_train=60000, n_test=10000,
                visible_bias_init="logit of train pixel mean clipped to [1e-3, 1-1e-3]",
                hidden_bias_init="zeros", W_init="N(0, 0.01^2)",
                chain_starts="each training batch (no persistent CD)",
                optimizer="plain SGD",
                # Actual executed key formula: initial parameters and per-epoch
                # data permutations are EXACTLY shared (seed 2026, sha256 in
                # artifacts/part2/shared_hashes.json); per-model training step
                # keys are split(jax.random.PRNGKey(2026), 2)[0] then
                # fold_in(_, k) and split(n_epochs+1), so the two models'
                # STOCHASTIC STREAMS DIFFER while init and data order match.
                step_keys="split(split(PRNGKey(2026))[0] folded in with k, 21); per-model streams differ",  # noqa: E501


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="permit writing into an existing artifacts/part2 "
                         "run directory (default: refuse)")
    args = ap.parse_args()
    if (ART / "shared.npz").exists() and not args.allow_overwrite:
        raise SystemExit(
            "refusing to overwrite existing artifacts/part2 checkpoints; "
            "pass --allow-overwrite only if you intend to redo the run")
    train_images, train_labels = load_mnist_pair(
        ROOT / "inputs/mnist/train-images-idx3-ubyte.gz",
        ROOT / "inputs/mnist/train-labels-idx1-ubyte.gz")
    test_images, _ = load_mnist_pair(
        ROOT / "inputs/mnist/t10k-images-idx3-ubyte.gz",
        ROOT / "inputs/mnist/t10k-labels-idx1-ubyte.gz")
    train = binarize(train_images.reshape(-1, 784))
    test = binarize(test_images.reshape(-1, 784))
    pixel_mean = np.asarray(train).mean(axis=0)  # TRAIN split only
    print(f"train pixel mean: min={pixel_mean.min():.4f} max={pixel_mean.max():.4f} "
          f"mean={pixel_mean.mean():.4f} (ink fraction)", flush=True)

    # shared init (deterministic in seed 2026); rebuild if absent
    ART.mkdir(parents=True, exist_ok=True)
    npz_path = ART / "shared.npz"
    if not npz_path.exists():
        shared = build_shared_init(
            jax.random.PRNGKey(SETTINGS["seed"]), n_epochs=SETTINGS["n_epochs"],
            n_items=SETTINGS["n_train"], n_hidden=SETTINGS["n_hidden"],
            pixel_mean=pixel_mean)
        params = {k: np.asarray(v) for k, v in shared["params"].items()}
        perms = np.asarray(shared["perms"])
        np.savez(npz_path, a=params["a"], b=params["b"], W=params["W"],
                 perms=perms)
        hashes = {k: (v if isinstance(v, list) else v) for k, v in shared["hashes"].items()}
        hashes["perms_all"] = sha256_array(perms)
        hashes["pixel_mean"] = sha256_array(pixel_mean)
        (ART / "shared_hashes.json").write_text(json.dumps(hashes, indent=2))
        master_key_bits = hashes["master_key_bits"]
    else:
        d = np.load(npz_path)
        params = {"a": d["a"], "b": d["b"], "W": d["W"]}
        perms = d["perms"]
        hashes = json.loads((ART / "shared_hashes.json").read_text())
        master_key_bits = hashes["master_key_bits"]
    print(f"shared init hashes: a={hashes['a'][:12]} W={hashes['W'][:12]} "
          f"perms={hashes['perms_all'][:12]}", flush=True)

    # step keys: derived once from the master key, shared by both runs
    step_key_root = jax.random.PRNGKey(SETTINGS["seed"])
    step_key_root, _ = jax.random.split(step_key_root)  # split off init key
    results = {"settings": SETTINGS, "hashes": hashes, "runs": {}}

    for k in (1, 10):
        p = {kk: jnp.asarray(vv) for kk, vv in params.items()}
        run_dir = ART / f"cd{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        epoch_fn = make_epoch_fn(k, SETTINGS["lr"], SETTINGS["batch_size"],
                                 SETTINGS["n_train"] // SETTINGS["batch_size"])
        key_seq = jax.random.split(jax.random.fold_in(step_key_root, k),
                                   SETTINGS["n_epochs"] + 1)
        epochs = []
        recon_key = key_seq[0]
        r0 = recon_hamming(test, p, recon_key)
        epochs.append({"epoch": 0, "recon_hamming": r0, "cd_surrogate": None,
                       "time_s": 0.0})
        print(f"CD-{k} init recon hamming: {r0:.4f}", flush=True)
        np.savez(run_dir / f"params_epoch{0:02d}.npz",
                 **{kk: np.asarray(vv) for kk, vv in p.items()})
        t_start = time.perf_counter()
        for e in range(SETTINGS["n_epochs"]):
            t0 = time.perf_counter()
            batched = epoch_batches(np.asarray(train), perms[e],
                                    SETTINGS["batch_size"])
            p, losses = epoch_fn(p, jnp.asarray(batched), key_seq[e + 1])
            r = recon_hamming(test, p, recon_key)
            s = float(np.mean(np.asarray(losses)))
            epochs.append({"epoch": e + 1, "recon_hamming": r,
                           "cd_surrogate": s,
                           "time_s": time.perf_counter() - t0})
            np.savez(run_dir / f"params_epoch{e + 1:02d}.npz",
                     **{kk: np.asarray(vv) for kk, vv in p.items()})
            print(f"CD-{k} epoch {e+1:2d}/20: surrogate={s:+.5f} "
                  f"recon={r:.4f} ({epochs[-1]['time_s']:.1f}s)", flush=True)
        total = time.perf_counter() - t_start
        results["runs"][f"cd{k}"] = {
            "k": k, "epochs": epochs, "total_time_s": total,
            "initial_recon": r0, "final_recon": epochs[-1]["recon_hamming"],
            "final_lt_half_initial": epochs[-1]["recon_hamming"] < 0.5 * r0,
        }
        (ROOT / "evidence/part2_training_metrics.json").write_text(
            json.dumps(results, indent=2))
        print(f"CD-{k} done in {total:.1f}s; final/init = "
              f"{epochs[-1]['recon_hamming'] / r0:.3f}", flush=True)

    print("PART2_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
