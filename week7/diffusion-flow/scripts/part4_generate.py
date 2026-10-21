"""Part 4: image DDPM sampling with the fixed pretrained CIFAR predictor.

Same Algorithm 2 as Part 1 (betas linear 1e-4..0.02, abar[t] length 1001,
reverse variance beta[t-1], no noise at t=1, no clipping anywhere), but the
predictor is the 1000-step-evaluated image UNet receiving the MODEL INDEX
t-1. Implemented as ten chunks of 100 original levels: one compiled scan
body per chunk shape, real progress logs, identical 1000-step result to a
single scan. Keys: PRNGKey(7401) split into independent xT and noise keys;
all 1000 innovations prepared once from the noise key.
"""

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.cifar_unet import cifar_unet_predict
from src.diffusion import abar_from_betas, beta_schedule

BASE = Path(__file__).resolve().parent.parent
STAGE_LEVELS = [1000, 800, 600, 400, 200, 100, 50, 20, 0]
SEED = 7401
B = 4


def chunked_image_ddpm(weights, xT, z_stack, betas, abar, chunk=100,
                       progress=None, predict=cifar_unet_predict):
    """Chunked 1000-step image DDPM (Algorithm 2, Part 1 equations).

    weights/xT/z_stack/betas/abar are explicit JIT arguments (no 140 MB
    closure constants); one compiled run_chunk outside the Python loop.
    The predictor receives the MODEL INDEX t-1 for original level t
    (subtraction happens HERE, exactly once — never inside the predictor).

    xT: (B,C,H,W) at level T=len(betas). z_stack: (T,B,C,H,W) innovations,
    z_stack[i] used for the step FROM level T-i (ignored at t=1).
    Returns states (T+1,B,C,H,W): states[i] is the state at level T-i.
    """
    import numpy as _np
    T = int(_np.asarray(betas).shape[0])
    if int(_np.asarray(abar).shape[0]) != T + 1:
        raise ValueError(f"abar length {_np.asarray(abar).shape[0]} != T+1")
    if tuple(_np.asarray(z_stack).shape) != (T, *xT.shape):
        raise ValueError(f"z_stack shape {_np.asarray(z_stack).shape} != "
                         f"{(T, *xT.shape)}")
    abar_j = jnp.asarray(abar)
    betas_j = jnp.asarray(betas)
    z_all = jnp.asarray(z_stack)

    def run_chunk(w, x, ts, zs):
        """One compiled chunk: original levels ts (descending), innovations
        zs aligned so zs[i] belongs to the step FROM level ts[i]."""
        def body(x, inp):
            t, z = inp
            eps = predict(w, x, t - 1)
            beta_t = betas_j[t - 1]
            alpha_t = 1.0 - beta_t
            a_t = abar_j[t]
            mu = (x - beta_t / jnp.sqrt(1.0 - a_t) * eps) / jnp.sqrt(alpha_t)
            x_next = jnp.where(t > 1, mu + jnp.sqrt(beta_t) * z, mu)
            return x_next, x_next
        return jax.lax.scan(jax.jit(body), x, (ts, zs))

    run_chunk_jit = jax.jit(run_chunk, static_argnames=())
    x = xT
    states = [np.asarray(xT)]
    n_chunks = (T + chunk - 1) // chunk
    for c in range(n_chunks):
        ts = jnp.arange(T - c * chunk, max(T - (c + 1) * chunk, 0), -1)
        zc = z_all[c * chunk: c * chunk + ts.shape[0]]
        x, xs = run_chunk_jit(weights, x, ts, zc)
        # xs: (n_steps, B, C, H, W) — one state per step in this chunk
        for i in range(xs.shape[0]):
            states.append(np.asarray(xs[i]))
        if progress:
            progress(T - (c + 1) * chunk if (c + 1) * chunk < T else 0, x)
    assert len(states) == T + 1 and not states[-1].shape == ()
    return np.stack(states)


def param_hash(weights):
    h = hashlib.sha256()
    for k in sorted(weights):
        h.update(k.encode())
        h.update(np.asarray(weights[k]).tobytes())
    return h.hexdigest()


def main():
    from src.cifar_unet import load_cifar_unet
    raw_dir = BASE / "artifacts/part4/generation"
    if (raw_dir / "generation_meta.json").exists():
        raise SystemExit(f"refusing to overwrite completed generation: "
                         f"{raw_dir}/generation_meta.json exists")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (BASE / "evidence").mkdir(exist_ok=True)

    weights = load_cifar_unet(BASE / "inputs/cifar-model/"
                              "diffusion_pytorch_model.safetensors",
                              BASE / "inputs/cifar-model/config.json")
    hash_before = param_hash(weights)

    betas = beta_schedule()
    abar = abar_from_betas(betas)

    k_root = jax.random.PRNGKey(SEED)
    k_xT, k_noise = jax.random.split(k_root)
    xT = jax.random.normal(k_xT, (B, 3, 32, 32), dtype=jnp.float32)
    z_stack = jax.random.normal(k_noise, (1000, B, 3, 32, 32),
                                dtype=jnp.float32)
    np.savez(raw_dir / "seed_inputs.npz", seed=np.int64(SEED),
             start_key=np.asarray(k_xT, dtype=np.uint32),
             noise_key=np.asarray(k_noise, dtype=np.uint32),
             xT=np.asarray(xT))
    # raw innovations kept out of git but on disk for audit
    np.save(raw_dir / "noise_stack.npy", np.asarray(z_stack))

    logs = []

    def progress(level_done, x):
        logs.append({"level_reached": int(level_done),
                     "mean_abs": float(jnp.abs(x).mean())})
        print(f"level {level_done}: mean|x| {float(jnp.abs(x).mean()):.4f}",
              flush=True)

    t0 = time.perf_counter()
    states = chunked_image_ddpm(weights, xT, z_stack, betas, abar,
                                progress=progress)
    wall = time.perf_counter() - t0
    assert states.shape == (1001, B, 3, 32, 32)
    np.save(raw_dir / "states_1001.npy", states)

    h_after = param_hash(weights)
    meta = {
        "seed": SEED, "batch": B, "levels": "original t 1000..1, model index t-1",
        "reverse_variance": "beta[t-1]", "final_noise": "omitted at t=1",
        "clipping": "none", "total_predictor_evaluations": 1000,
        "wall_time_s": wall,
        "weights_sha256_before": hash_before,
        "weights_sha256_after": h_after,
        "weights_unchanged": hash_before == h_after,
        "jax_version": jax.__version__,
        "stage_levels": STAGE_LEVELS,
        "progress_log": logs,
    }
    with open(raw_dir / "generation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in
                      ("wall_time_s", "weights_unchanged",
                       "total_predictor_evaluations")}, indent=1))
    # selected-stage snapshots for figures
    np.savez(raw_dir / "stage_states.npz",
             **{f"t{lv}": states[1000 - lv] for lv in STAGE_LEVELS})


if __name__ == "__main__":
    main()
