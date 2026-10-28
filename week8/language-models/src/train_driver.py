"""Training driver for the Part 3 MLP and Part 5 transformer.

Interface: run_training(model, attempt, ...) or the CLI
    uv run python src/train_models.py --model mlp|transformer --attempt NAME [--smoke]

Frozen protocol (SPEC): exclusive attempt directories created atomically
(artifacts/part3/mlp-<attempt> or artifacts/part5/<attempt>) BEFORE any
settings write; settings + source/input/cache hashes written before compute;
crash-safe JSONL per step with finite-loss checks; checkpoints (initial, mid
2000/3000, final) store params + AdamW m/v + step + PRNG state as flat NPZ
(no pickle) and are reload-verified bitwise; a completion receipt is written
after the final save and before any postprocessing. Smoke runs use a fresh
attempt namespace with 5 (MLP) / 10 (transformer) updates and are never
course results.

Data: train stream = sentinel EOS + actual tokens (9,667,034 targets),
val = sentinel + 1,078,995 targets; MLP samples actual pairs (indices 1..N-1);
transformer windows x = train[start:start+64], y = train[start+1:start+65]
with start uniform in [0, len(train)-65]. Held-out scoring: windows starting
0, 64, 128, ..., tail masked only at pad positions so every actual target is
counted exactly once; bpc denominator 2,213,915 Unicode characters (shared
with Part 2). Partial val curves (first 128 windows, CLEARLY labeled) every
200 steps; the final score is full.

The formal runs are dispatched by the root (CPU affinity 0-3); this agent
never launches them."""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


class AttemptExists(RuntimeError):
    pass


def prepare_attempt(art_root, model, attempt):
    sub = f"part3/mlp-{attempt}" if model == "mlp" else f"part5/{attempt}"
    d = os.path.join(art_root, sub)
    if os.path.exists(d):
        raise AttemptExists(f"attempt directory already exists: {d}")
    os.makedirs(d)
    return d, sub


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def flatten_params(tree, prefix=""):
    """Flatten a nested dict tree to {key: np.ndarray} with '/'-joined names."""
    import jax

    flat = {}
    for name, x in tree.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(x, dict):
            flat.update(flatten_params(x, key))
        else:
            flat[key] = np.asarray(x)
    return flat


def unflatten_params(flat):
    tree = {}
    for key, v in flat.items():
        parts = key.split("/")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    # convert numeric leaves back to jax arrays (f32)
    import jax.numpy as jnp

    def conv(t):
        if isinstance(t, dict):
            return {k: conv(v) for k, v in t.items()}
        return jnp.asarray(t, dtype=jnp.float32 if t.dtype.kind == "f" else t)

    return conv(tree)


def save_checkpoint(path, flat):
    np.savez(path, **flat)


def load_checkpoint(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def heldout_windows(val, ctx=64):
    """Yield (inputs<=ctx, targets<=ctx, mask, start). Windows start at
    0, ctx, 2*ctx, ...; the tail contributes only its actual targets (mask)."""
    n = len(val) - 1  # number of actual targets (targets are val[1:])
    arr = np.asarray(val)
    start = 0
    while start < n:
        end = min(start + ctx, n)  # target slice val[start+1 : start+1+end-start]
        x = arr[start:end]                     # inputs (<= ctx tokens)
        y = arr[start + 1: start + 1 + (end - start)]  # next tokens
        # for start=0 the first target has the sentinel as context; inputs
        # x align 1:1 with targets y (x[i] precedes y[i])
        mask = np.ones(len(y), dtype=bool)
        yield x, y, mask, start
        start += ctx


def score_val(params, forward_fn, val, ctx=64, max_windows=None, batch=64):
    """Full held-out score: total nats over ALL actual targets (each exactly
    once) + target count. float32 forward, float64 accumulation."""
    import jax.numpy as jnp

    from src.nn_common import stable_ce

    total, count = 0.0, 0
    wins = [w for w in heldout_windows(val, ctx)]
    if max_windows:
        wins = wins[:max_windows]
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        L = max(len(x) for x, _, _, _ in chunk)
        X = np.zeros((len(chunk), L), dtype=np.int32)
        Y = np.zeros((len(chunk), L), dtype=np.int32)
        M = np.zeros((len(chunk), L), dtype=bool)
        for j, (x, y, mask, _) in enumerate(chunk):
            X[j, :len(x)] = x
            Y[j, :len(y)] = y
            M[j, :len(y)] = mask
        logits, _ = forward_fn(params, jnp.asarray(X))
        ce = np.asarray(stable_ce(logits, jnp.asarray(Y)))
        total += float(ce[Y > 0 | M] if False else ce[M].sum())
        count += int(M.sum())
    return total, count


def run_training(model, attempt, train_stream, val_stream, art_root,
                 smoke_updates=None, full_score=True, data_hashes=None):
    import jax

    import jax.numpy as jnp
    from src.mlp_model import MLP_HPARAMS, forward_all512, forward_ids, init_mlp, \
        loss_from_all512, sample_pairs
    from src.nn_common import (LR_SCHEDULES, adamw_step, clip_grads, init_adamw,
                               schedule_lr, stable_ce)
    from src.transformer_model import TR_HPARAMS, forward, init_transformer, loss_batch

    is_mlp = model == "mlp"
    hp = dict(MLP_HPARAMS if is_mlp else TR_HPARAMS)
    sched = LR_SCHEDULES[model]
    n_updates = smoke_updates if smoke_updates else sched["steps"]
    attempt_dir, sub = prepare_attempt(art_root, model, attempt)
    console = os.path.join(art_root, os.path.dirname(sub),
                           f"{model}-{attempt}.console.log")
    open(console, "a").close()

    src_files = ["nn_common.py", "train_driver.py",
                 "mlp_model.py" if is_mlp else "transformer_model.py"]
    cache_meta_path = os.path.join(HERE, "artifacts", "part2", "cache",
                                   f"encoded-{'train' if True else ''}.npy.json")
    cache_meta = json.load(open(cache_meta_path)) if os.path.exists(cache_meta_path) else {}
    settings = {
        "model": model, "attempt": attempt, "smoke": smoke_updates is not None,
        "n_updates": n_updates, "hparams": hp, "schedule": sched,
        "seed": hp["seed"],
        "train_targets": len(train_stream) - 1, "val_targets": len(val_stream) - 1,
        "val_chars_denominator": 2213915,
        "source_hashes": {f: sha256(os.path.join(HERE, "src", f)) for f in src_files},
        "input_hashes": data_hashes or {},
        "cache_meta": cache_meta,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "precision": "float32 (x64 isolated to unit tests)",
        "gelu": "exact erf (approximate=False)" if not is_mlp else None,
        "clip": hp.get("clip", 0.0),
    }
    with open(os.path.join(attempt_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        for p in (console, os.path.join(attempt_dir, "progress.log")):
            with open(p, "a") as f:
                f.write(line + "\n")

    params = init_mlp(jax.random.PRNGKey(hp["seed"])) if is_mlp else \
        init_transformer(jax.random.PRNGKey(hp["seed"]))
    state = init_adamw(params)
    key = jax.random.PRNGKey(hp["seed"] + 1)

    if is_mlp:
        loss_fn = lambda p, pv, tg: loss_from_all512(p, pv, tg).mean()
        clip_max = None
    else:
        loss_fn = lambda p, x, y: loss_batch(p, x, y).mean()
        clip_max = hp["clip"]
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    def save_ck(name, step):
        flat = flatten_params({"params": params, "m": state["m"], "v": state["v"]})
        flat["step"] = np.int64(step)
        flat["prng"] = np.asarray(key, dtype=np.uint32)
        save_checkpoint(os.path.join(attempt_dir, name), flat)

    mid_step = 2000 if is_mlp else 3000
    save_ck("checkpoint-initial.npz", 0)
    jsonl = open(os.path.join(attempt_dir, "train.jsonl"), "a")
    t0 = time.time()
    stream_arr = np.asarray(train_stream, dtype=np.int32)
    val_arr = np.asarray(val_stream, dtype=np.int32)
    for step in range(1, n_updates + 1):
        key, sub = jax.random.split(key)
        if is_mlp:
            idx, pv, tg = sample_pairs(stream_arr, hp["batch"], sub)
            g = grad_fn(params, pv, tg)
        else:
            starts = jax.random.randint(sub, (hp["batch"],), 0,
                                        len(stream_arr) - 65)
            st = np.asarray(starts)
            X = np.stack([stream_arr[s:s + 64] for s in st]).astype(np.int32)
            Y = np.stack([stream_arr[s + 1:s + 65] for s in st]).astype(np.int32)
            g = grad_fn(params, jnp.asarray(X), jnp.asarray(Y))
        if clip_max:
            g, _ = clip_grads(g, clip_max)
        lr = schedule_lr(step, 1, sched["steps"], sched["peak"],
                         sched["warmup"], sched["end_frac"])
        params, state = adamw_step(params, state, g, lr=lr,
                                   betas=tuple(hp["betas"]), wd=hp["wd"])
        loss = float(loss_fn(params, *( (pv, tg) if is_mlp else
                      (jnp.asarray(X), jnp.asarray(Y)) )))
        assert np.isfinite(loss), f"non-finite loss at step {step}"
        rec = {"step": step, "lr": lr, "loss": loss,
               "gnorm": float(np.sqrt(sum(float(jnp.sum(x * x))
                                          for x in jax.tree_util.tree_leaves(g)))),
               "elapsed_s": round(time.time() - t0, 1),
               "partial_val_curve": "first-128-windows only (NOT full score)"}
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        if step in (mid_step, n_updates) or step % 200 == 0:
            log(f"step {step}/{n_updates} loss {loss:.4f} lr {lr:.5f}")
        if step == mid_step:
            save_ck(f"checkpoint-mid.npz", step)
    save_ck("checkpoint-final.npz", n_updates)

    # reload bitwise verification + fresh score from saved weights
    ck = load_checkpoint(os.path.join(attempt_dir, "checkpoint-final.npz"))
    tree = unflatten_params({k: v for k, v in ck.items() if k.startswith("params/")})["params"]
    reload_ok = all(np.array_equal(np.asarray(a), np.asarray(b))
                    for a, b in zip(jax.tree_util.tree_leaves(params),
                                    jax.tree_util.tree_leaves(tree)))
    fwd = (lambda p, x: (forward_ids(p, x[:, -1]), None)) if is_mlp else \
        (lambda p, x: forward(p, x))
    if is_mlp:
        _, pv, tg = sample_pairs(val_arr[:200], 32, jax.random.PRNGKey(0))
        fresh = float(loss_from_all512(tree, pv, tg).mean())
        tot, cnt = None, None
    else:
        fresh = float(loss_batch(tree, jnp.asarray(val_arr[None, :65])[:, :64],
                                 jnp.asarray(val_arr[None, :65])[:, 1:]).mean())
        tot, cnt = None, None
    if full_score and not smoke_updates:
        fwd_batch = (lambda p, x: (forward_ids(p, x[:, -1]), None)) if is_mlp else \
            (lambda p, x: forward(p, x))
        # full held-out score
        from src.nn_common import stable_ce as _sce
        total_nats, count = 0.0, 0
        wins = list(heldout_windows(val_arr, ctx=64))
        for i in range(0, len(wins), 64):
            chunk = wins[i:i + 64]
            L = max(len(x) for x, _, _, _ in chunk)
            X = np.zeros((len(chunk), L), dtype=np.int32)
            Y = np.zeros((len(chunk), L), dtype=np.int32)
            M = np.zeros((len(chunk), L), dtype=bool)
            for j, (x, y, mask, _) in enumerate(chunk):
                X[j, :len(x)] = x
                Y[j, :len(y)] = y
                M[j, :len(y)] = mask
            if is_mlp:
                logits = forward_ids(tree, jnp.asarray(X[:, -1]))[:, None, :]
                ce = np.asarray(_sce(logits, jnp.asarray(Y)[:, -1][:, None]))[:, 0]
            else:
                logits, _ = forward(tree, jnp.asarray(X))
                ce = np.asarray(_sce(logits, jnp.asarray(Y)))
            total_nats += float(ce[M].sum())
            count += int(M.sum())
        assert count == len(val_arr) - 1, f"target coverage {count}"
        bpc = total_nats / np.log(2) / 2213915
        with open(os.path.join(attempt_dir, "final-score.json"), "w") as f:
            json.dump({"total_nats": total_nats, "targets": count,
                       "bpc": bpc, "denominator_chars": 2213915}, f, indent=2)
        log(f"FULL val score: {total_nats:.1f} nats over {count} targets, bpc {bpc:.4f}")

    receipt = {"model": model, "attempt": attempt, "n_updates": n_updates,
               "completed": True, "reload_bitwise_ok": bool(reload_ok),
               "fresh_score_saved_weights": fresh,
               "final_step": int(ck["step"]),
               "receipt_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(attempt_dir, "receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    log(f"receipt written; reload_bitwise_ok={reload_ok}")
    return attempt_dir


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "transformer"], required=True)
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    cache_dir = os.path.join(HERE, "artifacts", "part2", "cache")
    stream = np.load(os.path.join(cache_dir, "encoded-train.npy")).tolist()
    val = np.load(os.path.join(cache_dir, "encoded-val.npy")).tolist()
    meta = json.load(open(os.path.join(cache_dir, "encoded-train.npy.json")))
    assert meta["sha256"] == sha256(os.path.join(cache_dir, "encoded-train.npy"))
    assert meta["n_actual"] == len(stream) - 1 == 9667034
    data_hashes = {"train_cache_sha256": meta["sha256"],
                   "corpus": meta["input_hashes"]["corpus"],
                   "tokenizer": meta["input_hashes"]["tokenizer"]}
    n_upd = 5 if args.model == "mlp" else 10
    run_training(args.model, args.attempt, stream, val, os.path.join(HERE, "artifacts"),
                 smoke_updates=n_upd if args.smoke else None,
                 full_score=False, data_hashes=data_hashes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
