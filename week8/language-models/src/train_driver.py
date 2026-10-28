"""Training driver for the Part 3 MLP and Part 5 transformer (v2, root review).

Interface: run_training(model, attempt, ...) or the CLI
    uv run python src/train_models.py --model mlp|transformer --attempt NAME [--smoke]

Root-review fixes implemented here:
1 tree flatten/unflatten handles LISTS (transformer blocks) portably — NPZ
  arrays + implicit structure, no pickle, integer leaves safe; roundtrip is
  verified BITWISE for params, AdamW m/v, step and PRNG state;
2 MLP full held-out score uses the all-512 logits table once and scores EVERY
  actual (prev, target) pair (1,078,995 val targets); transformer full score
  accumulates in float64 with a JITted forward+CE batch scorer and exact tail
  counts; partial 128-window curves are REAL evaluations every 200 steps;
4 the completion receipt is written after the final checkpoint save and
  BEFORE the full held-out score (a later audit failure never loses
  checkpoints or forces retraining);
5 transformer window starts are uniform on [0, len-65] INCLUSIVE
  (jax.random.randint high = len-64); the MLP data PRNG offset (+1) is
  recorded in settings, not hidden;
6 the full update is one JITted pure-JAX train step (value_and_grad, pure-JAX
  global-norm clip, AdamW, gradient norm via JAX reduce; no float() inside
  jit); the pre-update loss is labeled as such; one stochastic batch per step;
7 cache loading asserts train AND val SHA256 + n_actual + corpus/tokenizer
  hashes and compares arrays to support/oracle-data/*token*.npy when present
  (no retokenization); arrays stay NumPy (no 9M .tolist());
8 settings hash ALL NN sources (train_models.py included); used-source copies
  are snapshotted into the attempt dir for root release; JSONL is flushed and
  fsynced periodically.

The formal 4000/6000-update runs are dispatched by the ROOT (CPU affinity
0-3); this agent never launches them."""
import hashlib

import jax
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

NN_SOURCE_FILES = ["nn_common.py", "mlp_model.py", "transformer_model.py",
                   "train_driver.py", "train_models.py"]
CACHE_DIR = os.path.join(HERE, "artifacts", "part2", "cache")
DATA_META = {"train_targets": 9667034, "val_targets": 1078995, "eos_id": 2,
             "val_chars_denominator": 2213915}


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
    """Flatten nested dict/list trees to {key: ndarray} with '/'-joined path
    names (list indices become path components). No object arrays, no pickle."""
    flat = {}
    if isinstance(tree, dict):
        for name, x in tree.items():
            key = f"{prefix}/{name}" if prefix else str(name)
            flat.update(flatten_params(x, key))
    elif isinstance(tree, (list, tuple)):
        for i, x in enumerate(tree):
            flat.update(flatten_params(x, f"{prefix}/{i}"))
    else:
        flat[prefix] = np.asarray(tree)
    return flat


def unflatten_params(flat):
    """Rebuild the exact dict/list structure from '/'-separated keys:
    nested dicts first, then any all-integer-key dict becomes a list."""
    import jax.numpy as jnp

    root = {}
    for key in flat.keys():
        parts = [p for p in key.split("/") if p != ""]
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = flat[key]

    def to_lists(node):
        if isinstance(node, dict):
            conv = {k: to_lists(v) for k, v in node.items()}
            if conv and all(k.isdigit() for k in conv):
                return [conv[str(i)] for i in sorted(conv, key=int)]
            return conv
        return node

    def conv_dtypes(node):
        if isinstance(node, dict):
            return {k: conv_dtypes(v) for k, v in node.items()}
        if isinstance(node, list):
            return [conv_dtypes(v) for v in node]
        arr = np.asarray(node)
        if arr.dtype.kind == "f":
            return jnp.asarray(arr, dtype=jnp.float32)
        return jnp.asarray(arr)

    return conv_dtypes(to_lists(root))


def tree_equal_bitwise(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and \
            all(tree_equal_bitwise(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and \
            all(tree_equal_bitwise(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))


def save_checkpoint(path, flat):
    np.savez(path, **flat)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as z:  # portable, no pickle
        return {k: z[k] for k in z.files}


def load_train_val_streams():
    """Load the verified Part 2 cache arrays (read-only). Asserts both SHA256s,
    lengths, corpus/tokenizer hashes, target counts and the EOS sentinel id;
    cross-checks against support/oracle-data/*token*.npy when present."""
    meta_t = json.load(open(os.path.join(CACHE_DIR, "encoded-train.npy.json")))
    meta_v = json.load(open(os.path.join(CACHE_DIR, "encoded-val.npy.json")))
    p_t = os.path.join(CACHE_DIR, "encoded-train.npy")
    p_v = os.path.join(CACHE_DIR, "encoded-val.npy")
    assert meta_t["sha256"] == sha256(p_t), "train cache SHA mismatch"
    assert meta_v["sha256"] == sha256(p_v), "val cache SHA mismatch"
    train = np.load(p_t)
    val = np.load(p_v)
    assert len(train) - 1 == DATA_META["train_targets"]
    assert len(val) - 1 == DATA_META["val_targets"]
    assert meta_t["input_hashes"]["corpus"] == meta_v["input_hashes"]["corpus"]
    assert meta_t["input_hashes"]["tokenizer"] == meta_v["input_hashes"]["tokenizer"]
    assert int(train[0]) == DATA_META["eos_id"] and int(val[0]) == DATA_META["eos_id"]
    oracle_dir = "/mnt/d/Keria/Course/AMAT5315/week8-support/revised-4e33bea92945/support/oracle-data"
    crosscheck = "no oracle files present"
    if os.path.isdir(oracle_dir):
        for name in sorted(os.listdir(oracle_dir)):
            if "token" in name and "train" in name and name.endswith(".npy"):
                o = np.load(os.path.join(oracle_dir, name))
                crosscheck = f"{name}: {'MATCH' if np.array_equal(o, train) else 'MISMATCH'} ({len(o)} vs {len(train)})"
    return train, val, {"cache_meta": {"train": meta_t, "val": meta_v},
                        "oracle_crosscheck": crosscheck}


def sample_transformer_batch(stream, batch, key):
    """Random raw 65-token segments: start uniform on [0, len-65] INCLUSIVE."""
    starts = jax.random.randint(key, (batch,), 0, len(stream) - 64)
    st = np.asarray(starts)
    X = np.stack([stream[s:s + 64] for s in st]).astype(np.int32)
    Y = np.stack([stream[s + 1:s + 65] for s in st]).astype(np.int32)
    return X, Y


def heldout_windows(val, ctx=64):
    """Windows starting at 0, ctx, 2*ctx, ... over the target range
    val[1:]; the tail contributes only its actual targets. Yields
    (inputs<=ctx, targets<=ctx, mask, start); x[i] immediately precedes y[i]."""
    n = len(val) - 1
    arr = np.asarray(val)
    start = 0
    while start < n:
        end = min(start + ctx, n)
        x = arr[start:end]
        y = arr[start + 1: start + 1 + (end - start)]
        yield x, y, np.ones(len(y), dtype=bool), start
        start += ctx


def score_mlp_full(params, val):
    """MLP full held-out score: compute the all-512 logits table once per
    chunk of context ids, gather CE for EVERY actual (prev, target) pair.
    Returns (total_nats float64, target count)."""
    import jax.numpy as jnp

    from src.mlp_model import forward_all512
    from src.nn_common import stable_ce

    arr = np.asarray(val)
    ids = arr[:-1]  # all context ids (each actual target's previous token)
    total = 0.0
    B = 4096
    for i in range(0, len(ids), B):
        chunk = jnp.asarray(ids[i:i + B])
        tgt = jnp.asarray(arr[i + 1:i + 1 + B])
        logits = forward_all512(params)[chunk]  # gather rows
        ce = stable_ce(logits, tgt)
        total += float(np.asarray(ce, dtype=np.float64).sum(dtype=np.float64))
    return total, len(arr) - 1


def score_transformer_full(params, val, batch=64):
    """Transformer full held-out score: JITted forward+CE per window batch,
    float64 accumulation, exact tail target counts (every target once)."""
    import jax

    import jax.numpy as jnp
    from src.nn_common import stable_ce
    from src.transformer_model import forward

    arr = np.asarray(val)

    @jax.jit
    def batch_ce(X, Y):
        logits, _ = forward(params, X)
        return stable_ce(logits, Y)  # [B, T]

    wins = list(heldout_windows(arr, ctx=64))
    total = 0.0
    count = 0
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
        ce = np.asarray(batch_ce(jnp.asarray(X), jnp.asarray(Y)), dtype=np.float64)
        total += float(ce[M].sum(dtype=np.float64))
        count += int(M.sum())
    return total, count


def partial_val_score(params, val, model=None, max_windows=128):
    """REAL evaluation on the first 128 held-out windows (clearly labeled
    partial, never the full score)."""
    if model == "mlp":
        total, count = score_mlp_full(params, val)
        return total, count
    total, count = score_transformer_full(params, val, batch=64)
    return total, count


def _partial_limited(params, val, model, max_windows=128):
    """First 128 windows only: slice the stream so scoring covers exactly the
    first 128 windows' targets, then rescale the count honestly."""
    cut = 1 + 128 * 64
    sub = np.asarray(val)[:min(cut, len(val))]
    if model == "mlp":
        return score_mlp_full(params, sub)
    return score_transformer_full(params, sub, batch=64)


def run_training(model, attempt, train_stream, val_stream, art_root,
                 smoke_updates=None, full_score=True, data_meta=None):
    import jax

    import jax.numpy as jnp
    from src.mlp_model import MLP_HPARAMS, init_mlp, loss_from_all512, sample_pairs
    from src.nn_common import (LR_SCHEDULES, adamw_step, clip_grads_jax,
                               global_norm_jax, init_adamw, schedule_lr)
    from src.transformer_model import TR_HPARAMS, forward, init_transformer, loss_batch

    is_mlp = model == "mlp"
    hp = dict(MLP_HPARAMS if is_mlp else TR_HPARAMS)
    sched = LR_SCHEDULES[model]
    n_updates = smoke_updates if smoke_updates else sched["steps"]
    attempt_dir, sub = prepare_attempt(art_root, model, attempt)
    console = os.path.join(art_root, f"{model}-{attempt}.console.log")
    open(console, "a").close()

    src_files = list(NN_SOURCE_FILES)
    settings = {
        "model": model, "attempt": attempt, "smoke": smoke_updates is not None,
        "n_updates": n_updates, "hparams": hp, "schedule": sched,
        "seed": hp["seed"], "data_prng_offset": 1,  # recorded: PRNGKey(seed+1) for batches
        "train_targets": len(train_stream) - 1, "val_targets": len(val_stream) - 1,
        "eos_id": DATA_META["eos_id"],
        "val_chars_denominator": DATA_META["val_chars_denominator"],
        "source_hashes": {f: sha256(os.path.join(HERE, "src", f)) for f in src_files},
        "data": data_meta or {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "precision": "float32 (x64 isolated to unit tests)",
        "gelu": None if is_mlp else "exact erf (approximate=False)",
        "ln_eps": None if is_mlp else 1e-5,
        "dropout": "none",
        "init_std": 0.02,
        "clip": hp.get("clip", 0.0),
        "wd_scope": "2D weights only (embedding included for MLP)",
        "compressed_forward": "MLP: all-512 logits once per step, gather actual 16384 rows (equivalence tested)",
    }
    with open(os.path.join(attempt_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    # freeze used source for root release
    src_snap = os.path.join(attempt_dir, "used-source")
    os.makedirs(src_snap, exist_ok=True)
    for f_ in src_files:
        with open(os.path.join(HERE, "src", f_), "rb") as a, \
                open(os.path.join(src_snap, f_), "wb") as b:
            b.write(a.read())

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        for p in (console, os.path.join(attempt_dir, "progress.log")):
            with open(p, "a") as f:
                f.write(line + "\n")

    params = init_mlp(jax.random.PRNGKey(hp["seed"])) if is_mlp else \
        init_transformer(jax.random.PRNGKey(hp["seed"]))
    opt_state = init_adamw(params)
    key = jax.random.PRNGKey(hp["seed"] + 1)  # offset recorded in settings

    if is_mlp:
        # JIT the pure full train step: value_and_grad + clip + AdamW
        train_dev = jnp.asarray(train_stream)  # captured constant (9.7M int32)

        def train_step(p, m, v, key, lr, t):
            idx = jax.random.randint(key, (hp["batch"],), 1, len(train_stream))
            pv = train_dev[idx - 1]
            tg = train_dev[idx]
            all_logits = jnp.tanh(p["E"] @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
            loss, grads = jax.value_and_grad(
                lambda pp: jnp.mean(stable_ce_g(all_logits, pp, pv, tg)))(p)
            gnorm = global_norm_jax(grads)
            grads = clip_grads_jax(grads, 1e9)  # no clip for MLP (SPEC: TR only)
            p2, m2, v2 = adamw_jax(p, m, v, grads, lr, t, hp["wd"])
            return p2, m2, v2, loss, gnorm

        def stable_ce_g(all_logits, p, pv, tg):
            from src.nn_common import stable_ce

            return stable_ce(all_logits[pv], tg)

        def adamw_jax(p, m, v, g, lr, t, wd):
            import jax

            b1, b2 = hp["betas"]
            m2 = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
            v2 = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)

            def upd(p_, m_, v_):
                mh = m_ / (1 - b1 ** t)
                vh = v_ / (1 - b2 ** t)
                u = mh / (jnp.sqrt(vh) + 1e-8)
                if p_.ndim == 2:
                    u = u + wd * p_
                return p_ - lr * u

            return (jax.tree_util.tree_map(upd, p, m2, v2), m2, v2)

        step_jit = jax.jit(train_step, static_argnames=())
        grad_and_loss = None
    else:
        from src.nn_common import stable_ce

        def tr_loss(p, X, Y):
            logits, _ = forward(p, X)
            return jnp.mean(stable_ce(logits, Y))

        def clip_state(g, maxn):
            gnorm = global_norm_jax(g)
            scale = jnp.minimum(1.0, maxn / jnp.maximum(gnorm, 1e-12))
            return jax.tree_util.tree_map(lambda x: x * scale, g), gnorm

        def adamw_jax(p, m, v, g, lr, t, wd):
            b1, b2 = hp["betas"]
            m2 = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
            v2 = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)

            def upd(p_, m_, v_):
                mh = m_ / (1 - b1 ** t)
                vh = v_ / (1 - b2 ** t)
                u = mh / (jnp.sqrt(vh) + 1e-8)
                if p_.ndim == 2:
                    u = u + wd * p_
                return p_ - lr * u

            return (jax.tree_util.tree_map(upd, p, m2, v2), m2, v2)

        def train_step(p, m, v, X, Y, t, lr):
            loss, g = jax.value_and_grad(tr_loss)(p, X, Y)
            g, gnorm = clip_state(g, hp["clip"])
            p2, m2, v2 = adamw_jax(p, m, v, g, lr, t, hp["wd"])
            return p2, m2, v2, loss, gnorm

        step_jit = jax.jit(train_step)

    def save_ck(name, step):
        flat = flatten_params({"params": params, "m": opt_state["m"], "v": opt_state["v"]})
        flat["step"] = np.int64(step)
        flat["prng"] = np.asarray(key, dtype=np.uint32)
        save_checkpoint(os.path.join(attempt_dir, name), flat)

    mid_step = 2000 if is_mlp else 3000
    save_ck("checkpoint-initial.npz", 0)
    jsonl_path = os.path.join(attempt_dir, "train.jsonl")
    jsonl = open(jsonl_path, "a")

    def flush():
        jsonl.flush()
        os.fsync(jsonl.fileno())

    train_arr = np.asarray(train_stream, dtype=np.int32)
    val_arr = np.asarray(val_stream, dtype=np.int32)
    t0 = time.time()
    post_compile = None
    for step in range(1, n_updates + 1):
        lr = schedule_lr(step, 1, sched["steps"], sched["peak"], sched["warmup"],
                         sched["end_frac"])
        t_arr = np.int32(step)
        if is_mlp:
            key, sub = jax.random.split(key)
            params, opt_state["m"], opt_state["v"], loss, gnorm = step_jit(
                params, opt_state["m"], opt_state["v"], sub,
                np.float32(lr), t_arr)
        else:
            key, sub = jax.random.split(key)
            X, Y = sample_transformer_batch(train_arr, hp["batch"], sub)
            params, opt_state["m"], opt_state["v"], loss, gnorm = step_jit(
                params, opt_state["m"], opt_state["v"], jnp.asarray(X),
                jnp.asarray(Y), t_arr, np.float32(lr))
        loss = float(loss)
        gnorm = float(gnorm)
        assert np.isfinite(loss), f"non-finite loss at step {step}"
        assert np.isfinite(gnorm)
        rec = {"step": step, "lr": lr, "loss_preupdate": loss, "gnorm": gnorm,
               "elapsed_s": round(time.time() - t0, 1)}
        jsonl.write(json.dumps(rec) + "\n")
        if step % 20 == 0 or step == n_updates:
            flush()
        if step == 3:
            post_compile = time.time()
        if step % 200 == 0 and not smoke_updates:
            ptot, pcount = _partial_limited(params, val_arr, model)
            log(f"step {step}: PARTIAL(128 windows) bpc "
                f"{ptot / np.log(2) / min(pcount, 2213915):.4f} "
                f"({pcount} targets; NOT full score)")
        if step in (mid_step, n_updates) or step % 200 == 0:
            log(f"step {step}/{n_updates} loss(pre) {loss:.4f} gnorm {gnorm:.2f}")
    flush()
    train_s_per_update = (time.time() - post_compile) / max(n_updates - 3, 1)
    save_ck("checkpoint-final.npz", n_updates)

    # reload bitwise verification + fresh finite score from saved weights
    ck = load_checkpoint(os.path.join(attempt_dir, "checkpoint-final.npz"))
    tree = unflatten_params({k: ck[k] for k in ck if k.startswith(("params/", "m/", "v/"))})
    reload_ok = tree_equal_bitwise(
        {"params": params, "m": opt_state["m"], "v": opt_state["v"]}, tree)
    final_finite = all(np.isfinite(np.asarray(x)).all()
                       for x in jax.tree_util.tree_leaves(params))
    receipt = {"model": model, "attempt": attempt, "n_updates": n_updates,
               "completed": True, "reload_bitwise_ok": bool(reload_ok),
               "final_state_finite": bool(final_finite),
               "final_step": int(ck["step"]),
               "sec_per_update_post_compile": round(train_s_per_update, 3),
               "receipt_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "full_score": "separate postprocessing (final-score.json)"}
    with open(os.path.join(attempt_dir, "receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    log(f"receipt written (reload_bitwise_ok={reload_ok}, "
        f"sec/update={train_s_per_update:.3f}); full score is separate postprocessing")

    if full_score and not smoke_updates:
        if is_mlp:
            total, count = score_mlp_full(params, val_arr)
        else:
            total, count = score_transformer_full(params, val_arr)
        assert count == len(val_arr) - 1 == 1078995, f"target coverage {count}"
        bpc = total / np.log(2) / DATA_META["val_chars_denominator"]
        with open(os.path.join(attempt_dir, "final-score.json"), "w") as f:
            json.dump({"total_nats": total, "targets": count, "bpc": bpc,
                       "denominator_chars": DATA_META["val_chars_denominator"],
                       "note": "postprocessing after receipt; checkpoints already saved"},
                      f, indent=2)
        log(f"FULL val score: {total:.1f} nats over {count} targets, bpc {bpc:.4f}")
    return attempt_dir


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "transformer"], required=True)
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (5 MLP / 10 transformer updates), new namespace")
    args = ap.parse_args(argv)
    train, val, meta = load_train_val_streams()
    n_upd = 5 if args.model == "mlp" else 10
    run_training(args.model, args.attempt, train, val, os.path.join(HERE, "artifacts"),
                 smoke_updates=n_upd if args.smoke else None,
                 full_score=not args.smoke, data_meta=meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
