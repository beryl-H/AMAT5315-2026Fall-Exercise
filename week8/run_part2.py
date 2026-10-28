"""Part 2 formal run: train the causal decoder, audit causality, score, sample.

Producing commands:
    uv run python run_part2.py --attempt smoke-01    # nonformal pipeline check
    uv run python run_part2.py --attempt formal-01   # the formal run

Gates (SPEC): parameter count 116,673; masked probe exactly 0; attention
upper triangle exactly 0 and rows sum 1; unmasked 300-step negative control
probe (scored WITHOUT mask) > 0; full-validation loss < 1.85 and < bigram
2.481889. Attempt directories are exclusive-new (preflight assertion); all
settings, per-100 logs and timestamps are written incrementally; checkpoint
and manifest are saved before any figure/sampling; generation reloads the
saved checkpoint and verifies tensor/hash identity first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baselines as bl
import transformer as tf

ROOT = Path(__file__).resolve().parent
EVID = ROOT / "evidence"

STEPS = 3000
BATCH = 16
WD = 0.1
SEED = 1337
NEG_STEPS = 300   # predeclared unmasked negative-control budget
EVAL_BLOCKS = 96  # validation-curve points use the first 96 blocks
PROMPT_AUDIT = "ROMEO:\nBut soft, what light"
PROMPT_GEN = "ROMEO:\n"
GEN_N = 500
GEN_TEMP = (0.5, 1.0)
GEN_SEED = 7


# --------------------------------------------------------------------------
# scoring: every validation character except the first, scored exactly once
# --------------------------------------------------------------------------

def block_windows(VAL: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(871,129) full-block windows [inp, tgt[-1]] and the (52,) tail window
    (None when the validation text ends exactly on a block boundary)."""
    fulls, tail = [], None
    for inp, tgt in bl.validation_blocks(VAL):
        w = np.concatenate([inp, tgt[-1:]])  # the last input HAS a target
        if len(tgt) == 128:
            fulls.append(w)
        else:
            tail = w
    return np.stack(fulls), tail


def batched_nll(params, windows, mask=True):
    """windows [B,T+1] -> per-target nll [B,T] (targets = shifted windows)."""
    fwd = lambda w: tf.forward(params, w[:-1], mask=mask)
    logits = jax.vmap(fwd)(windows)
    lp = jax.nn.log_softmax(logits)
    tgt = windows[:, 1:]
    return -jnp.take_along_axis(lp, tgt[:, :, None], axis=-1)[:, :, 0]


@partial(jax.jit, static_argnames="mask")
def eval_windows(params, windows, mask=True):
    return batched_nll(params, windows, mask=mask)


def score_validation(params, VAL: np.ndarray, n_blocks: int | None = None,
                     sabotage: str | None = None) -> dict:
    """Token-weighted loss over the block protocol; 111,539 targets in full.

    sabotage='swap_targets' is the negative control: inputs and targets swap
    roles, which a correct target handling must reject (different loss).
    """
    full, tail = block_windows(VAL)
    if sabotage == "swap_targets":
        full = np.concatenate([full[:, 1:], full[:, :1]], axis=1)
        tail = np.concatenate([tail[1:], tail[:1]])
    if n_blocks is not None:
        full = full[:n_blocks]
    parts = list(np.asarray(eval_windows(params, jnp.asarray(full))).ravel())
    if n_blocks is None and tail is not None:
        parts += list(np.asarray(eval_windows(params, jnp.asarray(tail)[None]))[0])
    return tf.combine_nll([jnp.asarray(parts)])


# --------------------------------------------------------------------------
# training: jitted loss+grad+AdamW, incremental logs, attempt directories
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnames="mask")
def train_step(params, state, windows, lr, step, mask):
    def loss_fn(p):
        return batched_nll(p, windows, mask=mask).mean()
    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, state = tf.adamw_step(params, grads, state, lr=lr,
                                  weight_decay=WD, step=step)
    return params, state, loss


def train(seed, steps, masked, log_every, tag, train_ids, VAL, attempt_dir,
          log_name="train-log.jsonl", smoke=False):
    master = jax.random.PRNGKey(seed)
    k_init, k_data = jax.random.split(master)
    params = tf.init_params(k_init)
    state = tf.adamw_init(params)
    t0 = time.time()
    log = []
    for step in range(1, steps + 1):
        k_data, sub = jax.random.split(k_data)
        windows = tf.sample_windows(sub, train_ids, BATCH)
        lr = tf.lr_schedule(step)  # same schedule; a shorter run stops early
        params, state, loss = train_step(params, state, windows, lr, step, masked)
        if step % log_every == 0 or step == steps:
            val = score_validation(params, VAL, n_blocks=EVAL_BLOCKS)
            entry = {"step": step, "lr": lr, "train_loss": float(loss),
                     "val_loss_96blk": val["loss"],
                     "unix_ts": time.time(), "elapsed_s": round(time.time() - t0, 1)}
            log.append(entry)
            with open(attempt_dir / log_name, "a") as f:  # incremental save
                f.write(json.dumps(entry) + "\n")
            print(f"[{tag}] step {step:5d}  lr {lr:.5f}  train {loss:.4f}  "
                  f"val(96blk) {val['loss']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return params, log, time.time() - t0


# --------------------------------------------------------------------------
# checkpoint save / load / verify
# --------------------------------------------------------------------------

def save_params(params: dict, path: Path):
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    np.savez(path, **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})


def _detree(node):
    """Convert {'0':x,'1':y,...} dicts (from list indices) back to lists."""
    if isinstance(node, dict):
        conv = {k: _detree(v) for k, v in node.items()}
        if conv and all(k.lstrip("/").isdigit() for k in conv):
            keys = sorted(conv, key=lambda k: int(k.lstrip("/")))
            return [conv[k] for k in keys]
        return conv
    return node


def load_params(path: Path) -> dict:
    data = np.load(path)
    tree: dict = {}
    for k in data.files:
        # keystr renders "['layers'][0]['bk']" -> normalize to plain parts.
        parts = [p.strip("'") for p in
                 k.replace("[", "/").replace("]", "").strip("/").split("/")]
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = jnp.asarray(data[k])
    return _detree(tree)


def verify_saved_weights(params: dict, path: Path) -> tuple[bool, str]:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_params(path)
    for a, b in zip(jax.tree_util.tree_leaves(params),
                    jax.tree_util.tree_leaves(loaded)):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            return False, sha
    return True, sha


# --------------------------------------------------------------------------
# generation: fixed-length left-aligned zero-padded buffer, one compile
# --------------------------------------------------------------------------

def generate_padded(params, prompt_ids, n_new: int, temperature: float,
                    key: jax.Array) -> jax.Array:
    """Same sampling as tf.generate but with a fixed 128-token buffer."""
    seq = [int(x) for x in prompt_ids]
    for _ in range(n_new):
        ctx = seq[-tf.CONTEXT:]
        L = len(ctx)
        buf = jnp.zeros(tf.CONTEXT, dtype=jnp.int32).at[:L].set(jnp.asarray(ctx))
        logits = tf.forward(params, buf)[L - 1] / temperature
        key, sub = jax.random.split(key)
        seq.append(int(jax.random.categorical(sub, logits)))
    return jnp.array(seq)


# --------------------------------------------------------------------------
# attempt directories: exclusive-new with preflight emptiness assertion
# --------------------------------------------------------------------------

def prepare_attempt(root: Path, name: str) -> Path:
    d = root / name
    assert not d.exists(), f"attempt directory already exists, refusing to overwrite: {d}"
    d.mkdir(parents=True)
    return d


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True,
                    help="exclusive-new attempt name, e.g. smoke-01 or formal-01")
    args = ap.parse_args(argv)
    smoke = args.attempt.startswith("smoke")
    steps = 3 if smoke else STEPS
    neg_steps = 2 if smoke else NEG_STEPS
    gen_n = 20 if smoke else GEN_N
    outdir = prepare_attempt(ROOT / "artifacts" / "part2", args.attempt)

    text = (ROOT / "data" / "input.txt").read_text(encoding="utf-8")
    vocab = bl.sorted_vocab(text)
    ids = bl.encode(text, vocab)
    TRAIN, VAL = bl.train_validation_split(ids)

    manifest = {
        "agent": "pi (native WSL)",
        "provider": "volcengine-plan",
        "model": "glm-5-3-flash",
        "command": f"uv run python run_part2.py --attempt {args.attempt}",
        "attempt": args.attempt,
        "formal": not smoke,
        "jax": jax.__version__,
        "started_unix_ts": time.time(),
        "settings": {"steps": steps, "batch": BATCH, "lr_peak": 3e-3,
                     "warmup": 100, "lr_final": 3e-4, "wd": WD, "seed": SEED,
                     "negative_control_steps": neg_steps, "gen_n": gen_n,
                     "gen_temperatures": list(GEN_TEMP), "gen_seed": GEN_SEED,
                     "context": tf.CONTEXT, "width": tf.WIDTH,
                     "heads": tf.HEADS, "layers": tf.LAYERS, "mlp": tf.MLP},
    }
    (outdir / "settings.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- main training (checkpoint saved before any figure/sampling) ---
    params, log, dur = train(SEED, steps, True, 100 if not smoke else 1,
                             "main", TRAIN, VAL, outdir)
    n_params = tf.count_params(params)
    print(f"parameters: {n_params} (reference 116673)")
    save_params(params, outdir / "final-weights.npz")
    reload_ok, sha = verify_saved_weights(params, outdir / "final-weights.npz")
    assert reload_ok, "checkpoint reload verification failed"
    manifest.update({"train_seconds": round(dur, 1), "n_params": n_params,
                     "checkpoint_sha256": sha, "checkpoint_reload_verified": reload_ok})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"checkpoint saved and reload-verified (sha256 {sha[:16]}...)")

    # generation uses the RELOADED checkpoint, not the in-memory params
    reloaded = load_params(outdir / "final-weights.npz")

    # --- causality probe (masked) ---
    pairs = tf.find_prefix_pairs(np.asarray(ids), n_pairs=8)
    probe = tf.prefix_probe(reloaded, np.asarray(ids), pairs, mask=True)
    print(f"causality probe (largest logit change, first 64 positions): {probe!r}")
    manifest["probe_pairs"] = [[int(a), int(b)] for a, b in pairs]
    manifest["probe_masked"] = probe

    # --- attention audit (np.triu, not ndarray.triu) ---
    audit_prompt = jnp.asarray(bl.encode(PROMPT_AUDIT, vocab))
    attn = np.asarray(tf.attention_weights(reloaded, audit_prompt))
    largest_above = max(float(np.triu(attn[l, h], k=1).max())
                        for l in range(attn.shape[0]) for h in range(attn.shape[1]))
    row_dev = max(float(np.abs(attn[l, h].sum(-1) - 1).max())
                  for l in range(attn.shape[0]) for h in range(attn.shape[1]))
    print(f"largest above-diagonal attention weight: {largest_above!r}")
    print(f"largest row-sum deviation from 1:        {row_dev:.2e}")
    manifest["attention_largest_above_diagonal"] = largest_above
    manifest["attention_row_sum_max_dev"] = row_dev

    # --- unmasked negative control: probe MUST be scored with mask=False ---
    neg_params, neg_log, neg_dur = train(SEED, neg_steps, False,
                                         1 if smoke else 100, "unmasked-ctl",
                                         TRAIN, VAL, outdir,
                                         log_name="unmasked-control-log.jsonl")
    save_params(neg_params, outdir / "unmasked-control-weights.npz")
    neg_probe_unmasked = tf.prefix_probe(neg_params, np.asarray(ids), pairs, mask=False)
    neg_probe_masked = tf.prefix_probe(neg_params, np.asarray(ids), pairs, mask=True)
    print(f"negative-control probe, scored WITHOUT mask: {neg_probe_unmasked!r}")
    print(f"  (same model scored WITH mask: {neg_probe_masked!r} — zero by "
          f"construction, which is why mask=False is required)")
    manifest["negative_control"] = {
        "steps": neg_steps, "probe_scored_mask_false": neg_probe_unmasked,
        "probe_scored_mask_true_for_contrast": neg_probe_masked,
        "detects_leakage": bool(neg_probe_unmasked > 0),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- full validation score (all 111,539 targets) ---
    full = score_validation(reloaded, VAL)
    print(f"full held-out loss: {full['loss']:.6f}  PPL {full['ppl']:.3f}  "
          f"({full['n_scored']} targets)")
    manifest["full_validation"] = full
    gates = {
        "param_count_116673": n_params == 116673,
        "probe_exactly_zero": probe == 0.0,
        "attention_upper_exactly_zero": largest_above == 0.0,
        "attention_rows_sum_one": row_dev < 1e-5,
        "negative_control_detects_leakage": neg_probe_unmasked > 0,
        "full_loss_below_185": full["loss"] < 1.85,
        "beats_bigram_2481889": full["loss"] < 2.481889,
    }
    for k, ok in gates.items():
        print(f"gate {k:34s}: {'PASS' if ok else 'FAIL'}")
    manifest["gates"] = {k: bool(v) for k, v in gates.items()}
    manifest["finished_unix_ts"] = time.time()
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- generation from the verified saved weights ---
    prompt_ids = jnp.asarray(bl.encode(PROMPT_GEN, vocab))
    for temp in GEN_TEMP:
        out = generate_padded(reloaded, prompt_ids, gen_n, temp,
                              jax.random.PRNGKey(GEN_SEED))
        s = "".join(vocab[i] for i in np.asarray(out))
        name = f"sample-temp{temp}.txt"
        (outdir / name).write_text(s)
        if not smoke:  # evidence copies for the formal run only
            (EVID / f"part2-{name}").write_text(s)
        print(f"--- generation temp={temp} ---\n{s}\n")
    manifest["generation"] = {"prompt": PROMPT_GEN, "n": gen_n,
                              "temperatures": list(GEN_TEMP), "seed": GEN_SEED,
                              "weights": "reloaded checkpoint"}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- figures (formal: also copied to evidence/) ---
    tagp = "smoke" if smoke else "part2"
    figdirs = [outdir] + ([EVID] if not smoke else [])
    steps_log = [d["step"] for d in log]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_log, [d["train_loss"] for d in log], "-o", ms=3,
            label="training loss (batch)")
    ax.plot(steps_log, [d["val_loss_96blk"] for d in log], "-o", ms=3,
            label="validation loss (first 96 blocks)")
    ax.axhline(3.347305, color="gray", ls=":", label="unigram 3.347")
    ax.axhline(2.481889, color="crimson", ls=":", label="bigram 2.482")
    ax.axhline(full["loss"], color="green", ls="--", lw=1,
               label=f"full validation {full['loss']:.4f}")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats/char)")
    suffix = " (SMOKE, nonformal)" if smoke else " (seed 1337)"
    ax.set_title(f"Part 2: causal transformer training on CPU{suffix}")
    ax.legend(fontsize=8)
    for d in figdirs:
        fig.savefig(d / f"{tagp}-loss-curves.png", dpi=150)
    plt.close(fig)

    chars = PROMPT_AUDIT.replace("\n", "\\n")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for h in range(4):
        im = axes[h].imshow(attn[0, h], cmap="viridis", vmin=0, vmax=1)
        axes[h].set_title(f"layer 1, head {h+1}")
        axes[h].set_xlabel("key position")
    axes[0].set_ylabel("query position")
    fig.suptitle(f"{'[SMOKE] ' if smoke else ''}Part 2: first-layer attention "
                 f"for {chars!r} (largest above-diagonal weight {largest_above:g})")
    for d in figdirs:
        fig.savefig(d / f"{tagp}-attention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"attempt {args.attempt} complete: {outdir}")


if __name__ == "__main__":
    main()
