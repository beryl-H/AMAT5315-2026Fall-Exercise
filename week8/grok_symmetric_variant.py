"""Part 4: grokking modular addition c = (a+b) mod 97.

Self-contained module (does not modify the audited Part 2/3 files).
Architecture frozen in SPEC.md before any formal run:
  one layer, NO LayerNorm, width 64, 4 heads (head width 16), MLP 256,
  vocab 98 (residues 0..96 + '=' id 97), input [a, b, '='], answer read at
  position 2, no positional embeddings, no causal mask, output softmax over
  the 97 residue classes (chance 1/97, uniform loss ln 97).
  Parameters: emb 98*64 + attn (3*(64*64+64) + 64*64+64) + mlp (64*256+256
  + 256*64+64) + untied head 64*97+97 = 62,305.
  Init: weights N(0, sqrt(2/(64+97))), biases 0 (initial param norm ~28).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

P = 97
EQ = 97            # '=' token id; vocab size 98
VOCAB = 98
WIDTH = 64
HEADS = 4
MLP = 256
LR = 3e-3
BETA1, BETA2 = 0.9, 0.98
WD = 1.0
MAX_STEPS = 10000
LOG_EVERY = 100
ACC_THRESHOLD = 0.99
STOP_DELAY = 1000
SPLIT_SEED = 897
LABEL_SEED = 8970
ABLATION_SEED = 4497


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def init_params(key: jax.Array) -> dict:
    std = math.sqrt(2 / (WIDTH + P))
    kemb, kq, kk, kv, ko, km1, km2, kh = jax.random.split(key, 8)

    def normal(k, shape):
        return std * jax.random.normal(k, shape)

    return {
        "emb": normal(kemb, (VOCAB, WIDTH)),
        "wq": normal(kq, (WIDTH, WIDTH)), "bq": jnp.zeros(WIDTH),
        "wk": normal(kk, (WIDTH, WIDTH)), "bk": jnp.zeros(WIDTH),
        "wv": normal(kv, (WIDTH, WIDTH)), "bv": jnp.zeros(WIDTH),
        "wo": normal(ko, (WIDTH, WIDTH)), "bo": jnp.zeros(WIDTH),
        "mlp1": {"w": normal(km1, (WIDTH, MLP)), "b": jnp.zeros(MLP)},
        "mlp2": {"w": normal(km2, (MLP, WIDTH)), "b": jnp.zeros(WIDTH)},
        "head": {"w": normal(kh, (WIDTH, P)), "b": jnp.zeros(P)},
    }


def count_params(params: dict) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def param_norm(params: dict) -> float:
    return float(jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(params))))


def forward(params: dict, tokens: jax.Array) -> jax.Array:
    """tokens [B,3] -> logits [B,3,VOCAB]. No mask, no positional embedding."""
    B, T = tokens.shape
    x = params["emb"][tokens]                       # [B,T,W]
    q = (x @ params["wq"] + params["bq"]).reshape(B, T, HEADS, WIDTH // HEADS)
    k = (x @ params["wk"] + params["bk"]).reshape(B, T, HEADS, WIDTH // HEADS)
    v = (x @ params["wv"] + params["bv"]).reshape(B, T, HEADS, WIDTH // HEADS)
    d = WIDTH // HEADS
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d)
    w = jax.nn.softmax(scores, axis=-1)             # no causal mask (frozen)
    att = jnp.einsum("bhts,bshd->bthd", w, v).reshape(B, T, WIDTH)
    x = x + att @ params["wo"] + params["bo"]
    h = jax.nn.gelu(x @ params["mlp1"]["w"] + params["mlp1"]["b"])
    x = x + h @ params["mlp2"]["w"] + params["mlp2"]["b"]
    return x @ params["head"]["w"] + params["head"]["b"]


@jax.jit
def loss_acc(params, tokens, targets):
    logits = forward(params, tokens)[:, 2, :]       # answer read at '='
    lp = jax.nn.log_softmax(logits)
    nll = -lp[jnp.arange(targets.shape[0]), targets]
    acc = (logits.argmax(-1) == targets).mean()
    return nll.mean(), acc


@jax.jit
def train_step(params, state, tokens, targets, lr, step, weight_decay=WD):
    """One full-batch update with the frozen betas/wd; state is threaded."""
    def loss_fn(p):
        return loss_acc(p, tokens, targets)[0]
    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, state = grok_adamw_step(params, grads, state, lr=lr, step=step,
                                     weight_decay=weight_decay)
    return params, state, loss


def grok_adamw_init(params):
    return {"m": jax.tree_util.tree_map(jnp.zeros_like, params),
            "v": jax.tree_util.tree_map(jnp.zeros_like, params)}


def grok_adamw_step(params, grads, state=None, lr=LR, step=1, weight_decay=WD):
    """Decoupled AdamW with betas (0.9, 0.98)."""
    if state is None:
        state = grok_adamw_init(params)
    m = jax.tree_util.tree_map(lambda m, g: BETA1 * m + (1 - BETA1) * g,
                               state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: BETA2 * v + (1 - BETA2) * g * g,
                               state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - BETA1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - BETA2 ** step), v)
    new = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * weight_decay * p - lr * mh / (jnp.sqrt(vh) + 1e-8),
        params, m_hat, v_hat)
    return new, {"m": m, "v": v}


# AdamW alias used by tests (stateful form identical to the frozen formula)
def adamw_init(params):
    return grok_adamw_init(params)


def adamw_step(params, grads, state, lr, step, weight_decay=WD):
    return grok_adamw_step(params, grads, state, lr=lr, step=step,
                           weight_decay=weight_decay)


# --------------------------------------------------------------------------
# dataset, split, labels
# --------------------------------------------------------------------------

def build_dataset() -> dict:
    a = np.repeat(np.arange(P), P)
    b = np.tile(np.arange(P), P)
    return {"inputs": np.stack([a, b, np.full(a.size, EQ)], axis=1).astype(np.int32),
            "targets": ((a + b) % P).astype(np.int32)}


def pair_sums() -> np.ndarray:
    a = np.repeat(np.arange(P), P)
    b = np.tile(np.arange(P), P)
    return (a + b) % P


def fixed_split(fraction: float) -> dict:
    """Label-independent split from the frozen RNG 897; 30% -> 2823,
    50% -> 4704 (round convention per coordinator instruction); nested."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(P * P)
    n = int(round(fraction * P * P))
    return {"train": np.sort(perm[:n]), "test": np.sort(perm[n:])}


def random_labels(seed: int = 0) -> np.ndarray:
    """Vector shuffle of ALL 9409 true labels across pair indices (frozen RNG
    8970): class frequencies preserved, pair-to-label structure broken.
    NOT a global class permutation pi((a+b)%97)."""
    true = build_dataset()["targets"]
    return true[np.random.default_rng(LABEL_SEED).permutation(true.size)]


# --------------------------------------------------------------------------
# Fourier analysis and ablations
# --------------------------------------------------------------------------

def fourier_power(E: np.ndarray) -> np.ndarray:
    """|rfft| power per non-DC frequency k=1..48, summed over dimensions.
    E: (98,64) full embedding; rows 0..96 are the residues."""
    X = np.fft.rfft(np.asarray(E)[:P], axis=0)      # bins 0..48
    return (np.abs(X[1:]) ** 2).sum(axis=1)         # (48,) for k=1..48


def top5(E: np.ndarray) -> tuple[list[int], float]:
    """Top-5 frequencies of this snapshot and their share of k=1..48 power."""
    power = fourier_power(E)
    order = np.argsort(power)[::-1][:5]
    share = float(power[order].sum() / power.sum())
    return [int(k) for k in (order + 1)], share


def project_embedding(E: np.ndarray, keep: list[int], mode: str) -> np.ndarray:
    """Reconstruct residue rows through a chosen frequency set.

    keep-mode:  E' = DC + components of `keep`      (others dropped)
    remove-mode: E' = DC + components of all k except `keep`
    DC (k=0) is preserved in every mode; the '=' row (97) is untouched.
    Exact trig reconstruction: x[a] = DC + (2/97) sum_k (C_k cos + S_k sin).
    """
    E = np.asarray(E, dtype=np.float64)
    out = E.copy()
    a = np.arange(P)
    DC = E[:P].mean(axis=0)
    S = set(keep) if mode == "keep" else set(range(1, 49)) - set(keep)
    recon = np.broadcast_to(DC, (P, E.shape[1])).copy()
    for k in sorted(S):
        C = (E[:P] * np.cos(2 * np.pi * k * a / P)[:, None]).sum(0)
        Ssin = (E[:P] * np.sin(2 * np.pi * k * a / P)[:, None]).sum(0)
        recon += (2 / P) * (np.outer(np.cos(2 * np.pi * k * a / P), C)
                            + np.outer(np.sin(2 * np.pi * k * a / P), Ssin))
    out[:P] = recon
    return out


def random_other_draws(top5: list[int], n_draws: int = 10,
                       seed: int = ABLATION_SEED) -> list[list[int]]:
    """10 fixed draws of 5 frequencies from the complement of the top-5."""
    complement = np.array(sorted(set(range(1, 49)) - set(top5)))
    rng = np.random.default_rng(seed)
    return [sorted(int(k) for k in rng.choice(complement, 5, replace=False))
            for _ in range(n_draws)]


# --------------------------------------------------------------------------
# log analysis and attempt protocol
# --------------------------------------------------------------------------

def analyze_log(log: list[dict], max_steps: int = MAX_STEPS) -> dict:
    """Crossings from the ACTUAL logged grid only."""
    mem = next((d["step"] for d in log if d["train_acc"] >= ACC_THRESHOLD), None)
    grok = next((d["step"] for d in log if d["test_acc"] >= ACC_THRESHOLD), None)
    if grok is not None:
        stop = min(max_steps, grok + STOP_DELAY)
    else:
        stop = max_steps
    delay = (grok / mem) if (grok is not None and mem) else None
    return {"memorization_step": mem, "grok_step": grok, "stop_step": stop,
            "delay_ratio": delay}


def prepare_attempt(root: Path, name: str) -> Path:
    d = Path(root) / name
    assert not d.exists(), f"attempt directory already exists: {d}"
    d.mkdir(parents=True)
    return d


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {n: _sha(root / n) for n in ("grok.py", "SPEC.md")}


def save_params(params: dict, path: Path) -> None:
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    np.savez(path, **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})


def load_params(path: Path) -> dict:
    data = np.load(path)
    tree: dict = {}
    for k in data.files:
        parts = [p.strip("'") for p in
                 k.replace("[", "/").replace("]", "").strip("/").split("/")]
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = jnp.asarray(data[k])
    return tree


def _detree(node):
    if isinstance(node, dict):
        conv = {k: _detree(v) for k, v in node.items()}
        if conv and all(k.lstrip("/").isdigit() for k in conv):
            return [conv[k] for k in sorted(conv, key=lambda k: int(k.lstrip("/")))]
        return conv
    return node


# --------------------------------------------------------------------------
# training run (formal and smoke share this exact pipeline)
# --------------------------------------------------------------------------

def _verify_params(params: dict, path: Path) -> tuple[bool, str]:
    sha = _sha(path)
    loaded = _detree(load_params(path))
    for a, b in zip(jax.tree_util.tree_leaves(params),
                    jax.tree_util.tree_leaves(loaded)):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            return False, sha
    return True, sha


def run_one(seed: int, fraction: float = 0.30, wd: float = WD,
            max_steps: int = MAX_STEPS, log_every: int = LOG_EVERY,
            outdir: Path = None, labels: np.ndarray | None = None,
            permuted_labels: bool = False) -> dict:
    """One full-batch training run with incremental saves.

    Order guaranteed: settings -> step-0 log + init checkpoint -> training
    (log every `log_every`, memorization checkpoint at first logged
    crossing) -> final checkpoint -> completion receipt -> FFT/ablation
    postprocessing. If no memorization/grokking occurs, explicit nulls are
    recorded (never faked).
    """
    outdir = Path(outdir)
    assert not outdir.exists(), f"attempt directory already exists: {outdir}"
    outdir.mkdir(parents=True)
    is_smoke = any("smoke" in p for p in outdir.parts)
    started = time.time()
    settings = {"seed": seed, "fraction": fraction, "wd": wd, "lr": LR,
                "beta1": BETA1, "beta2": BETA2, "max_steps": max_steps,
                "log_every": log_every, "acc_threshold": ACC_THRESHOLD,
                "permuted_labels": permuted_labels,
                "started_unix_ts": started,
                "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_hashes": source_hashes(), "formal": not is_smoke}
    (outdir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    data = build_dataset()
    targets_all = random_labels() if permuted_labels else data["targets"]
    split = fixed_split(fraction)
    tr_tok = jnp.asarray(data["inputs"][split["train"]])
    tr_tgt = jnp.asarray(targets_all[split["train"]])
    te_tok = jnp.asarray(data["inputs"][split["test"]])
    te_tgt = jnp.asarray(targets_all[split["test"]])

    master = jax.random.PRNGKey(seed)
    params = init_params(master)
    state = grok_adamw_init(params)

    def snapshot(step):
        tr_l, tr_a = loss_acc(params, tr_tok, tr_tgt)
        te_l, te_a = loss_acc(params, te_tok, te_tgt)
        return {"step": step, "train_loss": float(tr_l), "train_acc": float(tr_a),
                "test_loss": float(te_l), "test_acc": float(te_a),
                "param_norm": param_norm(params), "unix_ts": time.time()}

    log = []
    entry0 = snapshot(0)  # initialization, BEFORE any optimizer update
    log.append(entry0)
    with open(outdir / "train-log.jsonl", "a") as f:
        f.write(json.dumps(entry0) + "\n")
    save_params(params, outdir / "checkpoint-init.npz")
    ok, sha_init = _verify_params(params, outdir / "checkpoint-init.npz")
    assert ok, "init checkpoint verification failed"

    mem_step, grok_step, saved_mem = None, None, False
    stop_step = max_steps
    for step in range(1, max_steps + 1):
        params, state, _ = train_step(params, state, tr_tok, tr_tgt, LR, step,
                                          weight_decay=wd)
        if step % log_every == 0 or step == max_steps:
            entry = snapshot(step)
            log.append(entry)
            with open(outdir / "train-log.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            if mem_step is None and entry["train_acc"] >= ACC_THRESHOLD:
                mem_step = entry["step"]
            if grok_step is None and entry["test_acc"] >= ACC_THRESHOLD:
                grok_step = entry["step"]
                stop_step = min(max_steps, grok_step + STOP_DELAY)
            if mem_step is not None and not saved_mem and mem_step == entry["step"]:
                save_params(params, outdir / "checkpoint-memorization.npz")
                ok, sha_mem = _verify_params(params, outdir / "checkpoint-memorization.npz")
                assert ok, "memorization checkpoint verification failed"
                saved_mem = True
        if step >= stop_step:
            break

    save_params(params, outdir / "checkpoint-final.npz")
    ok, sha_final = _verify_params(params, outdir / "checkpoint-final.npz")
    assert ok, "final checkpoint verification failed"
    receipt = {"seed": seed, "fraction": fraction, "wd": wd,
               "steps_run": log[-1]["step"],
               "memorization_step": mem_step, "grok_step": grok_step,
               "stop_step": stop_step if grok_step is not None else max_steps,
               "delay_ratio": (grok_step / mem_step) if (grok_step and mem_step) else None,
               "checkpoint_sha256": {"init": sha_init,
                                     "memorization": _sha(outdir / "checkpoint-memorization.npz") if saved_mem else None,
                                     "final": sha_final},
               "finished_unix_ts": time.time(),
               "utc_finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (outdir / "completion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (outdir / "train-log.json").write_text(json.dumps(log, indent=2) + "\n")

    # postprocessing AFTER the receipt (failures here never force retraining)
    fourier = {}
    for name in ("init", "final"):
        E = np.asarray(load_params(outdir / f"checkpoint-{name}.npz")["emb"])
        ks, share = top5(E)
        fourier[name] = {"top5": ks, "share": share}
    if saved_mem:
        E = np.asarray(load_params(outdir / "checkpoint-memorization.npz")["emb"])
        ks, share = top5(E)
        fourier["memorization"] = {"top5": ks, "share": share}
    else:
        fourier["memorization"] = None
    (outdir / "fourier.json").write_text(json.dumps(fourier, indent=2) + "\n")

    final_E = np.asarray(load_params(outdir / "checkpoint-final.npz")["emb"])
    final_params = _detree(load_params(outdir / "checkpoint-final.npz"))
    key5 = fourier["final"]["top5"]

    def test_accuracy_with(E):
        p = dict(final_params)
        p["emb"] = jnp.asarray(E, dtype=jnp.float32)
        _, acc = loss_acc(p, te_tok, te_tgt)
        return float(acc)

    ablations = {"keep_only_key5": test_accuracy_with(
                     project_embedding(final_E, keep=key5, mode="keep")),
                 "remove_key5": test_accuracy_with(
                     project_embedding(final_E, keep=key5, mode="remove")),
                 "remove_random_other5": [],
                 "keep_only_random_other5": []}
    for draw in random_other_draws(key5, n_draws=10, seed=ABLATION_SEED):
        ablations["remove_random_other5"].append(test_accuracy_with(
            project_embedding(final_E, keep=draw, mode="remove")))
        ablations["keep_only_random_other5"].append(test_accuracy_with(
            project_embedding(final_E, keep=draw, mode="keep")))
    (outdir / "ablations.json").write_text(json.dumps(ablations, indent=2) + "\n")

    metrics = dict(receipt, log=log, fourier=fourier, ablations=ablations,
                   final_log_entry=log[-1])
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


# PRESERVED VERBATIM from grok.py at commit 7b9719b (2026-10-28).
# This is the INVALIDATED symmetric variant (no positional embeddings, no
# causal mask): answer logits are invariant to swapping a and b, so it cannot
# memorize the worksheet random-label control (393 conflicting unordered
# training pairs; max train accuracy 0.860786 < 0.99). Kept for provenance;
# the corrected decoder lives in grok.py. See PART4_DECODER_PROTOCOL.md.
