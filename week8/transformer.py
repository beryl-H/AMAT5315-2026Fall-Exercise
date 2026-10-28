"""Part 2: pre-LN causal transformer decoder, AdamW, schedule, probes.

Architecture fixed by the sheet and parameter count 116,673:
  token emb 65x64 (4160) + position emb 128x64 (8192)
  per layer: LN(128) + qkv 3*(64*64+64) (12480) + out (64*64+64) (4160)
             + LN(128) + MLP (64*256+256)+(256*64+64) (33088) = 49,984
  final LN (128) + untied head 64*65+65 (4225)
  total = 4160 + 8192 + 2*49984 + 128 + 4225 = 116,673
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

VOCAB = 65
CONTEXT = 128
WIDTH = 64
HEADS = 4
LAYERS = 2
MLP = 256


def init_params(key: jax.Array) -> dict:
    """Gaussian init, std scaled like GPT (embeddings 0.02+ head 0.02)."""

    def normal(k, shape, std=0.02):
        return std * jax.random.normal(k, shape)

    ktok, kpos, *klayers = jax.random.split(key, LAYERS + 3)
    params = {
        "tok_emb": normal(ktok, (VOCAB, WIDTH)),
        "pos_emb": normal(kpos, (CONTEXT, WIDTH)),
        "layers": [],
        "ln_f": {"g": jnp.ones(WIDTH), "b": jnp.zeros(WIDTH)},
        "head": {"w": normal(klayers[-1], (WIDTH, VOCAB)), "b": jnp.zeros(VOCAB)},
    }
    for kl in klayers[:LAYERS]:
        kq, kk, kv, ko, km1, km2 = jax.random.split(kl, 6)
        params["layers"].append({
            "ln1": {"g": jnp.ones(WIDTH), "b": jnp.zeros(WIDTH)},
            "wq": normal(kq, (WIDTH, WIDTH), 1 / math.sqrt(WIDTH)), "bq": jnp.zeros(WIDTH),
            "wk": normal(kk, (WIDTH, WIDTH), 1 / math.sqrt(WIDTH)), "bk": jnp.zeros(WIDTH),
            "wv": normal(kv, (WIDTH, WIDTH), 1 / math.sqrt(WIDTH)), "bv": jnp.zeros(WIDTH),
            "wo": normal(ko, (WIDTH, WIDTH), 1 / math.sqrt(WIDTH)), "bo": jnp.zeros(WIDTH),
            "ln2": {"g": jnp.ones(WIDTH), "b": jnp.zeros(WIDTH)},
            "mlp1": {"w": normal(km1, (WIDTH, MLP), 1 / math.sqrt(WIDTH)), "b": jnp.zeros(MLP)},
            "mlp2": {"w": normal(km2, (MLP, WIDTH), 1 / math.sqrt(MLP)), "b": jnp.zeros(WIDTH)},
        })
    return params


def count_params(params: dict) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves))


def layer_norm(x, ln):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * ln["g"] + ln["b"]


def attention_block(x, p, mask: bool):
    """x: [T, W] -> (out [T, W], weights [H, T, T])."""
    T = x.shape[0]
    q = (x @ p["wq"] + p["bq"]).reshape(T, HEADS, WIDTH // HEADS)
    k = (x @ p["wk"] + p["bk"]).reshape(T, HEADS, WIDTH // HEADS)
    v = (x @ p["wv"] + p["bv"]).reshape(T, HEADS, WIDTH // HEADS)
    d = WIDTH // HEADS
    scores = jnp.einsum("thd,shd->hts", q, k) / math.sqrt(d)
    if mask:
        tri = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores = jnp.where(tri, scores, -jnp.inf)
    w = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hts,shd->thd", w, v).reshape(T, WIDTH)
    return out @ p["wo"] + p["bo"], w


def forward(params: dict, tokens: jax.Array, mask: bool = True) -> jax.Array:
    """tokens: [T] -> logits [T, VOCAB]."""
    T = tokens.shape[0]
    x = params["tok_emb"][tokens] + params["pos_emb"][:T]
    for p in params["layers"]:
        h, _ = attention_block(layer_norm(x, p["ln1"]), p, mask)
        x = x + h
        m = jax.nn.gelu(layer_norm(x, p["ln2"]) @ p["mlp1"]["w"] + p["mlp1"]["b"])
        x = x + m @ p["mlp2"]["w"] + p["mlp2"]["b"]
    x = layer_norm(x, params["ln_f"])
    return x @ params["head"]["w"] + params["head"]["b"]


def attention_weights(params: dict, tokens: jax.Array) -> jax.Array:
    """[layers, heads, T, T] attention matrices for the audit figures."""
    T = tokens.shape[0]
    x = params["tok_emb"][tokens] + params["pos_emb"][:T]
    ws = []
    for p in params["layers"]:
        h, w = attention_block(layer_norm(x, p["ln1"]), p, mask=True)
        x = x + h
        m = jax.nn.gelu(layer_norm(x, p["ln2"]) @ p["mlp1"]["w"] + p["mlp1"]["b"])
        x = x + m @ p["mlp2"]["w"] + p["mlp2"]["b"]
        ws.append(w)
    return jnp.stack(ws)


def nll(params: dict, tokens: jax.Array) -> jax.Array:
    """Token nll of targets tokens[1:] given inputs tokens[:-1]: [T-1]."""
    logits = forward(params, tokens[:-1])
    return -jax.nn.log_softmax(logits)[jnp.arange(tokens.shape[0] - 1), tokens[1:]]


def combine_nll(parts: list) -> dict:
    """Token-weighted combination of per-target nll arrays."""
    nlls = jnp.concatenate(parts)
    loss = float(nlls.sum() / nlls.size)
    return {"loss": loss, "ppl": math.exp(loss), "n_scored": int(nlls.size)}


def lr_schedule(step: int) -> float:
    """3e-3 linear warmup over 100 steps, cosine decay to 3e-4 at 3000."""
    peak, final, warmup, total = 3e-3, 3e-4, 100, 3000
    if step < warmup:
        return peak * step / warmup
    t = min(step - warmup, total - warmup) / (total - warmup)
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * t))


def adamw_init(params: dict) -> dict:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros}


def adamw_step(params: dict, grads: dict, state: dict, lr: float,
               weight_decay: float, step: int, beta1=0.9, beta2=0.999,
               eps=1e-8):
    """Decoupled AdamW: p <- p - lr*wd*p - lr*mhat/(sqrt(vhat)+eps)."""
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g,
                               state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * g * g,
                               state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2 ** step), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * weight_decay * p - lr * mh / (jnp.sqrt(vh) + eps),
        params, m_hat, v_hat)
    return new_params, {"m": m, "v": v}


def find_prefix_pairs(ids: np.ndarray, n_pairs: int = 8, prefix: int = 64) -> list:
    """Deterministic corpus pairs sharing a 64-char prefix and differing later.

    The corpus contains only ~5 distinct duplication sites, so pairs are
    greedily kept (sorted by position) if both members are at least 8
    characters away from every kept member; this spans every site.
    """
    import collections

    view = np.lib.stride_tricks.sliding_window_view(ids, prefix)
    h = np.zeros(len(view), dtype=np.int64)
    for k in range(prefix):
        h = h * 131 + view[:, k]
    order = np.argsort(h, kind="stable")
    dup = h[order][1:] == h[order][:-1]
    dpos = np.unique(np.concatenate([order[:-1][dup], order[1:][dup]]))
    groups = collections.defaultdict(list)
    for p in dpos:
        groups[int(h[p])].append(int(p))
    candidates = []
    for hv, g in groups.items():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                if np.array_equal(view[a], view[b]) and not np.array_equal(
                        ids[a:a + 128], ids[b:b + 128]):
                    candidates.append((a, b))
    candidates.sort()
    kept, anchors = [], []
    for a, b in candidates:
        if all(abs(a - c) >= 8 and abs(b - c) >= 8 for c in anchors):
            kept.append((a, b))
            anchors.extend([a, b])
        if len(kept) == n_pairs:
            break
    return kept


def prefix_probe(params: dict, ids: np.ndarray, pairs: list, mask: bool = True) -> float:
    """Largest change in the logits of the first 64 positions over the pairs."""
    worst = 0.0
    for a, b in pairs:
        la = forward(params, jnp.asarray(ids[a:a + 128]), mask=mask)[:64]
        lb = forward(params, jnp.asarray(ids[b:b + 128]), mask=mask)[:64]
        worst = max(worst, float(jnp.max(jnp.abs(la - lb))))
    return worst


def sample_windows(key: jax.Array, train_ids: np.ndarray, batch: int) -> jax.Array:
    """(batch, 129) windows: 128 inputs + the next character as target."""
    offsets = jax.random.randint(key, (batch,), 0, len(train_ids) - 129)
    idx = offsets[:, None] + jnp.arange(129)
    return jnp.asarray(train_ids)[idx]


def generate(params: dict, prompt_ids: jax.Array, n_new: int, temperature: float,
             key: jax.Array) -> jax.Array:
    """Autoregressive sampling with the causal model; returns prompt + n_new."""
    seq = [int(x) for x in prompt_ids]
    for _ in range(n_new):
        ctx = jnp.array(seq[-CONTEXT:])
        logits = forward(params, ctx)[-1] / temperature
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, logits)
        seq.append(int(nxt))
    return jnp.array(seq)
