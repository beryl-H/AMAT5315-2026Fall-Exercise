"""Part 5: pre-norm transformer decoder.

2 layers, width 96, 4 heads, context 64, GELU MLP 384, learned token (512x96)
and position (64x96) embeddings, every linear map biased (q/k/v/out, MLP1/2,
readout), 2 LayerNorms per block + final norm. 328832 parameters.
Frozen hyperparameters in TR_HPARAMS. GELU = exact erf form (jax.nn.gelu with
approximate=False), LayerNorm eps 1e-5. Init: normal(0.02) weights, zeros
biases, LN g=1/b=0 — conventional choices frozen before any run."""
import jax
import jax.numpy as jnp

from src.nn_common import stable_ce

TR_HPARAMS = {"steps": 6000, "batch": 32, "ctx": 64, "peak_lr": 0.003,
              "warmup": 300, "end_frac": 0.1, "betas": (0.9, 0.999), "wd": 0.01,
              "seed": 0, "width": 96, "heads": 4, "mlp_hidden": 384,
              "clip": 1.0, "layers": 2}


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


def _linear(x, W, b):
    return x @ W + b


def _block(params, x, causal, return_attn):
    B, T, d = x.shape
    h = _ln(x, params["ln1_g"], params["ln1_b"])
    q = _linear(h, params["Wq"], params["bq"])
    k = _linear(h, params["Wk"], params["bk"])
    v = _linear(h, params["Wv"], params["bv"])
    hd = d // 4  # heads = 4
    qh = q.reshape(B, T, 4, hd).transpose(0, 2, 1, 3)  # B x 4 x T x hd
    kh = k.reshape(B, T, 4, hd).transpose(0, 2, 1, 3)
    vh = v.reshape(B, T, 4, hd).transpose(0, 2, 1, 3)
    scores = qh @ kh.transpose(0, 1, 3, 2) / jnp.sqrt(hd)  # B x 4 x T x T
    if causal:
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(mask[None, None], scores, -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1)
    mixed = (attn @ vh).transpose(0, 2, 1, 3).reshape(B, T, d)
    attn_out = _linear(mixed, params["Wo"], params["bo"])
    x = x + attn_out
    h2 = _ln(x, params["ln2_g"], params["ln2_b"])
    mlp_out = _linear(jax.nn.gelu(_linear(h2, params["W1"], params["b1"]),
                                  approximate=False),
                      params["W2"], params["b2"])
    x = x + mlp_out
    return (x, attn) if return_attn else (x, None)


def init_transformer(key):
    kemb, kpos, *kblocks, kread = jax.random.split(key, 2 + 2 * 8 + 1)
    std = 0.02
    params = {
        "tok_emb": jax.random.normal(kemb, (512, 96), dtype=jnp.float32) * std,
        "pos_emb": jax.random.normal(kpos, (64, 96), dtype=jnp.float32) * std,
        "blocks": [],
    }
    for i in range(2):
        kb = kblocks[8 * i: 8 * i + 8]
        params["blocks"].append({
            "ln1_g": jnp.ones(96, dtype=jnp.float32),
            "ln1_b": jnp.zeros(96, dtype=jnp.float32),
            "Wq": jax.random.normal(kb[0], (96, 96), dtype=jnp.float32) * std,
            "bq": jnp.zeros(96, dtype=jnp.float32),
            "Wk": jax.random.normal(kb[1], (96, 96), dtype=jnp.float32) * std,
            "bk": jnp.zeros(96, dtype=jnp.float32),
            "Wv": jax.random.normal(kb[2], (96, 96), dtype=jnp.float32) * std,
            "bv": jnp.zeros(96, dtype=jnp.float32),
            "Wo": jax.random.normal(kb[3], (96, 96), dtype=jnp.float32) * std,
            "bo": jnp.zeros(96, dtype=jnp.float32),
            "ln2_g": jnp.ones(96, dtype=jnp.float32),
            "ln2_b": jnp.zeros(96, dtype=jnp.float32),
            "W1": jax.random.normal(kb[4], (96, 384), dtype=jnp.float32) * std,
            "b1": jnp.zeros(384, dtype=jnp.float32),
            "W2": jax.random.normal(kb[5], (384, 96), dtype=jnp.float32) * std,
            "b2": jnp.zeros(96, dtype=jnp.float32),
        })
    params["ln_f_g"] = jnp.ones(96, dtype=jnp.float32)
    params["ln_f_b"] = jnp.zeros(96, dtype=jnp.float32)
    params["read_W"] = jax.random.normal(kread, (96, 512), dtype=jnp.float32) * std
    params["read_b"] = jnp.zeros(512, dtype=jnp.float32)
    return params


def num_params(params):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def forward(params, ids, causal=True, return_attn=False):
    """ids: [B, T] int. Returns logits [B, T, 512] and (optionally) the
    per-block attention matrices [B, 4, T, T]."""
    B, T = ids.shape
    x = params["tok_emb"][ids] + params["pos_emb"][None, :T]
    attns = []
    for blk in params["blocks"]:
        x, attn = _block(blk, x, causal, return_attn)
        if return_attn:
            attns.append(attn)
    x = _ln(x, params["ln_f_g"], params["ln_f_b"])
    logits = _linear(x, params["read_W"], params["read_b"])
    return (logits, attns) if return_attn else (logits, None)


def loss_batch(params, ids, targets):
    """Mean CE over all B*T positions ([B, T, V] -> per-position [B, T])."""
    logits, _ = forward(params, ids)
    return stable_ce(logits, targets)
