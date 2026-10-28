"""Part 4 decoder-correction tests (pi-task-08 TASK A), written BEFORE the fix.

The symmetric variant (preserved verbatim in grok_symmetric_variant.py) is
answer-invariant under swapping a and b and cannot memorize the worksheet
random-label control (393 conflicting unordered training pairs; max train
accuracy 0.860786 < 0.99). The corrected architecture is the Part 2 decoder
structure — learned positional embeddings + causal mask — with ONLY
LayerNorm removed. Exactly 62,497 parameters (62,305 + 192 positional).
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import grok as gk
import grok_symmetric_variant as gsv

ROOT = Path(__file__).resolve().parent.parent


# ---- architecture: decoder correction ----------------------------------------

def test_decoder_params_exactly_62497_with_pos_no_layernorm():
    params = gk.init_params(jax.random.PRNGKey(0))
    assert params["pos"].shape == (3, 64)          # 3x64 learned positions
    assert gk.count_params(params) == 62497        # 62305 + 192
    paths = [jax.tree_util.keystr(k) for k, _ in
             jax.tree_util.tree_flatten_with_path(params)[0]]
    assert not any("ln" in p.lower() for p in paths)   # LayerNorm removed only


def test_symmetric_variant_preserved_at_62305():
    # provenance: the invalidated variant is kept and still counts 62305
    params = gsv.init_params(jax.random.PRNGKey(0))
    assert gsv.count_params(params) == 62305
    assert "pos" not in params


def test_decoder_causal_mask_upper_triangle_exactly_zero():
    params = gk.init_params(jax.random.PRNGKey(0))
    attn = gk.attention_weights(params, jnp.asarray([[1, 2, 97]]))
    assert attn.shape == (1, 4, 3, 3)              # 1 layer, 4 heads
    above = jnp.triu(attn[0], k=1)                 # per head
    assert float(jnp.max(jnp.abs(above))) == 0.0
    np.testing.assert_allclose(np.asarray(attn[0].sum(-1)), 1.0, atol=1e-6)


def test_swap_ab_not_structurally_invariant():
    """With positions and a causal mask, [a,b,=] and [b,a,=] must give
    different answer logits on the SAME parameters — and actual gradients
    must differ too (structural, not an accidental zero)."""
    params = gk.init_params(jax.random.PRNGKey(0))
    diffs = []
    for a, b in [(5, 40), (0, 96), (13, 13), (31, 77)]:
        la = gk.forward(params, jnp.asarray([[a, b, 97]]))[0, 2]
        lb = gk.forward(params, jnp.asarray([[b, a, 97]]))[0, 2]
        diffs.append(float(jnp.max(jnp.abs(la - lb))))
    assert max(diffs) > 1e-6                       # at least some pairs differ
    # actual gradient check through the real loss path: gradient w.r.t. the
    # positional embedding must differ between the two orders
    def loss_pos(p, order):
        tok = jnp.asarray([[5, 40, 97]]) if order == 0 else jnp.asarray([[40, 5, 97]])
        q = dict(p)
        return gk.loss_acc(q, tok, jnp.asarray([45]))[0]
    g1 = jax.grad(loss_pos)(params, 0)["pos"]
    g2 = jax.grad(loss_pos)(params, 1)["pos"]
    assert float(jnp.max(jnp.abs(g1 - g2))) > 1e-9


def test_symmetric_variant_is_invariant_control():
    # the preserved variant must still BE invariant (contrast for the fix)
    params = gsv.init_params(jax.random.PRNGKey(0))
    la = gsv.forward(params, jnp.asarray([[5, 40, 97]]))[0, 2]
    lb = gsv.forward(params, jnp.asarray([[40, 5, 97]]))[0, 2]
    assert float(jnp.max(jnp.abs(la - lb))) == 0.0


def test_output_97_and_api_unchanged():
    params = gk.init_params(jax.random.PRNGKey(0))
    logits = gk.forward(params, jnp.asarray(gk.build_dataset()["inputs"][:4]))
    assert logits.shape == (4, 3, 97)
    loss, acc = gk.loss_acc(params, jnp.asarray(gk.build_dataset()["inputs"][:8]),
                            jnp.asarray(gk.build_dataset()["targets"][:8]))
    assert np.isfinite(float(loss)) and 0.0 <= float(acc) <= 1.0
    # full-batch train step and stop rule unchanged
    state = gk.grok_adamw_init(params)
    p2, s2, l = gk.train_step(params, state,
                              jnp.asarray(gk.build_dataset()["inputs"][:8]),
                              jnp.asarray(gk.build_dataset()["targets"][:8]),
                              3e-3, 1)
    assert np.isfinite(float(l))
    log = [{"step": 100 * i, "train_acc": 1.0, "test_acc": 0.995} for i in range(1, 4)]
    res = gk.analyze_log(log, max_steps=10000)
    assert res["grok_step"] == 100 and res["stop_step"] == 1100


# ---- random-label conflicts are expressible -----------------------------------

def test_random_label_conflicts_count_and_are_expressible():
    """The shuffled-label control contains conflicting unordered pairs; the
    corrected decoder can actually memorize them (the symmetric variant
    cannot — that was the verified defect)."""
    true = gk.build_dataset()["targets"]
    shuffled = gk.random_labels()
    a = np.repeat(np.arange(97), 97)
    b = np.tile(np.arange(97), 97)
    pos_map = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(a, b))}
    train = set(gk.fixed_split(0.30)["train"].tolist())
    conflicts = 0
    conflict_pairs = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x < y:
            j = pos_map[(int(y), int(x))]
            if shuffled[i] != shuffled[j] and i in train and j in train:
                conflicts += 1
                conflict_pairs.append((i, j))
    assert conflicts == 393                        # frozen fact from the audit
    # capacity check: train the ACTUAL decoder on the conflicting set
    idx = np.array([i for pair in conflict_pairs for i in pair])
    tok = jnp.asarray(gk.build_dataset()["inputs"][idx])
    tgt = jnp.asarray(shuffled[idx])
    params = gk.init_params(jax.random.PRNGKey(0))
    state = gk.grok_adamw_init(params)
    for step in range(1, 401):
        params, state, _ = gk.train_step(params, state, tok, tgt, 3e-3, step)
    _, acc = gk.loss_acc(params, tok, tgt)
    assert float(acc) >= 0.999, f"conflicting pairs not memorizable: {float(acc)}"
