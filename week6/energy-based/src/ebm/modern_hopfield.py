"""Modern Hopfield network / softmax attention (Week 6 Part 4).

Update (sheet eq. 17): x' = X softmax(beta X^T x) with the patterns as
COLUMNS of X (784, P). Fixed decision policies (recorded before any
outcome): sign(0) = 0 for the overlap metric; nearest stored pattern by
spin overlap with first-index argmax tie rule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ebm.rbm import binarize  # same half-intensity threshold (>=128)

LOWER_ROWS = 14  # rows 14..27 are replaced by random spins in queries


def spinize(images):
    """(B, 28, 28) uint8 images -> (784, B) spin patterns 2v-1 (columns)."""
    v = binarize(np.asarray(images).reshape(-1, 784))     # (B, 784)
    return (2.0 * v.T - 1.0).astype(jnp.float32)          # (784, B)


def modern_update(X, x, beta: float):
    """x' = X @ softmax(beta * X^T x), float32 (sheet eq. 17)."""
    X = jnp.asarray(X, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.float32)
    w = jax.nn.softmax(beta * (X.T @ x))
    return X @ w


def attention_update(X, x, beta: float):
    """The SAME update via jax.nn.dot_product_attention with scale=beta.

    query = x as a length-1 sequence; keys = values = X.T (P, 784). The
    attention output softmax(beta k.q) v is exactly the modern update.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.float32)
    # layout: query (T, N, H) = (1, 1, 784); keys/values (S, K, H) =
    # (P, 1, 784). The attention output softmax(beta k.q) v is exactly the
    # modern update.
    out = jax.nn.dot_product_attention(
        x[None, None, :], X.T[:, None, :], X.T[:, None, :], scale=beta)
    return out[0, 0]


def energy_modern(X, x, beta: float):
    """E(x) = -logsumexp(beta X^T x)/beta + 0.5 x.x (sheet eq. 16)."""
    X = jnp.asarray(X, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.float32)
    u = beta * (X.T @ x)
    return -jnp.logaddexp.reduce(u) / beta + 0.5 * jnp.dot(x, x)


def sign_overlap(target, s):
    """Spin overlap (1/N) sum target_i * sign(s_i); sign(0) = 0 policy."""
    target = jnp.asarray(target)
    sg = jnp.sign(jnp.asarray(s))
    return jnp.sum(target * sg) / target.size


def nearest_pattern(X, s):
    """Index of the stored pattern nearest to sign(s) by spin overlap;
    first index wins ties (fixed policy)."""
    sg = jnp.sign(jnp.asarray(s))
    ov = jnp.asarray(X).T @ sg / X.shape[0]
    return int(jnp.argmax(ov))


def retrieval_success(X, target_idx: int, s, threshold: float = 0.95):
    """Full criterion: overlap >= threshold AND target is nearest."""
    return bool(sign_overlap(X[:, target_idx], s) >= threshold
                and nearest_pattern(X, s) == target_idx)
