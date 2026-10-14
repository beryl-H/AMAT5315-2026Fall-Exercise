"""Classical Hopfield network (Week 6 Part 3), JAX/CPU.

Hebbian couplings K = Xi^T Xi with exact int32 arithmetic (cast BEFORE the
matmul; diagonal removed). Sign/tie decisions use the integer fields
h = K s (the 1/N normalization is irrelevant to the sign and avoids float32
fictitious ties). Async zero-temperature updates s_i <- sign(h_i); a zero
local field leaves the spin unchanged. Sweeps repeat until one full sweep
flips nothing. Energy E(s) = -0.5 s (K/N) s; overlap m_mu = xi_mu . s / N;
identity E(s) = -N/2 sum_mu m_mu^2 + P/2 (sheet eqs. 12-13).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def random_patterns(key, *, P: int, N: int):
    """iid random +/-1 patterns, dtype int32."""
    return jax.random.bernoulli(key, 0.5, (P, N)).astype(jnp.int32) * 2 - 1


def hebb_k(Xi):
    """K = Xi^T Xi as exact int32 with zero diagonal (cast before matmul)."""
    Xi = jnp.asarray(Xi).astype(jnp.int32)
    K = Xi.T @ Xi
    return K - jnp.diag(jnp.diag(K))


def energy(s, K, N: int | None = None) -> float:
    """E(s) = -0.5 s (K/N) s (float normalization for reporting only)."""
    sf = jnp.asarray(s, dtype=jnp.float32)
    Kf = jnp.asarray(K, dtype=jnp.float32)
    if N is None:
        N = Kf.shape[0]
    return float(-0.5 * (sf @ Kf @ sf) / N)


def overlap(Xi, s):
    """m_mu = xi_mu . s / N for every stored pattern."""
    Xf = jnp.asarray(Xi, dtype=jnp.float32)
    sf = jnp.asarray(s, dtype=jnp.float32)
    return Xf @ sf / Xf.shape[-1]


def spin_step(s, h, K, i: int):
    """One production async update of spin i from integer fields h = K s.

    Returns (s_new, h_new); the fields are maintained incrementally:
    flipping s_i by delta changes h_j by K[j, i] * delta. A zero field
    leaves the spin unchanged (exact integer comparison).
    """
    si = jnp.where(h[i] > 0, 1, jnp.where(h[i] < 0, -1, s[i]))
    delta = si - s[i]
    s_new = s.at[i].add(delta)
    h_new = h + K[:, i] * delta
    return s_new, h_new


def unstable_fraction(K, Xi):
    """Fraction of pattern bits with xi^mu_i h_i < 0 over ALL stored
    patterns; zero fields are not counted as unstable."""
    K = jnp.asarray(K)
    Xi = jnp.asarray(Xi)
    H = (K.astype(jnp.float32) @ Xi.T.astype(jnp.float32)).astype(jnp.int32)
    H = H.T                                  # (P, N): field on each pattern bit
    negative = (Xi * H) < 0
    return jnp.mean(negative)


def recall(K, starts, key, *, max_sweeps: int, perms=None):
    """Batched asynchronous zero-temperature recall for ONE pattern set.

    K: (N, N) int32 Hebb matrix. starts: (C, N) int32 initial states.
    key: PRNG key for independent update orders (per chain, per sweep);
    ignored when `perms` (max_sweeps, C, N) is supplied explicitly (tests).

    Returns dict: states (C, N), sweeps_used (C,) = number of sweeps until
    (including) the first zero-flip sweep, converged (C,), flips history
    (max_sweeps, C).
    """
    K = jnp.asarray(K)
    starts = jnp.asarray(starts)
    C, N = starts.shape

    h0 = K @ starts.T          # (N, C) integer fields
    s0 = starts.T              # (N, C)

    if perms is not None:
        perms_t = jnp.transpose(jnp.asarray(perms), (0, 2, 1))  # (S, N, C)

        def make_perm(sweep, key_carry):
            p = jax.lax.dynamic_index_in_dim(perms_t, sweep, axis=0,
                                             keepdims=False)
            return p, key_carry
    else:
        def make_perm(sweep, key_carry):
            key_carry, k1 = jax.random.split(key_carry)
            ks = jax.random.split(k1, C)
            p = jax.vmap(lambda kk: jax.random.permutation(kk, N))(ks)
            return jnp.transpose(p), key_carry

    def sweep_body(carry, i_col):
        s, h, flips = carry          # s,h: (N, C); flips: (C,)
        idx = i_col                  # (C,) spin index per chain
        ar = jnp.arange(C)
        h_i = h[idx, ar]             # (C,)
        s_i = s[idx, ar]             # (C,)
        new = jnp.where(h_i > 0, 1, jnp.where(h_i < 0, -1, s_i))
        delta = new - s_i            # (C,) in {0, +/-2}
        s = s.at[idx, ar].set(new)
        h = h + K[:, idx] * delta[None, :]   # (N, C)
        flips = flips + (delta != 0).astype(jnp.int32)
        return (s, h, flips), None

    def do_sweep(carry):
        s, h, key, sweep, hist = carry
        perm, key = make_perm(sweep, key)      # (N, C)
        (s, h, flips), _ = jax.lax.scan(
            sweep_body, (s, h, jnp.zeros(C, jnp.int32)), perm, length=N)
        return (s, h, key, sweep + 1, hist.at[sweep].set(flips))

    # run sweeps until a full sweep flips nothing (or the safety cap)
    def cond2(carry):
        _, _, _, sweep, hist = carry
        last = jnp.where(sweep == 0, jnp.ones(C, jnp.int32),
                         hist[sweep - 1])
        return (sweep < max_sweeps) & jnp.any(last > 0)

    init = (s0, h0, key, jnp.asarray(0, jnp.int32),
            jnp.zeros((max_sweeps, C), jnp.int32))
    s, h, key, sweep, hist = jax.lax.while_loop(cond2, do_sweep, init)

    # sweeps_used: index of first zero-flip sweep + 1
    nonzero = hist > 0
    first_zero = jnp.argmax(~nonzero, axis=0)        # (C,)
    had_zero = jnp.any(~nonzero, axis=0)
    sweeps_used = jnp.where(had_zero, first_zero + 1, sweep)
    # converged iff the LAST executed sweep (index sweep-1, always >= 0)
    # flipped nothing
    converged = ~jnp.any(hist[sweep - 1] > 0)
    return {"states": s.T, "sweeps_used": sweeps_used,
            "converged": converged, "flips": hist}


# ---------------------------------------------------------------- NumPy
def numpy_recall_reference(K, starts, perms):
    """Independent NumPy implementation of the same deterministic rule.

    K: (N, N) integer Hebb matrix; starts: (C, N); perms: (S, C, N) update
    orders. Fields are recomputed exactly (no incremental bookkeeping), so
    agreement with the JAX incremental implementation is a real check.
    Returns states (C, N).
    """
    K = np.asarray(K, dtype=np.int64)
    states = np.asarray(starts, dtype=np.int64).copy()
    C, N = states.shape
    perms = np.asarray(perms)
    for sweep in range(perms.shape[0]):
        flipped = False
        for c in range(C):
            for i in perms[sweep, c]:
                h = K @ states[c]
                if h[i] > 0 and states[c][i] != 1:
                    states[c][i] = 1
                    flipped = True
                elif h[i] < 0 and states[c][i] != -1:
                    states[c][i] = -1
                    flipped = True
        if not flipped:
            break
    return states
