"""Shared NN kernels: stable cross-entropy, AdamW (decoupled, 2D-weight decay
only, bias correction from step 1), LR schedule. Float32 in training; the f64
equivalence checks live in tests only."""
import jax
import jax.numpy as jnp
import numpy as np

LR_SCHEDULES = {  # frozen (SPEC)
    "mlp": {"steps": 4000, "peak": 0.006, "warmup": 200, "end_frac": 0.1},
    "transformer": {"steps": 6000, "peak": 0.003, "warmup": 300, "end_frac": 0.1},
}
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WD = 0.01  # 2D weights only


def stable_ce(logits, targets):
    """CE = -log softmax(logits)[target] computed as
    softplus(logsumexp(non-target logits) - target logit).

    Algebraically identical to logsumexp(all) - t but numerically stable and
    well-gradiented when the target logit dominates. logits: [..., V] float32;
    targets: [...] integer with the same leading shape."""
    V = logits.shape[-1]
    flat_logits = logits.reshape(-1, V)
    flat_t = targets.reshape(-1)
    rows = jnp.arange(flat_logits.shape[0])
    t_logit = flat_logits[rows, flat_t]
    masked = flat_logits.at[rows, flat_t].set(-jnp.inf)  # non-target logits
    lse_others = jax.scipy.special.logsumexp(masked, axis=-1)
    ce = jax.nn.softplus(lse_others - t_logit)
    return ce.reshape(targets.shape)


def init_adamw(params):
    leaves, _ = jax.tree_util.tree_flatten(params)
    zeros = [jnp.zeros_like(x) for x in leaves]
    m = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), zeros)
    v = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), zeros)
    return {"m": m, "v": v, "step": 0}


def adamw_step(params, state, grads, lr, betas=ADAM_BETAS, wd=WD, eps=ADAM_EPS):
    """One AdamW update. Bias correction uses step starting at 1 (no hidden
    reset); weight decay is decoupled and applied to 2D weights only."""
    b1, b2 = betas
    t = state["step"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)

    def upd(p, m_i, v_i):
        mh = m_i / (1 - b1 ** t)
        vh = v_i / (1 - b2 ** t)
        u = mh / (jnp.sqrt(vh) + eps)
        if p.ndim == 2:
            u = u + wd * p
        return p - lr * u

    new_params = jax.tree_util.tree_map(upd, params, m, v)
    return new_params, {"m": m, "v": v, "step": t}


def schedule_lr(step, step_min, step_max, peak, warmup, end_frac):
    """Linear warmup 0 -> peak over `warmup` steps, then cosine decay to
    end_frac * peak at step_max."""
    if step < warmup:
        return peak * step / warmup
    progress = min(max((step - warmup) / max(step_max - warmup, 1), 0.0), 1.0)
    cos = 0.5 * (1 + np.cos(np.pi * progress))
    return peak * (end_frac + (1 - end_frac) * cos)


def global_norm(grads):
    leaves = jax.tree_util.tree_leaves(grads)
    return float(np.sqrt(sum(float(jnp.sum(g * g)) for g in leaves)))


def clip_grads(grads, max_norm=1.0):
    gnorm = global_norm(grads)
    if gnorm <= max_norm:
        return grads, gnorm
    scale = max_norm / gnorm
    return jax.tree_util.tree_map(lambda g: g * scale, grads), gnorm


def global_norm_jax(grads):
    """Pure-JAX global L2 norm over all leaves (jit-compatible; no host float)."""
    leaves = jax.tree_util.tree_leaves(grads)
    total = jnp.zeros((), dtype=jnp.float32)
    for g in leaves:
        total = total + jnp.sum(g * g)
    return jnp.sqrt(total)


def clip_grads_jax(grads, max_norm):
    """Pure-JAX global-norm clipping (jit-compatible). Returns scaled grads."""
    gnorm = global_norm_jax(grads)
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(gnorm, 1e-12))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)
