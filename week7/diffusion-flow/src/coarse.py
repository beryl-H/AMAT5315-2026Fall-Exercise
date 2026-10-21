"""Part 3: coarse DDPM jumps between selected original levels.

Jump from original level t to s (s may skip levels), a = abar[t]/abar[s]:
    mu = (x - (1-a)/sqrt(1-abar[t]) * eps_theta(x, t)) / sqrt(a)
    x_next = mu + sqrt(1-a) * z   for s > 0, no noise for s = 0.
For adjacent levels a = alpha_t so 1-a = beta_t and this is exactly the
Part 1 update (checked by test). The network receives the ORIGINAL t.
"""

import numpy as np


def _validate_levels(levels, t_levels):
    lv = np.asarray(levels)
    if lv.ndim != 1 or len(lv) < 2:
        raise ValueError("levels must be 1-D with >= 2 entries")
    if not np.issubdtype(lv.dtype, np.integer):
        raise ValueError("levels must be integers")
    if not np.all(lv[1:] < lv[:-1]):
        raise ValueError("levels must be strictly decreasing")
    if lv[0] != t_levels:
        raise ValueError("levels must start at T")
    if lv.min() < 0 or lv.max() > t_levels:
        raise ValueError("levels out of range")
    return lv


def coarse_ddpm_sample(predict, params, xT, betas, abar, levels,
                       reverse_noise=None, key=None, record_states=False):
    """Random sampler on a coarse grid of original levels.

    reverse_noise: dict {t: array} for the jump FROM level t (entries for
    s = 0 ignored), or key for fresh draws of shape (len(levels)-1, ...).
    """
    T = int(np.asarray(betas).shape[0])
    lv = _validate_levels(levels, T)
    import jax
    import jax.numpy as jnp
    import jax.numpy as jnp
    abar_j = jnp.asarray(abar)
    betas_j = jnp.asarray(betas)
    if reverse_noise is not None and key is not None:
        raise ValueError("pass either key or reverse_noise, not both")
    if reverse_noise is not None:
        z_stack = jnp.stack(
            [jnp.asarray(reverse_noise[int(t)])
             if int(lv[i + 1]) > 0 else jnp.zeros_like(xT)
             for i, t in enumerate(lv[:-1])])
    elif key is not None:
        z_stack = jax.random.normal(key, (len(lv) - 1, *xT.shape),
                                    dtype=xT.dtype)
    else:
        raise ValueError("fresh mode requires an explicit key")

    lv_j = jnp.asarray(lv, dtype=jnp.int32)

    def body(x, i):
        t = lv_j[i]             # tracer-safe original level
        s = lv_j[i + 1]
        a = abar_j[t] / abar_j[s]
        t_arr = jnp.full((x.shape[0],), t, dtype=jnp.int32)
        eps_hat = predict(params, x, t_arr)
        mu = (x - (1.0 - a) / jnp.sqrt(1.0 - abar_j[t]) * eps_hat) \
            / jnp.sqrt(a)
        x_next = jnp.where(s > 0, mu + jnp.sqrt(1.0 - a) * z_stack[i], mu)
        return x_next, x_next

    x0, xs = jax.lax.scan(body, xT, np.arange(len(lv) - 1))
    if record_states:
        return x0, xs
    return x0
