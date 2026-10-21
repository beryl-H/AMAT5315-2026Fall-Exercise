"""Algorithm 2: DDPM sampling with the trained noise predictor.

Indexing convention (single, used everywhere):
- betas has length T and is indexed beta[t-1] for mathematical level t
  (alpha[t-1] = 1 - betas[t-1]).
- abar has length T+1 and is indexed abar[t] (abar[0] = 1).
- The network's time features use phase (t-1)*omega_j.

Reverse step (no clipping, no added noise at t = 1):
    mu_t = (x - beta[t]/sqrt(1-abar[t]) * eps_theta(x, t)) / sqrt(alpha[t])
    x <- mu_t + sqrt(beta[t]) * z,  z ~ N(0, I) only for t > 1.

RNG convention: in fresh-key mode the sampler takes an explicit `key` and
draws one independent Gaussian array per level (a single Normal draw of shape
(T, batch, 2) gives mutually independent entries; no implicit or unadvanced
key is ever used). In replay mode a supplied {t: array} dict is used
verbatim (the t=1 entry is ignored), so demonstrations can share one initial
Gaussian and one reverse-step noise stream across training stages.
"""


import jax
import jax.numpy as jnp


def ddpm_sample(predict, params, xT, betas, abar, key=None,
                reverse_noise=None, record_states=False):
    """Run Algorithm 2 from initial states xT (at level T = len(betas)).

    predict(params, x, t) -> eps_theta; t is an int array of levels.
    Exactly one of `key` (fresh noise per step) or `reverse_noise`
    (replay dict {t: array}, t = T..1) must be supplied.
    Returns x0, or (x0, [x_T, x_{T-1}, ..., x_0]) if record_states.
    """
    T = int(betas.shape[0])
    if reverse_noise is not None and key is not None:
        raise ValueError("pass either key or reverse_noise, not both")
    if reverse_noise is not None:
        z_stack = jnp.stack(
            [jnp.asarray(reverse_noise[t]) if t > 1
             else jnp.zeros_like(xT)  # t = 1 adds no noise
             for t in range(T, 0, -1)])  # (T, batch, 2), row i <-> level T-i
    elif key is not None:
        z_stack = jax.random.normal(key, (T, *xT.shape), dtype=xT.dtype)
    else:
        raise ValueError("fresh-key mode requires an explicit sampler key")

    betas_j = jnp.asarray(betas)
    alpha_j = 1.0 - betas_j
    abar_j = jnp.asarray(abar)
    predict_jit = jax.jit(lambda x, t: predict(params, x, t))

    def body(x, i):
        t = T - i  # mathematical level
        t_arr = jnp.full((x.shape[0],), t, dtype=jnp.int32)
        eps_hat = predict_jit(x, t_arr)
        beta_t = betas_j[t - 1]
        alpha_t = alpha_j[t - 1]
        a_t = abar_j[t]
        mu = (x - beta_t / jnp.sqrt(1.0 - a_t) * eps_hat) / jnp.sqrt(alpha_t)
        z = z_stack[i]
        x_next = jnp.where(t > 1, mu + jnp.sqrt(beta_t) * z, mu)
        return x_next, x_next

    x0, xs = jax.lax.scan(body, xT, jnp.arange(T))
    if record_states:
        # states list: x_T, x_{T-1}, ..., x_1, x_0
        states = [xT] + [xs[i] for i in range(T)]
        return x0, states
    return x0
