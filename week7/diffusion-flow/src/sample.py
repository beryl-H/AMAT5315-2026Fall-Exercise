"""Algorithm 2: DDPM sampling with the trained noise predictor.

Indexing convention (single, used everywhere):
- betas has length T and is indexed beta[t-1] for mathematical level t
  (alpha[t-1] = 1 - betas[t-1]).
- abar has length T+1 and is indexed abar[t] (abar[0] = 1).
Reverse step (no clipping, no added noise at t = 1):
    mu_t = (x - beta[t]/sqrt(1-abar[t]) * eps_theta(x, t)) / sqrt(alpha[t])
    x <- mu_t + sqrt(beta[t]) * z,  z ~ N(0, I) only for t > 1.
"""

import jax
import jax.numpy as jnp


def ddpm_sample(predict, params, xT, betas, abar, reverse_noise=None,
                record_states=False):
    """Run Algorithm 2 from initial states xT (at level T).

    predict(params, x, t) -> eps_theta; t is an int array of levels.
    reverse_noise: optional dict {t: array} supplying the shared reverse-step
    Gaussian draws (used by the training-movie demonstration); when None,
    fresh noise is drawn from `key`. The t=1 entry, if given, is ignored.
    record_states: if True, also return the list of states x_T..x_0.
    """
    key = jax.random.PRNGKey(0)
    x = xT
    T = betas.shape[0]
    states = [x] if record_states else None
    for t in range(T, 0, -1):
        t_arr = jnp.full((x.shape[0],), t, dtype=jnp.int32)
        eps_hat = predict(params, x, t_arr)
        beta_t = betas[t - 1]
        alpha_t = 1.0 - beta_t
        a_t = abar[t]
        mu = (x - beta_t / jnp.sqrt(1.0 - a_t) * eps_hat) / jnp.sqrt(alpha_t)
        if t > 1:
            if reverse_noise is not None:
                z = reverse_noise[t]
            else:
                z = jax.random.normal(key, x.shape, dtype=x.dtype)
            x = mu + jnp.sqrt(beta_t) * z
        else:
            x = mu
        if record_states:
            states.append(x)
    return (x, states) if record_states else x
