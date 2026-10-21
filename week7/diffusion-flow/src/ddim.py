"""Part 2: deterministic DDIM sampling and the analytic mixture noise.

Indexing convention (same as Part 1): betas[t-1], abar[t] with abar[0]=1;
the predictor receives the ORIGINAL integer level t (never a loop index).

DDIM jump from level t to level s (no clipping of states or clean
estimates; the mixture's centres are far outside [-1,1]):
    eps_hat  = eps_theta(x, t)
    x0_hat   = (x - sqrt(1-abar[t]) * eps_hat) / sqrt(abar[t])
    x_next   = sqrt(abar[s]) * x0_hat + sqrt(1-abar[s]) * eps_hat
At s = 0 the noise contribution vanishes and the clean estimate is returned.

Exact conditional-mean noise (analytic derivation). At level t with
b = abar[t], the forward marginal is a mixture of
N(sqrt(b) mu_k, v I) with v = 0.09 b + (1-b), because
x_t = sqrt(b) mu_k + sqrt(b) 0.3 z + sqrt(1-b) eps per component.
The responsibilities w_k are the component posteriors at x (computed with a
stable softmax on the log-densities). Since x_t - sqrt(b) x0 = sqrt(1-b) eps
and the component-wise posterior mean of x0 is
mu_k + (0.09 sqrt(b)/v)(x - sqrt(b) mu_k),
    E[eps | x] = (x - sqrt(b) E[x0|x]) / sqrt(1-b)
               = sqrt(1-b)/v * (x - sqrt(b) * sum_k w_k mu_k),
using v - 0.09 b = 1 - b. Equivalently E[eps|x] = -sqrt(1-b) * grad log p_t,
an independent identity checked in the tests. At t = 0 the factor
sqrt(1-b) = 0 makes the result exactly zero.
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.diffusion import CENTRES, CLUSTER_STD


def _validate_levels(levels, t_levels):
    lv = np.asarray(levels)
    if lv.ndim != 1 or len(lv) < 2:
        raise ValueError("levels must be a 1-D array with at least 2 entries")
    if not np.issubdtype(lv.dtype, np.integer):
        raise ValueError("levels must be integers")
    if not np.all(lv[1:] < lv[:-1]):
        raise ValueError("levels must be strictly decreasing")
    if lv[0] != t_levels or lv[-1] != 0:
        raise ValueError("levels must start at T and end at 0")
    if lv.min() < 0 or lv.max() > t_levels:
        raise ValueError("levels out of range [0, T]")
    return jnp.asarray(lv, dtype=jnp.int32)


def ddim_sample(predict, params, xT, levels, abar, record_states=False):
    """Deterministic DDIM (Algorithm 3) over an explicit level grid.

    predict(params, x, t) -> eps_hat with t the original integer levels.
    No randomness anywhere: identical inputs give identical outputs.
    Returns x0, or (x0, levels, states) with states[k] the state after
    jump k (level levels[k+1]) when record_states=True.
    """
    T = int(np.asarray(abar).shape[0] - 1)
    lv = _validate_levels(levels, T)
    abar_j = jnp.asarray(abar)

    def body(x, i):
        t = lv[i]
        s = lv[i + 1]
        t_arr = jnp.full((x.shape[0],), t, dtype=jnp.int32)
        eps_hat = predict(params, x, t_arr)
        b_t = abar_j[t]
        b_s = abar_j[s]
        x0_hat = (x - jnp.sqrt(1.0 - b_t) * eps_hat) / jnp.sqrt(b_t)
        x_next = jnp.sqrt(b_s) * x0_hat + jnp.sqrt(1.0 - b_s) * eps_hat
        return x_next, x_next

    x0, xs = jax.lax.scan(body, xT, jnp.arange(len(lv) - 1))
    if record_states:
        return x0, lv, xs
    return x0


def mixture_exact_noise(x, t, abar, centres=CENTRES, cluster_std=CLUSTER_STD):
    """E[eps | x_t = x] for the equal-weight Gaussian mixture (analytic)."""
    b = jnp.asarray(abar)[t]
    v = jnp.float32(cluster_std) ** 2 * b + (1.0 - b)
    sqrt_b = jnp.sqrt(b)
    d = x[:, None, :] - sqrt_b * centres[None]      # (N, K, 2)
    logp = -0.5 * (d ** 2).sum(-1) / v              # up to shared constants
    w = jax.nn.softmax(logp, axis=-1)               # responsibilities
    m = (w[..., None] * centres[None]).sum(axis=1)  # sum_k w_k mu_k
    return jnp.sqrt(1.0 - b) / v * (x - sqrt_b * m)


def exact_predictor(abar, centres=CENTRES, cluster_std=CLUSTER_STD):
    """Wrap the analytic noise as a predictor(params, x, t) for DDIM."""
    def predict(params, x, t):
        # t is (possibly a tracer) array of identical per-jump levels;
        # abar indexing supports dynamic scalar indices.
        return mixture_exact_noise(x, t[0], abar, centres, cluster_std)
    return predict
