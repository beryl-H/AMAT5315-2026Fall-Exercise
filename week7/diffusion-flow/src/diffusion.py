"""Shared target and forward diffusion for the Gaussian mixture (Part 1).

Equations follow the Week 7 lecture sheet exactly:
- target: x0 = mu_k + 0.3 (z1, z2), k uniform in {1..4},
  centres (3,0),(0,3),(-3,0),(0,-3), covariance 0.09 I.
- forward: x_t = sqrt(alpha_t) x_{t-1} + sqrt(beta_t) z, beta_t linear
  1e-4 -> 0.02, T = 1000, alpha_t = 1 - beta_t, abar_t = prod alpha_s, abar_0 = 1.
"""

import jax
import jax.numpy as jnp

T_LEVELS = 1000
CENTRES = jnp.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]],
                    dtype=jnp.float32)
CLUSTER_STD = 0.3


def sample_target(key, n):
    """Draw n exact samples from the four-cluster Gaussian mixture."""
    key_k, key_z = jax.random.split(key)
    k = jax.random.randint(key_k, (n,), 0, 4)
    z = jax.random.normal(key_z, (n, 2), dtype=jnp.float32)
    return CENTRES[k] + CLUSTER_STD * z


def beta_schedule(t_levels=T_LEVELS):
    """beta_t linear from 1e-4 (t=1) to 0.02 (t=1000)."""
    return jnp.linspace(1e-4, 0.02, t_levels, dtype=jnp.float32)


def abar_from_betas(betas):
    """abar_t = prod_{s=1..t} alpha_s, with abar_0 = 1 (length T+1)."""
    alphas = 1.0 - betas
    log_abar = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float32),
         jnp.cumsum(jnp.log(alphas.astype(jnp.float32)))]
    )
    return jnp.exp(log_abar)


def forward_step(x, alpha, beta, key):
    """One forward step x_t = sqrt(alpha) x + sqrt(beta) z."""
    z = jax.random.normal(key, x.shape, dtype=x.dtype)
    return jnp.sqrt(alpha) * x + jnp.sqrt(beta) * z


def noise_to_level(x0, t, abar, key):
    """Direct noising to mathematical level t (Equation 7).

    t = 0 returns x0 unchanged; t >= 1 uses abar[t-1] and fresh noise.
    """
    if t == 0:
        return x0
    a = abar[t - 1]
    eps = jax.random.normal(key, x0.shape, dtype=x0.dtype)
    return jnp.sqrt(a) * x0 + jnp.sqrt(1.0 - a) * eps
