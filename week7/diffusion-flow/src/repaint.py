"""Part 5: RePaint inpainting (Algorithm 6, lecture pages 17-19).

Original mathematical levels [1000, 980, ..., 20, 0]; 50 jumps; 2 attempts
per jump except the final one -> 99 predictor evaluations and 49 forward
re-noises. The predictor receives the MODEL INDEX t-1 (subtraction happens
exactly once here when the default cifar_unet_predict is used).

Per attempt from level t to s (a = abar[t]/abar[s]):
  retry r>1:  x_t = sqrt(a) * previous_x_s + sqrt(1-a) * f   (fresh f)
  eps    = predictor(x_t, t-1)
  x0hat  = clip((x_t - sqrt(1-abar[t]) * eps) / sqrt(abar[t]), -1, 1)
           — the ONLY algorithmic clipping; noisy/merged states unclipped
  mean   = sqrt(abar[s])*(1-a)/(1-abar[t]) * x0hat
           + sqrt(a)*(1-abar[s])/(1-abar[t]) * x_t
  var    = (1-abar[s])*(1-a)/(1-abar[t])        (POSTERIOR variance,
           not the coarse-DDPM variance 1-a; var = 0 at s = 0)
  proposal = mean + sqrt(var) * xi              (fresh xi)
  known    = sqrt(abar[s]) * observations + sqrt(1-abar[s]) * eta (fresh eta)
  x_s      = mask * known + (1 - mask) * proposal

The API receives observations and mask ONLY — never the withheld truth.

JIT design: ONE module-level compiled core `_repaint_core` receives the
weights, observations, mask, initial state, schedule/event arrays and the
RNG key as explicit DYNAMIC arguments — no large weight closure constants.
The only closed-over value is the small predictor FUNCTION, keyed in a
module cache so all seeds/images reuse the same compiled executable.

RNG convention: root key is split into (k_initial, k_carry); the carry
advances once per event, each event splitting into (k_f, k_xi, k_eta,
k_next). A standard-normal f is drawn at every event from k_f but USED
only when is_retry (the unused draws are deterministic and reproducible
from the recorded root key; all consumed keys are distinct). Tests may
instead pass explicit noise_stream/xT.
"""

import jax
import jax.numpy as jnp
import numpy as np

OFFICIAL_LEVELS = list(range(1000, 19, -20)) + [0]  # 1000..20 then 0

_CORE_CACHE = {}


def repaint_events(levels, attempts=2):
    """Static event list [{t, s, is_retry}]: `attempts` per jump, the final
    jump to 0 taken once. Levels validated strictly (no float truncation)."""
    if isinstance(attempts, float) or not isinstance(attempts, int) \
            or attempts < 1:
        raise ValueError(f"attempts must be a positive int, got {attempts!r}")
    if any(isinstance(v, float) or not isinstance(v, (int, np.integer))
           for v in levels):
        raise ValueError("levels must be integers (no float truncation)")
    lv = [int(v) for v in levels]
    if len(lv) < 2 or any(lv[i + 1] >= lv[i] for i in range(len(lv) - 1)):
        raise ValueError("levels must be strictly decreasing")
    if lv[-1] != 0:
        raise ValueError("levels must end at 0")
    events = []
    for k in range(len(lv) - 1):
        t, s = lv[k], lv[k + 1]
        n = attempts if s > 0 else 1
        for r in range(n):
            events.append({"t": t, "s": s, "is_retry": r > 0})
    return events


def _validate_schedule(betas, abar, levels):
    betas = np.asarray(betas)
    abar = np.asarray(abar)
    T = betas.shape[0]
    if abar.shape[0] != T + 1:
        raise ValueError(f"abar length {abar.shape[0]} != T+1 ({T + 1})")
    if not np.all(np.isfinite(betas)) or not np.all(np.isfinite(abar)):
        raise ValueError("non-finite schedule")
    if not (0.0 < float(abar[0]) <= 1.0) or float(abar[0]) != 1.0:
        raise ValueError("abar[0] must be exactly 1")
    if np.any(abar <= 0) or np.any(abar > 1.0):
        raise ValueError("abar values must lie in (0, 1]")
    lv = [int(v) for v in levels]
    if lv[0] != T:
        raise ValueError(f"levels must start at T={T}")
    if lv[-1] != 0 or any(lv[i + 1] >= lv[i] for i in range(len(lv) - 1)):
        raise ValueError("levels must end at 0, strictly decreasing")
    if any(v < 0 or v > T for v in lv):
        raise ValueError("levels out of range")
    return T


def make_hole_mask(shape=(4, 3, 32, 32)):
    """1 for observed, 0 inside the central 20x20 square (rows/cols 6:26,
    all channels)."""
    m = np.ones(shape, dtype=np.float32)
    m[..., 6:26, 6:26] = 0.0
    return jnp.asarray(m)


def _repaint_core(w, obs, mask, x, key, ev_t, ev_s, ev_r, abar, predict):
    """One compiled scan over all events; every array is a dynamic arg."""
    a_all = abar[ev_t] / abar[ev_s]
    t_idx = ev_t - 1                                   # model index
    btf = jnp.sqrt(1.0 - abar[ev_t])
    bsf = jnp.sqrt(abar[ev_s])
    omf = (1.0 - a_all) / (1.0 - abar[ev_t])
    psf = jnp.sqrt((1.0 - abar[ev_s]) * (1.0 - a_all)
                   / (1.0 - abar[ev_t]))

    def body(carry, inputs):
        x, k = carry
        i, t, s, is_retry, a, mi, btf_, bsf_, omf_, psf_ = inputs
        k_f, k_xi, k_eta, k_next = jax.random.split(k, 4)
        f = jax.random.normal(k_f, x.shape, dtype=x.dtype)  # used on retry
        x = jnp.where(is_retry, jnp.sqrt(a) * x + jnp.sqrt(1.0 - a) * f, x)
        eps = predict(w, x, mi)
        x0hat = jnp.clip((x - btf_ * eps) / jnp.sqrt(abar[t]), -1.0, 1.0)
        mean = bsf_ * omf_ * x0hat \
            + jnp.sqrt(a) * (1.0 - abar[s]) / (1.0 - abar[t]) * x
        xi = jax.random.normal(k_xi, x.shape, dtype=x.dtype)
        eta = jax.random.normal(k_eta, x.shape, dtype=x.dtype)
        proposal = mean + psf_ * xi
        known = bsf_ * obs + jnp.sqrt(1.0 - abar[s]) * eta
        x_new = mask * known + (1.0 - mask) * proposal
        return (x_new, k_next), x_new

    return jax.lax.scan(
        jax.jit(body), (x, key),
        (jnp.arange(ev_t.shape[0]), ev_t, ev_s, ev_r, a_all, t_idx,
         btf, bsf, omf, psf))


def _get_core(predict, obs_shape):
    """Cached compiled core: same predictor function and shape -> the same
    compiled executable reused across all seeds/images."""
    key = (id(predict), obs_shape)
    if key not in _CORE_CACHE:
        def core(w, obs, mask, x, k, ev_t, ev_s, ev_r, abar):
            return _repaint_core(w, obs, mask, x, k, ev_t, ev_s, ev_r,
                                 abar, predict)
        _CORE_CACHE[key] = jax.jit(core)
    return _CORE_CACHE[key]


def repaint_sample(weights, observations, mask, betas, abar, levels,
                   attempts=2, key=None, predict=None, record_states=False,
                   noise_stream=None, xT=None):
    """Run Algorithm 6. Returns x0 (B,C,H,W); with record_states also the
    (n_events, ...) per-event merged states.

    observations: clean values at observed entries, ZERO at hidden entries
    (asserted); mask: 1 observed / 0 hidden, same shape. xT: initial noisy
    states; fresh mode (noise_stream None) draws them from an initial key
    independent of all event draws. Tests may pass noise_stream/xT with a
    recorded key for replay (the key argument is then used only for the
    recorded metadata, not for draws).
    """
    if predict is None:
        from src.cifar_unet import cifar_unet_predict as predict
    events = repaint_events(levels, attempts)
    _validate_schedule(betas, abar, levels)
    obs = jnp.asarray(observations, dtype=jnp.float32)
    m = jnp.asarray(mask, dtype=jnp.float32)
    if obs.shape != m.shape:
        raise ValueError(f"observations shape {obs.shape} != mask "
                         f"{m.shape}")
    if not bool((obs * (1.0 - m) == 0).all()):
        raise ValueError("observations must be zero at hidden entries")
    abar_j = jnp.asarray(abar, dtype=jnp.float32)
    ev_t = jnp.array([e["t"] for e in events], dtype=jnp.int32)
    ev_s = jnp.array([e["s"] for e in events], dtype=jnp.int32)
    ev_r = jnp.array([e["is_retry"] for e in events], dtype=jnp.bool_)

    if noise_stream is not None or xT is not None:
        # test/replay mode: explicit streams and initial state (no RNG)
        n = len(events)
        xi_all = jnp.asarray(noise_stream["xi"], dtype=jnp.float32) \
            .reshape((n, *obs.shape))
        eta_all = jnp.asarray(noise_stream["eta"], dtype=jnp.float32) \
            .reshape((n, *obs.shape))
        f_all = jnp.asarray(noise_stream["f"], dtype=jnp.float32) \
            .reshape((n, *obs.shape))
        if xT is None:
            raise ValueError("stream mode requires explicit xT")
        x = jnp.asarray(xT, dtype=jnp.float32)

        a_all = abar_j[ev_t] / abar_j[ev_s]
        t_idx = ev_t - 1
        btf = jnp.sqrt(1.0 - abar_j[ev_t])
        bsf = jnp.sqrt(abar_j[ev_s])
        omf = (1.0 - a_all) / (1.0 - abar_j[ev_t])
        psf = jnp.sqrt((1.0 - abar_j[ev_s]) * (1.0 - a_all)
                       / (1.0 - abar_j[ev_t]))

        def body(x, inputs):
            i, t, s, is_retry, a, mi, btf_, bsf_, omf_, psf_ = inputs
            x = jnp.where(is_retry,
                          jnp.sqrt(a) * x + jnp.sqrt(1.0 - a) * f_all[i], x)
            eps = predict(weights, x, mi)
            x0hat = jnp.clip((x - btf_ * eps) / jnp.sqrt(abar_j[t]),
                             -1.0, 1.0)
            mean = bsf_ * omf_ * x0hat \
                + jnp.sqrt(a) * (1.0 - abar_j[s]) / (1.0 - abar_j[t]) * x
            proposal = mean + psf_ * xi_all[i]
            known = bsf_ * obs + jnp.sqrt(1.0 - abar_j[s]) * eta_all[i]
            x_new = m * known + (1.0 - m) * proposal
            return x_new, x_new

        _, xs = jax.lax.scan(
            jax.jit(body), x,
            (jnp.arange(n), ev_t, ev_s, ev_r, a_all, t_idx,
             btf, bsf, omf, psf))
        x0 = xs[-1]
    else:
        if key is None:
            raise ValueError("fresh mode requires an explicit key")
        k_init, k_carry = jax.random.split(key)
        x = jax.random.normal(k_init, obs.shape, dtype=jnp.float32)
        core = _get_core(predict, obs.shape)
        (x0, _), xs = core(weights, obs, m, x, k_carry,
                           ev_t, ev_s, ev_r, abar_j)

    return (x0, xs) if record_states else x0
