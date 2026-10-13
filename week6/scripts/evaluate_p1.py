"""Generate Part 1 DDPM samples and evaluate learned vs exact scores.

Run from week6/:

    uv run python scripts/evaluate_p1.py

Reads artifacts/p1/params.npz and writes:
  - artifacts/p1/samples.npz         (2000 generated samples)
  - artifacts/p1/target-samples.npz  (2000 fresh target samples)
  - artifacts/p1/score-errors.json
  - artifacts/p1/measurements.json
  - evidence/score.png
  - evidence/mixture.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK6 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEEK6))

import ddm  # noqa: E402


def draw_noised(key, sched, t, n):
    """Draw n fresh samples from the noised mixture distribution at level t."""
    key, k_x, k_eps = jax.random.split(key, 3)
    x0 = ddm.sample_mixture(k_x, n, ddm.P1_CENTERS, ddm.P1_DATA_STD)
    eps = jax.random.normal(k_eps, (n, 2), dtype=jnp.float32)
    a = sched["sqrt_abar"][t]
    b = sched["sqrt_1m_abar"][t]
    return a * x0 + b * eps


def learned_score(params, x_t, t):
    """Score estimate -eps_theta(x_t, t) / sqrt(1 - abar_t)."""
    n = x_t.shape[0]
    t_arr = jnp.full((n,), t, dtype=jnp.int32)
    eps_pred = ddm.predictor_apply(params, x_t, t_arr)
    return -eps_pred / ddm.make_schedule()["sqrt_1m_abar"][t]


def relative_l2_error(learned, exact):
    num = jnp.sqrt(jnp.sum((learned - exact) ** 2))
    den = jnp.sqrt(jnp.sum(exact ** 2))
    return float(num / den)


def load_params() -> dict:
    params = ddm.load_params(str(WEEK6 / "artifacts" / "p1" / "params.npz"),
                             ddm.p1_template_params())
    return params


def generate_samples(params, sched, key, n=2000, num_levels=ddm.P1_NUM_LEVELS):
    return ddm.ddpm_sample(params, sched, key, n_samples=n, num_levels=num_levels)


def plot_score(sched, params, t_levels, errors, noised_t500):
    xs = np.asarray(noised_t500)
    t500 = 500
    t_arr = jnp.full((xs.shape[0],), t500, dtype=jnp.int32)
    eps_pred = ddm.predictor_apply(params, jnp.asarray(xs), t_arr)
    learned = -eps_pred / sched["sqrt_1m_abar"][t500]
    exact = ddm.exact_score(
        jnp.asarray(xs), t_arr, sched, ddm.P1_CENTERS, ddm.P1_DATA_VAR
    )

    # Subsample for a readable quiver.
    n = xs.shape[0]
    idx = np.linspace(0, n - 1, 400).astype(int)
    xs_q = xs[idx]
    lq = np.asarray(learned)[idx]
    eq = np.asarray(exact)[idx]

    scale = 0.12
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, field, color, title in [
        (axes[0], lq, "tab:red", "learned score at t = 500"),
        (axes[1], eq, "tab:blue", "exact score at t = 500"),
    ]:
        ax.scatter(xs[:, 0], xs[:, 1], s=4, color="0.85", alpha=0.6)
        ax.quiver(
            xs_q[:, 0],
            xs_q[:, 1],
            field[:, 0],
            field[:, 1],
            color=color,
            alpha=0.85,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.006,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    axes[2].plot(t_levels, errors, marker="o", color="tab:green")
    axes[2].axhline(0.15, color="gray", linestyle="--", label="acceptance limit 0.15")
    axes[2].set_xlabel("noise level t")
    axes[2].set_ylabel("relative L2 error")
    axes[2].set_title("learned vs exact score error")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(WEEK6 / "evidence" / "score.png", dpi=150)
    plt.close(fig)


def plot_mixture(generated, target):
    gen = np.asarray(generated)
    tgt = np.asarray(target)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        tgt[:, 0], tgt[:, 1], s=8, color="0.8", alpha=0.5, label="target"
    )
    ax.scatter(
        gen[:, 0], gen[:, 1], s=8, color="tab:red", alpha=0.5, label="generated"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("generated vs target Gaussian mixture")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(WEEK6 / "evidence" / "mixture.png", dpi=150)
    plt.close(fig)


def main() -> int:
    out_dir = WEEK6 / "artifacts" / "p1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (WEEK6 / "evidence").mkdir(parents=True, exist_ok=True)

    sched = ddm.make_schedule()
    params = load_params()

    seed = 2026
    n_samples = 2000
    t_levels = [50, 200, 500, 900]
    n_fresh = 4096

    print("Generating 2000 DDPM samples (1000 steps)...", flush=True)
    key = jax.random.PRNGKey(seed)
    generated = generate_samples(params, sched, key, n=n_samples)
    np.savez(out_dir / "samples.npz", x=np.asarray(generated))

    key, k_target = jax.random.split(key)
    target = ddm.sample_mixture(k_target, n_samples, ddm.P1_CENTERS, ddm.P1_DATA_STD)
    np.savez(out_dir / "target-samples.npz", x=np.asarray(target))

    print("Comparing learned and exact scores...", flush=True)
    errors = []
    noised_t500 = None
    for t in t_levels:
        key, k_fresh = jax.random.split(key)
        x_t = draw_noised(k_fresh, sched, t, n_fresh)
        if t == 500:
            noised_t500 = x_t
        exact = ddm.exact_score(
            x_t, jnp.full((n_fresh,), t, dtype=jnp.int32), sched,
            ddm.P1_CENTERS, ddm.P1_DATA_VAR,
        )
        learned = -ddm.predictor_apply(
            params, x_t, jnp.full((n_fresh,), t, dtype=jnp.int32)
        ) / sched["sqrt_1m_abar"][t]
        err = relative_l2_error(learned, exact)
        errors.append(err)
        print(f"  t = {t:3d}  relative L2 error = {err:.6f}")

    mean_error = float(np.mean(errors))
    print(f"mean relative L2 error = {mean_error:.6f}")

    (out_dir / "score-errors.json").write_text(
        json.dumps(
            {"t_levels": t_levels, "relative_l2_errors": errors,
             "mean_relative_l2_error": mean_error},
            indent=2,
        )
        + "\n"
    )

    measurements = {
        "seed": seed,
        "n_samples": n_samples,
        "sampler": "ddpm",
        "sampling_steps": ddm.P1_NUM_LEVELS,
        "clip": False,
        "score_levels": t_levels,
        "score_fresh_samples_per_level": n_fresh,
        "score_relative_l2_errors": errors,
        "score_mean_relative_l2_error": mean_error,
        "score_acceptance_limit": 0.15,
        "accepted": bool(mean_error < 0.15),
    }
    (out_dir / "measurements.json").write_text(
        json.dumps(measurements, indent=2) + "\n"
    )

    print("Plotting evidence...", flush=True)
    plot_score(sched, params, t_levels, errors, noised_t500)
    plot_mixture(generated, target)

    print("Wrote:")
    print("  artifacts/p1/samples.npz")
    print("  artifacts/p1/target-samples.npz")
    print("  artifacts/p1/score-errors.json")
    print("  artifacts/p1/measurements.json")
    print("  evidence/score.png")
    print("  evidence/mixture.png")

    if not measurements["accepted"]:
        print("ACCEPTANCE CHECK FAILED: mean relative L2 error >= 0.15",
              file=sys.stderr)
        return 1
    print("ACCEPTANCE CHECK PASSED: mean relative L2 error < 0.15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())