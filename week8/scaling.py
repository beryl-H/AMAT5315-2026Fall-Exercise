"""Part 3: loss-space scaling-law fit, Birge inflation, blind prediction.

Conventions FROZEN in SPEC.md (spec-review-01 corrected) before any sweep:
  - per-width uncertainty sigma_i = |L(seed1) - L(seed2)| / 2  (standard error
    of the mean of two runs), floored at 1e-3 if zero;
  - fit L(N) = a * N^(-alpha) by direct nonlinear WLS in LOSS space,
    chi2(alpha) = sum_i ((L_i - a(alpha) N_i^-alpha)/sigma_i)^2 with
    a(alpha) = sum(L_i N_i^-alpha / sigma_i^2) / sum(N_i^-2alpha / sigma_i^2);
  - Birge ratio max(1, sqrt(chi2/dof)), dof = n_points - 2;
  - deterministic parametric bootstrap in loss space, seed 20260, 10,000
    draws: L_i* ~ Normal(L_fit(N_i), sigma_i * birge), refit, predict L(192);
  - blind prediction receipt with UTC timestamp and SHA256, and a guard that
    the predicted width exceeds every fitted width (leakage guard).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

PREDICTED_WIDTH = 192
BOOTSTRAP_SEED = 20260
N_DRAWS = 10_000
SIGMA_FLOOR = 1e-3


def nonembedding_params(width: int) -> int:
    """Attention + MLP parameters (embeddings and head excluded).

    For mlp = 4*width: per layer 12*w^2 + 13*w; total 24*w^2 + 286*w + 65;
    non-embedding = total - (65w + 128w) - (65w + 65) = 24*w^2 + 28*w.
    """
    return 24 * width**2 + 28 * width


def sem_two_runs(l1: float, l2: float) -> float:
    """Standard error of the mean of two runs: |diff| / 2."""
    return abs(l1 - l2) / 2.0


def sigma_points(losses: list[float]) -> float:
    """sigma_i for one width from its seed losses (SEM, floored)."""
    if len(losses) != 2:
        raise ValueError("expected exactly two seed losses")
    return max(sem_two_runs(losses[0], losses[1]), SIGMA_FLOOR)


def _a_profile(alpha: float, Ns, Ls, sig2) -> float:
    n = Ns ** (-alpha)
    return float((Ls * n / sig2).sum() / (n * n / sig2).sum())


def _chi2(alpha: float, Ns, Ls, sig2) -> float:
    a = _a_profile(alpha, Ns, Ls, sig2)
    r = (Ls - a * Ns ** (-alpha)) / np.sqrt(sig2)
    return float((r * r).sum())


def fit_wls(Ns, Ls, sigmas) -> dict:
    """Direct loss-space WLS; Ns/Ls/sigmas are the fitted points only."""
    Ns = np.asarray(Ns, dtype=float)
    Ls = np.asarray(Ls, dtype=float)
    sig2 = np.asarray(sigmas, dtype=float) ** 2
    res = minimize_scalar(lambda al: _chi2(al, Ns, Ls, sig2),
                          bounds=(1e-4, 1.0), method="bounded",
                          options={"xatol": 1e-8})
    alpha = float(res.x)
    a = _a_profile(alpha, Ns, Ls, sig2)
    chi2 = float(res.fun)
    dof = len(Ns) - 2
    birge = max(1.0, float(np.sqrt(chi2 / dof)))
    return {"a": a, "alpha": alpha, "chi2": chi2, "dof": dof,
            "birge": birge, "Ns": Ns, "Ls": Ls, "sigmas": np.asarray(sigmas)}


def predict_blind(fit: dict, target_N: int) -> float:
    """L(target_N) from the fit; refuses N not strictly larger than fitted Ns.

    All fits and predictions operate in NON-EMBEDDING parameter counts N;
    target_width (e.g. 192) and target_N = nonembedding_params(target_width)
    are kept explicit and never conflated.
    """
    if target_N <= fit["Ns"].max():
        raise ValueError(
            f"leakage guard: predicted N {target_N} must exceed every fitted "
            f"N (max fitted {fit['Ns'].max():.0f})")
    return float(fit["a"] * target_N ** (-fit["alpha"]))


def bootstrap(fit: dict, target_width: int = PREDICTED_WIDTH,
              target_N: int | None = None, n_draws: int = N_DRAWS,
              seed: int = BOOTSTRAP_SEED) -> dict:
    """Deterministic parametric bootstrap in loss space at target_N."""
    if target_N is None:
        target_N = nonembedding_params(target_width)
    if target_N <= fit["Ns"].max():
        raise ValueError("leakage guard: bootstrap target inside the fit set")
    rng = np.random.default_rng(seed)
    sig = fit["sigmas"] * fit["birge"]
    mu = fit["a"] * fit["Ns"] ** (-fit["alpha"])
    preds, alphas = np.empty(n_draws), np.empty(n_draws)
    for i in range(n_draws):
        Ls = rng.normal(mu, sig)
        f = fit_wls(fit["Ns"], Ls, fit["sigmas"])
        preds[i] = f["a"] * target_N ** (-f["alpha"])
        alphas[i] = f["alpha"]
    lo, hi = np.percentile(preds, [2.5, 97.5])
    alo, ahi = np.percentile(alphas, [2.5, 97.5])
    point = predict_blind(fit, target_N)
    return {"n_draws": n_draws, "seed": seed, "target_width": target_width,
            "target_N": int(target_N),
            "pred_point": point, "pred_interval": [float(lo), float(hi)],
            "alpha_point": fit["alpha"], "alpha_interval": [float(alo), float(ahi)],
            "interval_width": float(hi - lo)}


def source_hashes() -> dict:
    """SHA256 of the frozen protocol sources (review item 6)."""
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("SPEC.md", "transformer.py", "run_part2.py",
                         "scaling.py", "sweep.py")}


def save_prediction(outdir: Path, fit: dict, target_width: int = PREDICTED_WIDTH,
                    n_draws: int = N_DRAWS, seed: int = BOOTSTRAP_SEED,
                    frozen_hashes: dict | None = None,
                    fit_run_manifest_shas: list[str] | None = None) -> Path:
    """Immutable blind-prediction receipt: UTC timestamp + SHA256 of payload.

    The sheet requires the predicted model to have MORE THAN TWICE the
    non-embedding parameters of any fitted model; that is asserted here.
    Filename derives from target_width, independent of target_N.
    """
    target_N = nonembedding_params(target_width)
    if target_N <= 2 * fit["Ns"].max():
        raise ValueError(
            f"sheet requirement violated: target_N {target_N} must exceed "
            f"2*max fitted N {2 * fit['Ns'].max():.0f}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    boot = bootstrap(fit, target_width=target_width, target_N=target_N,
                     n_draws=n_draws, seed=seed)
    payload = {
        "target_width": target_width,
        "target_N": int(target_N),
        "fit": {"Ns": fit["Ns"].tolist(), "Ls": fit["Ls"].tolist(),
                "sigmas": fit["sigmas"].tolist(), "a": fit["a"],
                "alpha": fit["alpha"], "chi2": fit["chi2"], "dof": fit["dof"],
                "birge": fit["birge"]},
        "prediction": boot,
        "protocol": {"frozen_hashes": frozen_hashes or source_hashes(),
                     "fit_run_manifest_shas": fit_run_manifest_shas or []},
        "conventions": {"sigma": "SEM |L1-L2|/2, floor 1e-3",
                        "birge": "max(1, sqrt(chi2/dof))",
                        "bootstrap": f"loss-space Normal, seed {seed}, {n_draws} draws",
                        "fit": "loss-space WLS, a profiled analytically"},
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "saved BEFORE any width-192 optimizer step",
    }
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    receipt = dict(payload, sha256=sha)
    path = outdir / f"prediction{target_width}.json"
    assert not path.exists(), f"prediction receipt already exists: {path}"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


def verify_receipt(path: Path) -> tuple[bool, dict]:
    """Recompute payload SHA256 and consistency; run before either 192 run."""
    rec = json.loads(Path(path).read_text())
    payload = {k: v for k, v in rec.items() if k != "sha256"}
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    checks = {"sha_ok": sha == rec["sha256"],
              "target_N_consistent":
                  rec["target_N"] == nonembedding_params(rec["target_width"]),
              "protocol_present": bool(rec.get("protocol", {}).get("frozen_hashes"))}
    return all(checks.values()), checks


def assert_prediction_before_run(receipt_ts: str, run_start_unix: float) -> None:
    """The prediction must be older than the width-192 training start.
    True UTC parsing (Z = +00:00), no local-time assumptions."""
    from datetime import datetime, timezone
    ts = datetime.strptime(receipt_ts, "%Y-%m-%dT%H:%M:%SZ")
    ts = ts.replace(tzinfo=timezone.utc).timestamp()
    if not ts < run_start_unix:
        raise ValueError(
            f"blind protocol violated: prediction timestamp {receipt_ts} is not "
            f"older than the width-192 training start ({run_start_unix})")
