"""Part 6: Chinchilla scaling-law refit L = E + A/N^alpha + B/D^beta.

Frozen literal specification (SPEC): parameters theta = (logE, logA, logB,
alpha, beta); residual r_i = ln L_pred - ln L_obs; objective
F(theta) = mean(huber(r, delta)) / delta^2 with delta = 1e-3 (the constant
scaling avoids false early convergence at L-BFGS defaults and does not change
the minimizer); log L_pred computed via logsumexp (overflow-safe); L-BFGS-B
with analytic gradient, ftol 1e-15, gtol 1e-12, maxiter 20000, maxls 100;
bounds logE in [ln 1, ln 3], logA in [ln 1, ln 1e5], logB in [ln 1, ln 1e6],
alpha/beta in [0.05, 1.5]; multistart grid E x A x B x alpha x beta =
{1.5,1.7,1.9} x {10,100,1000} x {100,1000,10000} x {0.25,0.35,0.45} x
{0.25,0.35,0.45} (243 starts, best kept). Bootstrap: 400 resamples (240 runs
with replacement), seeds 0..399, each refit started from the best full fit;
percentile intervals for derived quantities. Optimizer statuses and projected
gradient norms are recorded for every attempt; a failed optimizer exit is
never reported as a scientific pass.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

DELTA = 1e-3
GRID = {"E": [1.5, 1.7, 1.9], "A": [10, 100, 1000], "B": [100, 1000, 10000],
        "alpha": [0.25, 0.35, 0.45], "beta": [0.25, 0.35, 0.45]}
BOUNDS = {"logE": (np.log(1.0), np.log(3.0)), "logA": (np.log(1.0), np.log(1e5)),
          "logB": (np.log(1.0), np.log(1e6)), "alpha": (0.05, 1.5),
          "beta": (0.05, 1.5)}
FIT_OPTIONS = {"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20000, "maxls": 100}
N_BOOTSTRAP = 400

BOUNCE_LIST = [BOUNDS["logE"], BOUNDS["logA"], BOUNDS["logB"],
               BOUNDS["alpha"], BOUNDS["beta"]]


def load_runs(csv_path):
    """Return (N, C, L) arrays for all 245 published runs (no filtering)."""
    N, C, L = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            N.append(float(row["Model Size"]))
            C.append(float(row["Training FLOP"]))
            L.append(float(row["loss"]))
    return np.array(N), np.array(C), np.array(L)


def filter_runs(N, C, L):
    """D = C/(6N); drop the five runs of highest loss -> 240 fitted runs."""
    D = C / (6.0 * N)
    keep = np.argsort(L)[:-5]
    return N[keep], D[keep], L[keep]


def log_L_pred(theta, N, D):
    logE, logA, logB, alpha, beta = theta
    terms = np.stack([np.full_like(N, logE),
                      logA - alpha * np.log(N),
                      logB - beta * np.log(D)])
    return logsumexp(terms, axis=0)


def objective_grad(theta, N, D, logL_obs, delta=DELTA):
    """F = mean(huber(r, delta))/delta^2 and its analytic gradient."""
    lp = log_L_pred(theta, N, D)
    r = lp - logL_obs
    a = np.abs(r)
    quad = a <= delta
    hub = np.where(quad, 0.5 * r ** 2, delta * a - 0.5 * delta ** 2)
    F = hub.mean() / delta ** 2
    H = np.where(quad, r, delta * np.sign(r))  # d huber/dr
    logE, logA, logB, alpha, beta = theta
    E, A, B = np.exp(logE), np.exp(logA), np.exp(logB)
    t1 = np.full_like(N, E)
    t2 = A * N ** -alpha
    t3 = B * D ** -beta
    Lp = t1 + t2 + t3
    dlog = np.vstack([t1 / Lp, t2 / Lp, t3 / Lp,
                      -(t2 * np.log(N)) / Lp, -(t3 * np.log(D)) / Lp])
    grad = (H @ dlog.T) / (len(r) * delta ** 2)
    return F, grad


def grid_starts(keys=("E", "A", "B", "alpha", "beta")):
    import itertools
    for vals in itertools.product(*(GRID[k] for k in keys)):
        E, A, B, alpha, beta = vals
        yield np.array([np.log(E), np.log(A), np.log(B), alpha, beta])


def fit_once(N, D, logL, theta0, bounds=BOUNCE_LIST, options=FIT_OPTIONS):
    return minimize(objective_grad, theta0, args=(N, D, logL), jac=True,
                    method="L-BFGS-B", bounds=bounds, options=options)


def fit_multistart(N, D, logL, starts, log=print):
    best, attempts = None, []
    for i, th0 in enumerate(starts):
        res = fit_once(N, D, logL, th0)
        attempts.append({"start": th0.tolist(), "fun": float(res.fun),
                         "success": bool(res.success), "status": int(res.status),
                         "message": str(res.message), "nit": int(res.nit),
                         "grad_proj": float(np.linalg.norm(res.jac))})
        if best is None or (res.fun < best.fun):
            best = res
    log(f"multistart: {len(attempts)} attempts, best F={best.fun:.6e}, "
        f"success={best.success}, |grad|={np.linalg.norm(best.jac):.3e}")
    return best, attempts


def derive_optimal(alpha, A, beta, B, C):
    """Analytic compute-optimal split: G = (alpha A / (beta B))^(1/(alpha+beta)),
    N* = G (C/6)^(beta/(alpha+beta)), D* = (C/6)^(alpha/(alpha+beta)) / G."""
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    Ns = G * (C / 6.0) ** (beta / (alpha + beta))
    Ds = (C / 6.0) ** (alpha / (alpha + beta)) / G
    return Ns, Ds


def theta_to_params(theta):
    logE, logA, logB, alpha, beta = theta
    return {"E": float(np.exp(logE)), "A": float(np.exp(logA)),
            "B": float(np.exp(logB)), "alpha": float(alpha), "beta": float(beta)}


def mean_log_residual(theta, N, D, logL):
    return float((log_L_pred(theta, N, D) - logL).mean())


def bootstrap(N, D, logL, theta_best, seeds=range(N_BOOTSTRAP), log=print):
    """400 refits on resamples; failure traces preserved, retry allowed only
    on the SAME resample with a justified numeric repair (no new seeds)."""
    out = []
    rng_master = np.random.default_rng(0)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(N), size=len(N))
        res = fit_once(N[idx], D[idx], logL[idx], theta_best)
        tries = 0
        while not res.success and tries < 3:  # same resample, numeric repair
            tries += 1
            res = fit_once(N[idx], D[idx], logL[idx], res.x,
                           options={**FIT_OPTIONS, "maxiter": FIT_OPTIONS["maxiter"]})
        out.append({"seed": seed, "params": theta_to_params(res.x),
                    "success": bool(res.success), "tries": tries,
                    "fun": float(res.fun), "grad_proj": float(np.linalg.norm(res.jac))})
        if (seed + 1) % 50 == 0:
            log(f"bootstrap {seed + 1}/{N_BOOTSTRAP}")
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--attempt", default=None)
    args = ap.parse_args(argv)

    N, C, L = load_runs(os.path.join(HERE, "data", "chinchilla-runs.csv"))
    Nf, Df, Lf = filter_runs(N, C, L)
    logL = np.log(Lf)
    assert len(Nf) == 240

    if args.smoke:
        art = os.path.join(HERE, "artifacts", "part6", "smoke-01")
        os.makedirs(art, exist_ok=True)
        starts = list(grid_starts())[:3]  # SMOKE ONLY: 3 of 243 starts
        best, attempts = fit_multistart(Nf, Df, logL, starts)
        params = theta_to_params(best.x)
        printed = np.array([np.log(1.69), np.log(406.4), np.log(410.7), 0.34, 0.28])
        smoke = {
            "scope": "SMOKE ONLY - 3 of 243 grid starts, no bootstrap; not formal evidence",
            "best_params": params, "F_best": float(best.fun),
            "success": bool(best.success),
            "grad_proj": float(np.linalg.norm(best.jac)),
            "attempts": attempts,
            "mean_log_residual_own": mean_log_residual(best.x, Nf, Df, logL),
            "mean_log_residual_printed": mean_log_residual(printed, Nf, Df, logL),
            "derive_check_C5.76e23": derive_optimal(params["alpha"], params["A"],
                                                    params["beta"], params["B"], 5.76e23),
        }
        with open(os.path.join(art, "smoke.json"), "w") as f:
            json.dump(smoke, f, indent=2)
        print(json.dumps(smoke, indent=1))
        return 0 if (best.success and np.isfinite(best.fun)) else 1
    print("full 243-start fit + 400 bootstrap reserved for root dispatch; "
          "implement --formal --attempt when released")
    return 1


if __name__ == "__main__":
    sys.exit(main())
