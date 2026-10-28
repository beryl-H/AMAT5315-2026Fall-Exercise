"""Samplers over next-token logits (pure NumPy; f64 target distributions on
stored f32 logits) and the pooled chi-squared goodness-of-fit helper with the
corrected rare-bin merge rule (self-contained; the n-grams module is frozen).

Rules frozen (SPEC): temperature divides logits by T; top-k keeps the k
largest; top-p keeps the SMALLEST set of most likely tokens with cumulative
probability >= p, INCLUDING the token that crosses the threshold; min-p keeps
tokens with p >= minp * pmax. Zero-probability tokens with observed draws are
support violations (rejected), never nan-ed away."""
import numpy as np


def temperature_dist(logits, T):
    z = np.asarray(logits, dtype=np.float64) / T
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def _kept_mask(order, keep):
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:keep]] = True
    return mask


def top_k_dist(logits, k, T):
    z = np.asarray(logits, dtype=np.float64) / T
    order = np.argsort(-z, kind="stable")
    mask = _kept_mask(order, min(k, len(z)))
    p = np.where(mask, np.exp(z - z.max()), 0.0)
    return p / p.sum()


def top_p_dist(logits, p_threshold, T):
    z = np.asarray(logits, dtype=np.float64) / T
    e = np.exp(z - z.max())
    probs = e / e.sum()
    order = np.argsort(-probs, kind="stable")
    cum = np.cumsum(probs[order])
    keep = int(np.searchsorted(cum, p_threshold, side="left")) + 1  # crossing token retained
    mask = _kept_mask(order, min(keep, len(probs)))
    p = np.where(mask, e, 0.0)
    return p / p.sum()


def top_p_dist_wrong_negative_control(logits, p_threshold, T):
    """Deliberately WRONG: drops the token that crosses the threshold."""
    z = np.asarray(logits, dtype=np.float64) / T
    e = np.exp(z - z.max())
    probs = e / e.sum()
    order = np.argsort(-probs, kind="stable")
    cum = np.cumsum(probs[order])
    keep = int(np.searchsorted(cum, p_threshold, side="left"))  # excludes crossing
    mask = _kept_mask(order, min(keep, len(probs)))
    p = np.where(mask, e, 0.0)
    return p / p.sum()


def min_p_dist(logits, min_p, T):
    probs = temperature_dist(logits, T)
    mask = probs >= min_p * probs.max()
    p = np.where(mask, probs, 0.0)
    return p / p.sum()


def sample_dist(dist, uniforms):
    """Inverse-CDF sampling from a target distribution."""
    cum = np.cumsum(dist)
    cum[-1] = 1.0
    return np.searchsorted(cum, uniforms, side="right")


def chi2_pooled(target_dist, draws, emin=5.0, return_meta=True):
    """Pooled chi-squared GOF for a single fixed context with the corrected
    rare-bin rule: bins with expected < emin are pooled; the pool is merged
    with the smallest adequate bin until ALL expected >= emin; one remaining
    group -> 0 stat / 0 df. Observed mass on zero-target-probability tokens is
    a support violation (reject, p = nan)."""
    from scipy.stats import chi2

    draws = np.asarray(draws)
    counts = np.bincount(draws, minlength=len(target_dist)).astype(np.float64)
    expected = len(draws) * np.asarray(target_dist, dtype=np.float64)
    meta = {"minimum_expected": None, "bins": 0, "df": 0, "n_draws": int(len(draws))}
    if (counts[expected == 0] > 0).any():
        meta["support_violation"] = True
        return float("inf"), 0, 0.0, meta  # EXPLICIT rejection (p=0), never nan
    meta["support_violation"] = False
    keep = expected >= emin
    groups = [(float(o), float(e)) for o, e in zip(counts[keep], expected[keep])]
    rare_o = float(counts[~keep].sum())
    rare_e = float(expected[~keep].sum())
    if (~keep).any():
        while rare_e < emin and groups:
            groups.sort(key=lambda g: g[1])
            o_g, e_g = groups.pop(0)
            rare_o += o_g
            rare_e += e_g
        if rare_e > 0 or rare_o > 0:
            groups.append((rare_o, rare_e))
    if len(groups) <= 1:
        meta["df"] = 0
        return 0.0, 0, 0.0, meta  # uninformative context: explicit non-pass, not nan
    stat = sum((o - e) ** 2 / e for o, e in groups)
    df = len(groups) - 1
    meta["bins"] = len(groups)
    meta["df"] = df
    meta["minimum_expected"] = min(e for _, e in groups)
    return float(stat), df, float(chi2.sf(stat, df)), meta


def pick_broad_context(logits_of_prefix, n_candidates=32):
    """Deterministic broad-context selection BEFORE any sampling: the maximum-
    entropy prefix among the first n_candidates. Returns (index, rule, logits).
    Never chosen by chi2 p-value."""
    entropies = []
    snaps = []
    for i in range(n_candidates):
        z = np.asarray(logits_of_prefix(i), dtype=np.float64)
        p = np.exp(z - z.max()); p /= p.sum()
        entropies.append(float(-(p * np.log(np.clip(p, 1e-300, None))).sum()))
        snaps.append(z)
    idx = int(np.argmax(entropies))
    rule = f"max next-token entropy among the first {n_candidates} held-out prefixes (fixed before sampling; never chosen by chi2 p)"
    return idx, {"rule": rule, "entropies": entropies}, snaps[idx]
