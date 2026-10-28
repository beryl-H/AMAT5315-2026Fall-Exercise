"""Part 2 n-gram models: 9-bit packed keys (orders <= 6 fit in 54 bits),
sparse counts via np.unique/searchsorted, add-k smoothing, explicit chunked
joint enumeration, bigram ancestral sampler, pooled chi-squared.

Frozen conventions (SPEC): stream[0] is the context sentinel (never a target);
targets are stream[1:]; histories left-padded with (n-1) EOS; counts never
cross the train/val split; all scoring in float64.
"""
import numpy as np

V = 512
BITS = 9


def pack_window(win: np.ndarray) -> np.ndarray:
    """Pack windows of 9-bit tokens into uint64 keys (n<=6 -> <=54 bits)."""
    n = win.shape[-1]
    shifts = np.uint64(BITS) * np.arange(n - 1, -1, -1, dtype=np.uint64)
    return (win << shifts).sum(axis=-1, dtype=np.uint64)


class NgramCounts:
    """Sparse n-gram counts with add-k smoothing."""

    def __init__(self, n, k, joint_keys, joint_counts, ctx_keys, ctx_counts, V=512):
        self.n, self.k, self.V = n, k, V
        self.joint_keys, self.joint_counts = joint_keys, joint_counts.astype(np.float64)
        self.ctx_keys, self.ctx_counts = ctx_keys, ctx_counts.astype(np.float64)

    @staticmethod
    def fit(stream, n, k, V=512, fit_targets=None):
        """fit_targets: number of leading actual tokens to fit (None = all)."""
        from src.corpus import ngram_windows

        ctx, tgt = ngram_windows(stream, n)
        if fit_targets is not None:
            ctx, tgt = ctx[:fit_targets], tgt[:fit_targets]
        if n == 1:
            jk, jc = np.unique(tgt.astype(np.uint64), return_counts=True)
            return NgramCounts(n, k, jk, jc, jk, jc, V)  # ctx unused
        joints = (ctx << np.uint64(BITS)) | tgt.astype(np.uint64)
        jk, jc = np.unique(joints, return_counts=True)
        ck, cc = np.unique(ctx, return_counts=True)
        return NgramCounts(n, k, jk, jc, ck, cc, V)

    def _lookup(self, keys_sorted, counts, query):
        if len(keys_sorted) == 0:
            return np.zeros(len(query))
        idx = np.searchsorted(keys_sorted, query)
        idx = np.clip(idx, 0, len(keys_sorted) - 1)
        hit = keys_sorted[idx] == query
        return np.where(hit, counts[idx], 0.0)

    def logprob_joint(self, joint_keys, ctx_keys):
        """P(target|ctx) for packed joint/context keys, add-k."""
        if self.n == 1:
            total = self.joint_counts.sum()
            cj = self._lookup(self.joint_keys, self.joint_counts, joint_keys)
            return (cj + self.k) / (total + self.k * self.V)
        cj = self._lookup(self.joint_keys, self.joint_counts, joint_keys)
        cc = self._lookup(self.ctx_keys, self.ctx_counts, ctx_keys)
        return (cj + self.k) / (cc + self.k * self.V)


def score_nats(model: NgramCounts, stream) -> np.ndarray:
    """Per-token -ln p for every actual token stream[1:] (float64)."""
    from src.corpus import ngram_windows

    ctx, tgt = ngram_windows(stream, model.n)
    joints = (ctx << np.uint64(BITS)) | tgt.astype(np.uint64)
    p = model.logprob_joint(joints, ctx)
    return -np.log(p)


def enumerate_joint_sum(model: NgramCounts, L: int, chunk: int = 1 << 20) -> float:
    """Explicitly enumerate ALL V**L token sequences in chunks and sum the
    model's joint probability (product of per-position add-k conditionals,
    histories EOS-padded). L <= 3 for V = 512 (L*9 <= 27 bits)."""
    Vl = model.V
    eos = np.uint64(2)  # EOS id used for padding (== sentinel id)
    assert L * BITS <= 54
    n = model.n
    if n == 1:
        # unigram: joint probability of any length-L sequence is prod p(x_t)
        # with p from the unigram distribution; sum over V^L equals 1 exactly
        total = 0.0
        for start in range(0, Vl ** L, chunk):
            base = np.arange(start, min(start + chunk, Vl ** L), dtype=np.uint64)
            digits = [(base // np.uint64(Vl ** (L - 1 - i))) % np.uint64(Vl)
                      for i in range(L)]
            p = np.ones(len(base), dtype=np.float64)
            for i in range(L):
                p *= model.logprob_joint(digits[i], digits[i])
            total += float(p.sum())
        return total
    total = 0.0
    for start in range(0, Vl ** L, chunk):
        base = np.arange(start, min(start + chunk, Vl ** L), dtype=np.uint64)
        digits = [(base // np.uint64(Vl ** (L - 1 - i))) % np.uint64(Vl)
                  for i in range(L)]
        p = np.ones(len(base), dtype=np.float64)
        for t in range(1, L + 1):  # position t, target digit x_{t-1}
            # window = last n-1 context tokens (EOS-padded) + target
            toks = []
            for j in range(n - 1):
                idx = t - n + j  # 0-indexed digit position, may be < 0
                toks.append(digits[idx] if idx >= 0 else np.full(len(base), eos, dtype=np.uint64))
            toks.append(digits[t - 1])
            win = np.stack(toks, axis=-1)
            jk = pack_window(win)
            ck = jk >> np.uint64(BITS)
            p *= model.logprob_joint(jk, ck)
        total += float(p.sum())
    return total


def bigram_dense_table(model: NgramCounts) -> np.ndarray:
    """Dense 512x512 add-k bigram conditional table (float64)."""
    assert model.n == 2
    a = np.arange(V, dtype=np.uint64)
    joints = (a[:, None] << np.uint64(BITS)) | a[None, :]
    ctx = np.repeat(a[:, None], V, axis=1)
    return model.logprob_joint(joints.ravel(), ctx.ravel()).reshape(V, V)


def ancestral_sample_bigram(P: np.ndarray, n_tokens: int, seed: int) -> list[int]:
    """Ancestral sampling: first token from the EOS (sentinel) row, then walk
    the chain. Returns n_tokens actual tokens."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0  # guard fp drift
    out = []
    ctx = 2  # EOS sentinel row
    for _ in range(n_tokens):
        u = rng.random()
        nxt = int(np.searchsorted(cum[ctx], u, side="right"))
        out.append(min(nxt, V - 1))
        ctx = out[-1]
    return out


def two_lag_sample_bigram(P: np.ndarray, n_tokens: int, seed: int) -> list[int]:
    """Negative-control sampler: draws x_t from the row of x_{t-2} (the token
    two places back). x_0 and x_1 come from the EOS sentinel row. This is a
    deliberately WRONG sampler; the pooled chi-squared test must reject it."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0
    out = []
    for t in range(n_tokens):
        ctx = out[t - 2] if t >= 2 else 2  # EOS sentinel row for t < 2
        u = rng.random()
        out.append(min(int(np.searchsorted(cum[ctx], u, side="right")), V - 1))
    return out


def pooled_chi2_bigram(sampled: np.ndarray, P: np.ndarray, top_ctx, emin=5.0,
                       return_meta=False):
    """One pooled chi-squared test over the given contexts.

    Per context a with n_a occurrences among sampled transitions:
    expected E_b = n_a * P[b|a]; bins with E < emin are pooled into one rare
    bin; if the pooled bin's own expected is still < emin it is merged with
    the smallest adequate bin (repeatedly) until ALL expected >= emin. If a
    context is reduced to a single group it contributes 0 statistic and 0 df
    (uninformative; overall p is nan when no context contributes).
    statistic and df = bins-1 are summed over contexts.

    Returns (stat, df, p) or (stat, df, p, meta) with return_meta=True; meta
    records minimum expected, bins/df and per-context counts.
    """
    from scipy.stats import chi2

    sampled = np.asarray(sampled)
    trans_ctx, trans_nxt = sampled[:-1], sampled[1:]
    stat, df = 0.0, 0
    meta = {"minimum_expected": None, "bins": 0, "df": 0,
            "context_counts": {}, "contexts_used": 0}
    for a in top_ctx:
        mask = trans_ctx == a
        n_a = int(mask.sum())
        if n_a == 0:
            continue
        meta["context_counts"][int(a)] = n_a
        obs = np.bincount(trans_nxt[mask], minlength=V).astype(np.float64)
        exp = n_a * P[a]
        keep = exp >= emin
        groups = [(float(o), float(e)) for o, e in zip(obs[keep], exp[keep])]
        rare_o = float(obs[~keep].sum())
        rare_e = float(exp[~keep].sum())
        if (~keep).any():
            # merge the rare pool until its expected reaches emin
            while rare_e < emin and groups:
                groups.sort(key=lambda g: g[1])
                o_g, e_g = groups.pop(0)  # smallest adequate bin
                rare_o += o_g
                rare_e += e_g
            if rare_e > 0 or rare_o > 0:
                groups.append((rare_o, rare_e))
        if len(groups) <= 1:
            continue  # uninformative context: 0 stat, 0 df
        s_ctx = sum((o - e) ** 2 / e for o, e in groups)
        stat += float(s_ctx)
        df += len(groups) - 1
        meta["contexts_used"] += 1
        meta["bins"] += len(groups)
        meta["df"] += len(groups) - 1
        me = min(e for _, e in groups)
        meta["minimum_expected"] = me if meta["minimum_expected"] is None \
            else min(meta["minimum_expected"], me)
    if df > 0:
        p = float(chi2.sf(stat, df))
    else:
        p = float("nan")
    meta["df"] = df
    if return_meta:
        return stat, df, p, meta
    return stat, df, p
