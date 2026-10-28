"""Corpus input preparation (SPEC Part 2 input preparation)."""
import numpy as np

EOT = "<|endoftext|>"

# Stand-in constant used by tests; the real stream always uses sp.eos_id().
SPECIAL_EOS_ID = 2


def split_stories(text: str) -> list[str]:
    """Split on <|endoftext|>, strip surrounding whitespace, drop empties."""
    return [s.strip() for s in text.split(EOT) if s.strip()]


def train_val_split(stories: list[str], frac: float = 0.9):
    """First `frac` of whole stories (file order) train, rest validation.

    n_train = floor(frac * len(stories)); 0.9 * 27630 = 24867 exactly.
    """
    n_train = int(frac * len(stories))
    return stories[:n_train], stories[n_train:]


def load_spm(path: str):
    from sentencepiece import SentencePieceProcessor

    sp = SentencePieceProcessor(model_file=path)
    return sp


def encode_stories(sp, stories: list[str]) -> list[int]:
    """Token stream with EOS before every story ('EOS between stories' with a
    leading EOS as the initial context); no BOS; no normalizer changes."""
    eos = sp.eos_id()
    stream: list[int] = []
    for s in stories:
        stream.append(eos)
        stream.extend(sp.encode(s))
    return stream


def chars_per_token(sp, stories: list[str]) -> float:
    """Unicode characters of the original story texts per ACTUAL corpus token
    (context sentinel excluded; inter-story EOS are actual tokens)."""
    stream = encode_stories(sp, stories)
    n_chars = sum(len(s) for s in stories)
    return n_chars / (len(stream) - 1)


def ngram_windows(stream: list[int], n: int):
    """Packed 9-bit-per-token CONTEXT keys (n-1 tokens, EOS-padded) and
    targets for an n-gram model.

    stream[0] is the context sentinel; targets are stream[1:] (all actual
    corpus tokens). Histories are left-padded with (n-1) EOS so the first
    targets are scored. For n = 1 the context key is 0 (null context).
    Returns (ctx_keys uint64[T], targets[T]).
    """
    tok = np.asarray(stream, dtype=np.uint64)
    eos = tok[0]  # sentinel == EOS id
    T = len(stream) - 1
    pad = np.full(n - 1, eos, dtype=np.uint64)
    padded = np.concatenate([pad, tok])  # length (n-1) + len(stream)
    # window for target t (stream index t, t = 1..T) is padded[t : t+n]
    win = np.lib.stride_tricks.sliding_window_view(padded, n)[1 : 1 + T]
    assert len(win) == T
    if n == 1:
        return np.zeros(T, dtype=np.uint64), tok[1:].copy()
    shifts = np.uint64(9) * np.arange(n - 2, -1, -1, dtype=np.uint64)
    ctx_keys = (win[:, : n - 1] << shifts).sum(axis=1, dtype=np.uint64)
    return ctx_keys, tok[1:].copy()


def kselect_boundary(n_actual: int, frac: float = 0.9) -> int:
    """Number of actual training tokens used to FIT counts when choosing k;
    the remaining last tokens form the selection set."""
    return int(frac * n_actual)


def roundtrip_exceptions(sp, stories: list[str]) -> list[int]:
    """Indices of stories where decode(encode(s)) != s."""
    return [i for i, s in enumerate(stories) if sp.decode(sp.encode(s)) != s]
