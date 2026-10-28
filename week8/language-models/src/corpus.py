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
    """Unicode characters of the original story texts per stream token
    (leading EOS included; negligible at corpus scale)."""
    stream = encode_stories(sp, stories)
    n_chars = sum(len(s) for s in stories)
    return n_chars / len(stream)


def roundtrip_exceptions(sp, stories: list[str]) -> list[int]:
    """Indices of stories where decode(encode(s)) != s."""
    return [i for i, s in enumerate(stories) if sp.decode(sp.encode(s)) != s]
