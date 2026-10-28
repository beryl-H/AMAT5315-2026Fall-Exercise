"""Part 2 formal driver: exclusive attempt directory and cache integrity.

Exclusivity: --attempt (default formal-02) maps to artifacts/part2/<attempt>/;
the directory must NOT exist and is created atomically (exist_ok=False)
BEFORE any settings write; a taken attempt name aborts. Cache integrity:
an encoded-stream cache is trusted only if its SHA256 matches the recorded
hash and its length matches the expected actual-token count + 1 sentinel.
"""
import hashlib
import json
import os

import numpy as np
import pytest

from src.run_part2 import (
    AttemptBusy,
    cache_meta_ok,
    load_or_encode,
    prepare_attempt,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_prepare_attempt_is_exclusive_and_atomic(tmp_path):
    art = tmp_path / "part2"
    d1, cfg1 = prepare_attempt(str(art), "formal-02")
    assert os.path.isdir(d1)
    with pytest.raises(AttemptBusy):
        prepare_attempt(str(art), "formal-02")  # same attempt name -> abort
    d2, _ = prepare_attempt(str(art), "formal-03")
    assert d2 != d1 and os.path.isdir(d2)


def test_cache_meta_ok_detects_tamper_and_length_mismatch(tmp_path):
    arr = np.arange(11, dtype=np.int32)  # 10 actual + sentinel
    p = tmp_path / "enc.npy"
    np.save(p, arr)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    meta = {"sha256": digest, "length": 11, "n_actual": 10}
    assert cache_meta_ok(str(p), meta, expected_n_actual=10)
    # tampered bytes -> fail
    np.save(p, np.arange(11, dtype=np.int32) + 1)
    assert not cache_meta_ok(str(p), meta, expected_n_actual=10)
    # wrong expected length -> fail
    np.save(p, arr)
    assert not cache_meta_ok(str(p), meta, expected_n_actual=11)


def test_load_or_encode_rejects_stale_cache_without_expected_length(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_encode(sp, stories):
        calls["n"] += 1
        return [2] + list(range(50))

    import src.run_part2 as rp

    monkeypatch.setattr(rp, "encode_stories", fake_encode)
    # no cache file: encode and record full integrity metadata
    s = load_or_encode(object(), ["stories"], "train", str(tmp_path),
                       expected_n_actual=50,
                       input_hashes={"corpus": "abc", "tokenizer": "def"})
    assert s[0] == 2 and len(s) == 51 and calls["n"] == 1
    meta = json.load(open(tmp_path / "encoded-train.npy.json"))
    assert meta["sha256"] == hashlib.sha256((tmp_path / "encoded-train.npy").read_bytes()).hexdigest()
    assert meta["n_actual"] == 50 and meta["input_hashes"] == {"corpus": "abc", "tokenizer": "def"}
    # tampered cache must NOT be trusted even though the file exists
    np.save(tmp_path / "encoded-train.npy", np.zeros(51, dtype=np.int32))
    s2 = load_or_encode(object(), ["stories"], "train", str(tmp_path),
                        expected_n_actual=50,
                        input_hashes={"corpus": "abc", "tokenizer": "def"})
    assert calls["n"] == 2  # re-encoded, cache rejected
    assert s2[0] == 2 and len(s2) == 51
