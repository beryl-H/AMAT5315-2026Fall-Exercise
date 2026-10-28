"""Part 3 memory sweep driver tests (smoke scope only — no full sweep here).

Frozen protocol: keys/values are unit vectors drawn from one seeded RNG as
largest-65536 arrays per d (seeds 101/102/103), smaller M are prefixes; draws
independent; EVERY stored key queries every store (query count == M); R is
own-normalized per corrected SPEC; per-draw aggregate = median over keys;
final curve = median over the 3 draws; capacity = log-log crossing of 1.
"""
import numpy as np
import pytest

from src.memory_sweep import (
    STOKES,
    aggregate_R,
    capacity_loglog,
    generate_unit,
    sweep_case,
)


def test_generate_unit_prefix_property_and_independence():
    K_big, V_big = generate_unit(65536, 8, 101)
    K_small, V_small = generate_unit(32, 8, 101)
    assert np.allclose(K_big[:32], K_small)  # prefix property
    assert np.allclose(V_big[:32], V_small)
    assert not np.allclose(K_big[:32], V_big[:32])  # keys/values independent
    assert np.allclose(np.linalg.norm(K_big, axis=1), 1.0)
    K2, _ = generate_unit(32, 8, 102)
    assert not np.allclose(K_small, K2)  # draws differ by seed


def test_sweep_case_returns_all_stores_with_query_count_M():
    rng = np.random.default_rng(0)
    d, M = 8, 20
    K = rng.normal(size=(M, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.normal(size=(M, d))
    out = sweep_case(K, V, tile=7)  # small tile forces chunking
    assert set(out) == set(STOKES)
    for s, r in out.items():
        assert len(r) == M  # every stored key queried


def test_chunked_matches_dense_exact_reference():
    rng = np.random.default_rng(1)
    d, M = 8, 20
    K = rng.normal(size=(M, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.normal(size=(M, d))
    out = sweep_case(K, V, tile=6)
    S = K @ K.T  # dense exact, allowed in the test only
    # outer / ReLU: own coeff 1
    for name, W in [("outer", S), ("relu1", np.maximum(S, 0) ** 1),
                    ("relu2", np.maximum(S, 0) ** 2), ("relu3", np.maximum(S, 0) ** 3)]:
        F = W @ V
        R = np.linalg.norm(F - V, axis=1) / np.linalg.norm(V, axis=1)
        assert np.allclose(out[name], R, atol=1e-10), name
    # LS via explicit H
    H = K @ np.linalg.pinv(K)
    F = H @ V
    h = np.diag(H)
    R = np.linalg.norm(F - h[:, None] * V, axis=1) / (np.abs(h) * np.linalg.norm(V, axis=1))
    assert np.allclose(out["ls"], R, atol=1e-8)


def test_orthonormal_keys_zero_interference():
    rng = np.random.default_rng(2)
    d, M = 16, 8
    Q, _ = np.linalg.qr(rng.normal(size=(d, M)))
    K = Q.T  # orthonormal unit keys
    V = rng.normal(size=(M, d))
    out = sweep_case(K, V, tile=3)
    for s in STOKES:
        assert np.max(np.abs(out[s])) < 1e-10, s


def test_aggregate_and_capacity():
    Ms = [8, 16, 32]
    curves = {s: {M: np.array([0.5 * M / 64, 1.0, 2.0]) for M in Ms} for s in ["outer"]}
    agg = aggregate_R(curves["outer"][8])
    assert agg["median"] == pytest.approx(np.median([0.5 * 8 / 64, 1.0, 2.0]))
    assert "rms" in agg and "q25" in agg and "q75" in agg
    med_curve = {M: float(np.median(curves["outer"][M])) for M in Ms}
    # R = M/64 exactly: crossing at M = 64, log-log interpolation
    cap = capacity_loglog(list(med_curve), med_curve)
    assert cap == pytest.approx(64.0, rel=1e-6)


def test_capacity_not_reached_reported_as_bound():
    cap = capacity_loglog([8, 16, 32], {8: 0.1, 16: 0.2, 32: 0.4})
    assert cap is None  # never misclassified
