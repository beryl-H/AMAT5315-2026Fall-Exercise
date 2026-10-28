"""Part 4 tests: 算 glyph rendering (32x32, +1 ink / -1 background), stable
softmax store with own-normalized recall, and the ISR gate helpers."""
import json
import os

import numpy as np
import pytest

from src.run_part4 import picture_ISR_gate_eval, recall_part4, softmax_weights

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/mnt/c/Windows/Fonts/simhei.ttf"

pytestmark = pytest.mark.skipif(not os.path.exists(FONT), reason="Windows SimHei font not mounted")


def test_render_mask_properties_and_provenance(tmp_path):
    from src.part4_glyph import render_mask

    mask, prov = render_mask(FONT, out_dir=str(tmp_path))
    assert mask.shape == (32, 32)
    assert set(np.unique(mask)) <= {-1.0, 1.0}
    ink = (mask == 1).mean()
    assert 0.05 < ink < 0.6, f"ink fraction {ink} implausible (tofu or empty?)"
    # spatial coverage: ink in all four quadrants
    for sl in [(slice(0, 16), slice(0, 16)), (slice(0, 16), slice(16, 32)),
               (slice(16, 32), slice(0, 16)), (slice(16, 32), slice(16, 32))]:
        assert (mask[sl] == 1).sum() > 0, f"no ink in quadrant {sl}"
    # not a hollow tofu box: center 8x8 has ink
    assert (mask[12:20, 12:20] == 1).sum() >= 2
    assert os.path.exists(tmp_path / "glyph-preview.png")
    saved = json.load(open(tmp_path / "glyph.json"))
    assert saved["font_sha256"] == prov["font_sha256"] and len(saved["font_sha256"]) == 64
    assert saved["character"] == "算"
    assert "rasterization" in saved


def test_softmax_stable_shift_and_single_memory_recall():
    rng = np.random.default_rng(0)
    d, M = 64, 50
    K = rng.normal(size=(M, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.choice([-1.0, 1.0], size=(M, 1024))
    beta = 8.0
    w = softmax_weights(K, K[0], beta)
    assert np.isclose(w.sum(), 1.0) and w.min() >= 0
    # stable shift: adding a constant to all scores must not change weights
    w2 = np.exp(beta * (K @ K[0]) + 123.0)
    w2 = w2 / w2.sum()
    assert np.allclose(w, w2, rtol=1e-9)
    # single memory: recall reproduces the picture exactly, R = 0
    r, w_own = recall_part4(K[:1], V[:1], K[0], "softmax", beta)
    assert np.allclose(r, V[0]) and np.isclose(w_own, 1.0)


def test_recall_all_stores_shapes_and_own_weight():
    rng = np.random.default_rng(1)
    d, M = 64, 128
    K = rng.normal(size=(M, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.choice([-1.0, 1.0], size=(M, 1024))
    for store in ["outer", "relu", "relu2", "softmax"]:
        r, w_own = recall_part4(K, V, K[0], store, 8.0)
        assert r.shape == (1024,)
        assert w_own > 0
        assert np.isfinite(r).all()


def test_gate_definitions():
    med = {("outer", 1024): 4.0, ("relu", 1024): 2.83, ("relu2", 4096): 1.2,
           ("softmax", 4096): 0.057, ("softmax", 64): 0.01,
           ("softmax", 256): 0.02, ("softmax", 1024): 0.04}
    show = {("softmax", 4096): 0.06, ("softmax", 64): 0.01,
            ("softmax", 256): 0.02, ("softmax", 1024): 0.04}
    res = picture_ISR_gate_eval(med, show)
    assert all(res.values())
    med_bad = {**med, ("relu2", 4096): 0.9}
    assert not picture_ISR_gate_eval(med_bad, show)["relu2_4096_gt_1"]
    med_wide = {**med, ("softmax", 4096): 0.2}
    assert not picture_ISR_gate_eval(med_wide, show)["softmax_allM_lt_0.1"]
