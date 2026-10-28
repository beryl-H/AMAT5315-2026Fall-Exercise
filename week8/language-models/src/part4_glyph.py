"""Part 4: rendering the 算 glyph (32x32, +1 ink / -1 background) with full
provenance. The copyrighted font file is never committed - only the derived
bitmap, the font path and its SHA256 are recorded."""
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHARACTER = "算"
FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simsun.ttc",
]


def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no CJK font available for the Part 4 glyph")


def render_mask(font_path: str, size: int = 32, hi: int = 256, out_dir: str = None):
    """Rasterize the glyph at hi x hi, mean-pool to size x size, threshold
    coverage > 0.5 -> +1 ink, else -1 background. Returns (mask, provenance)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties

    fp = FontProperties(fname=font_path)
    fig = plt.figure(figsize=(hi / 100, hi / 100), dpi=100)
    fig.text(0.5, 0.5, CHARACTER, fontproperties=fp, fontsize=hi * 0.78,
             ha="center", va="center")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(axis=2)
    plt.close(fig)
    # ink = dark pixels; coverage per cell
    ink_hi = 1.0 - buf / 255.0
    cells = ink_hi.reshape(size, hi // size, size, hi // size).mean(axis=(1, 3))
    mask = np.where(cells > 0.5, 1.0, -1.0)

    ink_fraction = float((mask == 1).mean())
    quadrants = [mask[:16, :16], mask[:16, 16:], mask[16:, :16], mask[16:, 16:]]
    prov = {
        "character": CHARACTER,
        "unicode": "U+7B97",
        "font_path": font_path,
        "font_sha256": hashlib.sha256(open(font_path, "rb").read()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "rasterization": f"rendered at {hi}x{hi}, mean-pooled to {size}x{size} coverage, threshold >0.5 -> +1 ink",
        "ink_fraction": ink_fraction,
        "quadrant_ink_counts": [int((q == 1).sum()) for q in quadrants],
        "center_ink_count": int((mask[12:20, 12:20] == 1).sum()),
        "note": "derived bitmap only; the font file is not redistributed",
    }
    # sanity: reject tofu (hollow box) / empty renders
    assert 0.05 < ink_fraction < 0.6, f"implausible ink fraction {ink_fraction}"
    assert all(c > 0 for c in prov["quadrant_ink_counts"]), "glyph missing spatial coverage"
    assert prov["center_ink_count"] >= 2, "center empty: probable tofu box"

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "glyph.json"), "w") as f:
            json.dump(prov | {"mask": mask.astype(int).tolist()}, f, indent=2)
        fig2, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(mask, cmap="gray_r")
        ax.set_title(f"{CHARACTER} 32x32 (+1 ink / -1 bg)")
        ax.axis("off")
        fig2.savefig(os.path.join(out_dir, "glyph-preview.png"), dpi=120)
        plt.close(fig2)
    return mask, prov


if __name__ == "__main__":
    font = find_font()
    mask, prov = render_mask(font, out_dir=os.path.join(HERE, "evidence"))
    print(json.dumps(prov, indent=2))
