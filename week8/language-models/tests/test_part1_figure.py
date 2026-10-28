"""Part 1 figure geometry: visual angle must equal arccos(cosine), which
requires equal x/y scaling (aspect 'equal'), rendered from saved numbers."""
import json
import os

import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "evidence", "part1-results.json")


def test_render_sets_equal_aspect_and_draws_all_pairs(tmp_path):
    from src.render_part1_figure import render

    results = json.load(open(RESULTS))
    out = tmp_path / "fig.png"
    fig = render(results, str(out))
    assert out.exists()
    for ax in fig.axes:
        assert ax.get_aspect() == 1.0, "axes must use set_aspect('equal')"
        arrows = [t for t in ax.texts if getattr(t, "arrow_patch", None) is not None]
        # 9 table + 9 control + 1 mean-direction arrow per relation panel
        assert len(arrows) == 19
    assert len(fig._pair_legend) == 27  # numbered pair key covers all 27 pairs
