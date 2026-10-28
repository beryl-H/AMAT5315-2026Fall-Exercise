"""Re-render the Part 1 relation figure with equal x/y aspect from saved
numbers (visual correction: with unequal scaling the drawn angle is not
arccos(cosine); the data are unchanged, only the geometry)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAIR_ORDER = {
    "country_capital": ["iran->tehran", "turkey->ankara", "russia->moscow", "china->beijing",
                        "france->paris", "sweden->stockholm", "japan->tokyo", "peru->lima", "kenya->nairobi"],
    "male_female": ["king->queen", "man->woman", "he->she", "brother->sister", "father->mother",
                    "son->daughter", "uncle->aunt", "boy->girl", "husband->wife"],
    "adj_comparative": ["big->bigger", "small->smaller", "fast->faster", "dark->darker", "cold->colder",
                        "old->older", "strong->stronger", "long->longer", "tall->taller"],
}


def render(results: dict, out_png: str):
    """Draw table arrows above the axis, control below, angle = arccos(cos),
    aspect equal so the drawn angle equals the true angle. Returns the figure."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 7.5))
    legend_lines = []
    for ax, (name, rel) in zip(axes, results["relations"].items()):
        cos_t = [rel["table_pair_cosines"][k] for k in PAIR_ORDER[name]]
        cos_c = rel.get("control_pair_cosines")
        for i, c in enumerate(cos_t):
            ang = np.arccos(np.clip(c, -1.0, 1.0))
            x, y = np.cos(ang), np.sin(ang)
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))
            ax.annotate(str(i + 1), xy=(x * 1.07, y * 1.07), color="tab:blue", fontsize=8,
                        ha="center", va="center")
            legend_lines.append(f"{name}[{i + 1}] = {PAIR_ORDER[name][i]}")
        if cos_c:
            for c in cos_c.values():
                ang = np.arccos(np.clip(c, -1.0, 1.0))
                ax.annotate("", xy=(np.cos(ang), -np.sin(ang)), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="0.55", lw=1.0))
        ax.annotate("", xy=(1.0, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="k", lw=2.5))
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_xlim(-0.08, 1.12)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")  # geometry fix: drawn angle == arccos(cos)
        ax.set_yticks([-1, 0, 1])
        ax.set_title(f"{name}\n table {rel['table_mean_cosine']:.2f} | "
                     f"control {rel['control_seed1730_mean_cosine']:+.2f}")
        ax.set_xlabel("cosine with mean of other 8 offsets")
    fig.suptitle("potion-base-8M relations as directions — equal-aspect geometry "
                 "(upper blue: table, numbered; lower grey: shuffled control)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig._pair_legend = legend_lines
    return fig


def main():
    results = json.load(open(os.path.join(HERE, "evidence", "part1-results.json")))
    out = os.path.join(HERE, "evidence", "part1-relations-equal-aspect.png")
    render(results, out)
    print(f"rendered {out} (data unchanged: {os.path.join(HERE, 'evidence', 'part1-results.json')})")


if __name__ == "__main__":
    main()
