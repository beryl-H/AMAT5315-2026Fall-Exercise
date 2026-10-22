"""Part 5 figures: per image — truth, observation, four labeled completions
(seed + hidden MSE), mean over seeds and pixelwise variance (RGB-averaged
for display, common scale + colorbar). Uses ONLY saved run arrays.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "artifacts/part5"
EVID = BASE / "evidence"
SEEDS = [7201, 7202, 7203, 7204]

d = np.load(RAW / "inputs_used.npz")
m = np.load(RAW / "metrics.npz")
truth, obs, mask = d["truth"], d["observations"], d["mask"]
comps = np.load(RAW / "completions_4seeds.npy")     # (4 seeds, 4 imgs, ...)
mses = m["mses"]                                    # (4, 4)
mean = m["mean"]
var = m["variance"]                                 # population /4


def disp(x):
    """(x+1)/2 clipped to [0,1] for display only, nearest pixels."""
    return np.clip((np.transpose(x, (0, 2, 3, 1)) + 1.0) / 2.0, 0.0, 1.0)


# shared scale across ALL four images from the actual global variance max
VMAX = float(max(np.transpose(var[i], (1, 2, 0)).mean(-1).max()
                 for i in range(4)))
CUT = (0.0, VMAX)

for ii in range(4):
    panels = [("truth", disp(truth[ii:ii + 1])[0]),
              ("observation", disp(obs[ii:ii + 1])[0])]
    for si, seed in enumerate(SEEDS):
        panels.append((f"seed {seed}\nMSE {mses[si, ii]:.3f}",
                       disp(comps[si, ii:ii + 1])[0]))
    panels.append(("mean over seeds", disp(mean[ii:ii + 1])[0]))
    var_disp = np.transpose(var[ii], (1, 2, 0)).mean(-1)  # RGB-averaged
    fig, axes = plt.subplots(1, len(panels) + 1,
                             figsize=(2.1 * (len(panels) + 1), 2.6))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    # draw the mask boundary on the observation panel
    from matplotlib.patches import Rectangle
    axes[1].add_patch(Rectangle((5.5, 5.5), 20, 20, fill=False,
                                color="crimson", lw=1))
    im = axes[-1].imshow(var_disp, interpolation="nearest", cmap="viridis",
                         vmin=CUT[0], vmax=CUT[1])
    axes[-1].set_title(f"pixelwise variance\n(population /4, RGB-avg; "
                       f"common 0-{VMAX:.3f})", fontsize=8)
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    fig.colorbar(im, ax=axes[-1], fraction=0.046)
    fig.suptitle(f"RePaint image {ii}: truth / observation / 4 completions / "
                 f"mean / disagreement (all 4 seeds, no selection)", fontsize=9)
    fig.tight_layout()
    fig.savefig(EVID / f"part5_repaint_image{ii}.png", dpi=130)
    plt.close(fig)
print("part5 figures written (4 panels)")
