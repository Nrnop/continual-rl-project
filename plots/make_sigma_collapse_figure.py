"""Figure 6.2: the exploration width sigma through training, sigma learned, three arms.

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_sigma_collapse_figure

Sigma is recovered from the logged policy entropy, as in scripts/check_sigma_collapse.py, because
only pt logs log_std directly. Left panel: physics change (results/clean, ten seeds). Right panel:
the reward-reversal benchmark (results/rewardflip). Median over seeds.
"""
import os

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from ..scripts.check_sigma_collapse import load_runs

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLOR = {"vanilla": "#eb6834", "ewc": "#1baf7a", "pt": "#2a78d6"}
LABEL = {"vanilla": "vanilla", "ewc": "EWC", "pt": "PT"}
SWITCH = 614_400
PANELS = (
    ("physics change", {"vanilla": "clean/vanilla", "ewc": "clean/ewc_fixed", "pt": "clean/pt"}),
    ("reward reversal (retired benchmark)", {"vanilla": "rewardflip/vanilla", "ewc": "rewardflip/ewc",
                                             "pt": "rewardflip/pt"}),
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_thesis")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, dirs) in zip(axes, PANELS):
        for arm in ("vanilla", "ewc", "pt"):
            runs = list(load_runs(dirs[arm]))
            if not runs:
                continue
            grid = min((r[1] for r in runs), key=len)
            stack = np.vstack([np.interp(grid, r[1], r[2]) for r in runs])
            ax.plot(grid / 1e6, np.median(stack, axis=0), color=COLOR[arm], linewidth=2.0,
                    label=f"{LABEL[arm]} (n={len(runs)})")
        for b in range(1, 5):
            ax.axvline(b * SWITCH / 1e6, color=AXIS, linewidth=0.7, linestyle=(0, (4, 3)))
        ax.set_yscale("log")
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=9)
        ax.set_xlabel("million env steps", color=INK, fontsize=10)
        ax.set_title(name, color=INK, fontsize=10.5, loc="left", pad=8)
        leg = ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3,
                        bbox_to_anchor=(0.5, -0.17))
        for t in leg.get_texts():
            t.set_color(INK)
    axes[0].set_ylabel("exploration width σ (median, log)", color=INK, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig6_sigma_collapse.{ext}"), dpi=190,
                    bbox_inches="tight", facecolor=SURFACE)
    print("wrote fig6_sigma_collapse.png / .pdf")


if __name__ == "__main__":
    main()
