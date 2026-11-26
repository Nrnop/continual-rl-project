"""One figure per environment, with one panel per protocol it was run under.

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_per_environment_figures

The transpose of the per-protocol grids (Figures 6.11, 6.17, 6.18). Loaders and palette come from
make_setup_figures, so the two sets of figures show the same data.
"""
import os

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from . import make_setup_figures as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_per_env")
SETUPS = [("piecewise", "piecewise change at four observable boundaries"),
          ("lipschitz1", "boundary-free drift, slow drift only"),
          ("lipschitz2", "boundary-free drift, slow cycle plus fast ripple")]
CEILING = 1000


def main():
    os.makedirs(OUT, exist_ok=True)
    for env in S.ENV_ORDER:
        panels = [(name, label) for name, label in SETUPS if env in S.SOURCES[name]]
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        drew = False
        for ax, (name, label) in zip(axes, panels):
            cell = S.SOURCES[name][env]
            for arm in S.ARMS:
                grid, mat = S.curves(cell, arm)
                if grid is None:
                    continue
                med = np.median(mat, axis=0)
                lo, hi = np.percentile(mat, [25, 75], axis=0)
                ax.fill_between(grid / 1e6, lo, hi, color=S.ARM_COLOR[arm], alpha=0.16, linewidth=0)
                ax.plot(grid / 1e6, med, color=S.ARM_COLOR[arm], linewidth=1.9,
                        label=S.ARM_LABEL[arm])
                drew = True
            if name == "piecewise":
                for b in range(1, 5):
                    ax.axvline(b * S.SWITCH / 1e6, color=S.AXIS, linewidth=0.7,
                               linestyle=(0, (4, 3)))
            ax.set_ylim(0, CEILING)
            S._style(ax, "million env steps",
                     "return (ceiling 1000)" if ax is axes[0] else "", label)
        if not drew:
            plt.close(fig); print(f"  [skip] {env}: no runs"); continue
        leg = axes[0].legend(frameon=False, fontsize=9, loc="upper left")
        for t in leg.get_texts():
            t.set_color(S.INK)
        fig.patch.set_facecolor(S.SURFACE)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = os.path.join(OUT, f"{env}.{ext}")
            fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=S.SURFACE)
        print(f"  wrote {env}.png  ({len(panels)} protocol panel(s))")
        plt.close(fig)


if __name__ == "__main__":
    main()
