"""Return curves for the switching-rate study.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.plots.plot_switchrate

Four panels: two environments x two phase lengths, same 3.07M-step budget in each. The dashed lines
mark the task boundaries — 4 in the left column, 19 in the right — which is the only thing that
differs between the columns.

Rows do not share a y-axis: cartpole returns are bounded by 1000 and HalfCheetah's are not.
"""
import glob
import os
import pickle

import numpy as np

from .make_phase2_figures import (AXIS, EWC_AQUA, INK, PT_BLUE, SURFACE, VANILLA_ORANGE, _style)

R = "src_continuous_control/results"
ARMS = [("pt", "PT", PT_BLUE), ("vanilla", "Vanilla PPO", VANILLA_ORANGE),
        ("ewc", "Online EWC", EWC_AQUA)]
CELLS = [("cartpole", "switch5", 614400), ("cartpole", "switch20", 153600),
         ("halfcheetah", "switch5", 614400), ("halfcheetah", "switch20", 153600)]
TOTAL = 3072000


def _curves(pattern, smooth=15):
    out, steps = [], None
    for p in sorted(glob.glob(pattern)):
        if "ep_returns" in p or "eval_returns" in p:
            continue
        a = np.asarray(pickle.load(open(p, "rb")), dtype=float)
        if a.ndim == 2 and a[-1, 0] >= 3_000_000:
            out.append(a[:, 1])
            if steps is None:
                steps = a[:, 0]
    if not out:
        return None, None
    m = np.vstack(out)
    if smooth > 1:
        k = np.ones(smooth)
        norm = np.convolve(np.ones(m.shape[1]), k, mode="same")
        m = np.vstack([np.convolve(r, k, mode="same") / norm for r in m])
    return m, steps


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6))
    fig.patch.set_facecolor(SURFACE)

    for ax, (env, cell, switch) in zip(axes.ravel(), CELLS):
        n = 0
        for key, label, color in ARMS:
            mat, steps = _curves(f"{R}/{cell}_{env}/{key}_ppo_seed_*_returns.pkl")
            if mat is None:
                continue
            n = mat.shape[0]
            ax.plot(steps, np.median(mat, axis=0), color=color, linewidth=2.0, label=label,
                    solid_capstyle="round")
            ax.fill_between(steps, np.percentile(mat, 25, axis=0),
                            np.percentile(mat, 75, axis=0), color=color, alpha=0.15, linewidth=0)
        # Boundaries. Thin and faint at 19 of them, or the panel becomes a grid.
        n_b = TOTAL // switch - 1
        for b in range(1, n_b + 1):
            ax.axvline(b * switch, color=AXIS, linestyle=(0, (4, 3)),
                       linewidth=0.9 if n_b < 10 else 0.5, alpha=1.0 if n_b < 10 else 0.55)
        if env == "cartpole":
            ax.axhline(1000, color="#898781", linestyle=(0, (1, 2)), linewidth=1.0)
        phases = TOTAL // switch
        _style(ax, "Environment steps", "Episodic return",
               f"{env} - {phases} tasks of {switch:,} steps  (n={n})")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center",
                     ncol=3, bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle("Same budget, shorter tasks: less time to relearn between boundaries",
                 color=INK, fontsize=12, x=0.055, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out = "src_continuous_control/plots/figures_switchrate"
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"switchrate_returns.{ext}")
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        print(f"[fig] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
