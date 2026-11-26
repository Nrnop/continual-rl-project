"""The permanent's consolidation regression loss on cartpole and HalfCheetah, actor and critic.

    # from the parent directory of the package
    python -m src_continuous_control.plots.plot_consolidation_loss

At each consolidation the permanent regresses toward `perm + rho*trans`. "Before" is the loss at
the start of that regression (how far the target moved since the last one); "after" is the
residual the permanent could not fit. HalfCheetah runs learn sigma and cartpole runs fix it at
0.37, so the columns differ in more than the environment.
"""
import glob
import os
import pickle

import numpy as np

from .make_phase2_figures import (AXIS, GRID, INK, INK_MUTED, INK_SECONDARY, PT_BLUE, SURFACE,
                                  VANILLA_ORANGE, _style)

SWITCH = 614400
TOTAL = 3072000
EDGE = 32           # minibatches averaged at each end of a consolidation, to de-noise start/end

PANELS = [
    ("actor", "cartpole", "src_continuous_control/results/cartpole/pt_ppo_seed_*_actor_consol_loss_traces.pkl"),
    ("actor", "HalfCheetah", "src_continuous_control/results/clean/pt/*_actor_consol_loss_traces.pkl"),
    ("critic", "cartpole", "src_continuous_control/results/cartpole/pt_ppo_seed_*_consol_loss_traces.pkl"),
    ("critic", "HalfCheetah", "src_continuous_control/results/clean/pt/*_consol_loss_traces.pkl"),
]


def _series(pattern, grid):
    """Median start/end regression loss across seeds on `grid`; returns (start, end, n_seeds).

    Seeds are interpolated onto `grid` rather than stacked by index, because the extra
    consolidation at each boundary means the k-th consolidation is not at the same step in every run.
    """
    starts, ends = [], []
    for path in sorted(glob.glob(pattern)):
        # The critic glob also matches `*_actor_consol_loss_traces.pkl`; skip those.
        if "actor" not in pattern and "actor" in os.path.basename(path):
            continue
        try:
            traces = pickle.load(open(path, "rb"))
        except Exception:
            continue
        if not traces:
            continue
        steps = np.array([t[0] for t in traces], dtype=float)
        curves = [np.asarray(t[1], dtype=float) for t in traces]
        s = np.array([c[:EDGE].mean() for c in curves])
        e = np.array([c[-EDGE:].mean() for c in curves])
        order = np.argsort(steps)
        starts.append(np.interp(grid, steps[order], s[order]))
        ends.append(np.interp(grid, steps[order], e[order]))
    if not starts:
        return None, None, 0
    return (np.median(np.vstack(starts), axis=0),
            np.median(np.vstack(ends), axis=0), len(starts))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(20480, TOTAL, 300)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (role, env, pat) in zip(axes.ravel(), PANELS):
        start, end, n = _series(pat, grid)
        if start is None:
            ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center", color=INK_MUTED)
            continue
        ax.plot(grid, start, color=PT_BLUE, linewidth=2.0, label="before the regression",
                solid_capstyle="round")
        ax.plot(grid, end, color=VANILLA_ORANGE, linewidth=2.0, label="after it",
                solid_capstyle="round")
        ax.set_yscale("log")
        for b in range(SWITCH, TOTAL, SWITCH):
            ax.axvline(b, color=AXIS, linestyle=(0, (4, 3)), linewidth=0.9)
        _style(ax, "", "regression loss", f"{env} - permanent {role} (n={n})")
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)

    for ax in axes[1]:
        ax.set_xlabel("Environment steps", color=INK, fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center",
                     ncol=2, bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=[0, 0.045, 1, 1])

    out_dir = "src_continuous_control/plots/figures_cartpole"
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"d_consolidation_loss.{ext}")
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        print(f"[fig] {p}")
    plt.close(fig)

    # How much of the gap the regression closes, per panel.
    print("\n  environment / role      start -> end (median over the run)   reduction")
    for role, env, pat in PANELS:
        s, e, n = _series(pat, grid)
        if s is None:
            continue
        print(f"  {env:<12} {role:<7} {np.median(s):.3e} -> {np.median(e):.3e}   "
              f"{100 * (1 - np.median(e) / np.median(s)):.1f}%")


if __name__ == "__main__":
    main()
