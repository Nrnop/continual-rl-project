"""Return curves for the boundary-free drift study (Phase 2b).

    cd "e:/update-single task + videos"
    python -m src_continuous_control.plots.plot_drift

Four panels: two environments x two drift conditions. No boundary lines to draw -- there are no
boundaries in this benchmark, which is the point of it.

THREE ARMS PLOTTED, FOUR RUN. The second EWC lambda is reported in the table in DRIFT_RESULTS.md
but not drawn: a fourth line would have to be a second step of the EWC hue, and the palette
validator rejects that pair (normal-vision dE 12.2, below the 15 floor) -- two lines a reader
cannot reliably tell apart are worse than a table.

Rows do not share a y-axis: cartpole returns are bounded by 1000 and HalfCheetah's are not, so a
shared scale would flatten one of them.
"""
import glob
import os
import pickle

import numpy as np

from .make_phase2_figures import (EWC_AQUA, INK, PT_BLUE, SURFACE, VANILLA_ORANGE, _style)

R = "src_continuous_control/results"
ARMS = [("pt", "PT", PT_BLUE), ("vanilla", "Vanilla PPO", VANILLA_ORANGE),
        ("ewc", "Online EWC", EWC_AQUA)]
CELLS = [("cartpole", "slow"), ("cartpole", "dual"),
         ("halfcheetah", "slow"), ("halfcheetah", "dual")]
TITLE = {("cartpole", "slow"): "cartpole - slow drift only",
         ("cartpole", "dual"): "cartpole - slow + fast drift",
         ("halfcheetah", "slow"): "HalfCheetah - slow drift only",
         ("halfcheetah", "dual"): "HalfCheetah - slow + fast drift"}


def _curves(pattern, smooth=15):
    """Per-seed return curves plus the shared step axis.

    `*_returns.pkl` also matches `*_ep_returns.pkl` and `*_eval_returns.pkl`, which are flat lists
    rather than (step, value) arrays. The step axis is taken from a file that survived the filter,
    not from the first alphabetical match, which is otherwise `_ep_returns` and is 1-D.
    """
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
        # Normalise by the number of points actually inside the window. A plain
        # convolve(..., mode="same") divides by the full window everywhere, so both ends get
        # pulled toward zero and the last point plunges — which reads as a collapse that never
        # happened.
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

    for ax, (env, cond) in zip(axes.ravel(), CELLS):
        n_seeds = 0
        for key, label, color in ARMS:
            mat, steps = _curves(f"{R}/drift_{cond}_{env}/{key}_ppo_seed_*_returns.pkl")
            if mat is None:
                continue
            n_seeds = mat.shape[0]
            med = np.median(mat, axis=0)
            lo, hi = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
            ax.plot(steps, med, color=color, linewidth=2.0, label=label, solid_capstyle="round")
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)
        if env == "cartpole":
            ax.axhline(1000, color="#898781", linestyle=(0, (1, 2)), linewidth=1.0)
            ax.text(0, 1000, " ceiling = 1000", color="#898781", fontsize=8, va="bottom")
        _style(ax, "Environment steps", "Episodic return", f"{TITLE[(env, cond)]}  (n={n_seeds})")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center",
                     ncol=3, bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle("Boundary-free drift: the physics change smoothly and never stop",
                 color=INK, fontsize=12, x=0.055, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out = "src_continuous_control/plots/figures_drift"
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"drift_returns.{ext}")
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        print(f"[fig] {p}")
    plt.close(fig)
    _plot_verification(plt)


def _plot_verification(plt):
    """Why the return curves oscillate — and why that is NOT a task boundary.

    The return rises and falls on a regular interval, which looks exactly like the boundary-based
    benchmarks. It is not. It is the drift itself: the physics follow a sine of period 1,228,800
    steps, so the task genuinely gets easier and harder, smoothly and forever. Plotting the logged
    multiplier directly above the return makes the correspondence visible instead of asserted.
    """
    import pickle
    steps_m = pickle.load(open(
        f"{R}/drift_slow_cartpole/pt_ppo_seed_0_scalars.pkl", "rb"))["drift/multiplier"]
    m = np.asarray(steps_m, dtype=float)
    mat, steps = _curves(f"{R}/drift_slow_cartpole/pt_ppo_seed_*_returns.pkl")

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes[0].plot(m[:, 0], m[:, 1], color="#898781", linewidth=2.0)
    _style(axes[0], "", "Physics multiplier", "What the environment is doing (logged, not assumed)")
    axes[1].plot(steps, np.median(mat, axis=0), color=PT_BLUE, linewidth=2.0)
    _style(axes[1], "Environment steps", "Episodic return", "What the agent scores (PT, median of 10)")
    fig.suptitle("The oscillation is the drift, not a task boundary",
                 color=INK, fontsize=12, x=0.055, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        p = os.path.join("src_continuous_control/plots/figures_drift", f"drift_is_smooth.{ext}")
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        print(f"[fig] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
