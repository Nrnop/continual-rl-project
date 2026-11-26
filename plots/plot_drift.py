"""Return curves for the boundary-free drift study.

    # from the parent directory of the package
    python -m src_continuous_control.plots.plot_drift

2x2 panels: cartpole and HalfCheetah, slow drift and slow + fast drift. The second EWC lambda that
was run is not drawn, because a second EWC hue is too close to the first to tell apart. Rows do not
share a y-axis: cartpole return is bounded by 1000, HalfCheetah's is not.
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
    """Per-seed return curves (n_seeds, n_points) plus the shared step axis.

    The glob also matches the 1-D `*_ep_returns.pkl` and `*_eval_returns.pkl`, so those are skipped
    and the step axis is taken from a file that passed the filter.
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
        # Divide by the number of points inside the window, so the ends are not pulled to zero.
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
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out = "src_continuous_control/plots/figures_drift"
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"drift_returns.{ext}")
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        print(f"[fig] {p}")
    plt.close(fig)
    _plot_verification(plt)


def _plot_verification(plt):
    """Plot the logged physics multiplier above the return curve (cartpole, slow drift, pt).

    The return oscillates because the multiplier follows a sine of period 1,228,800 steps, not
    because of task boundaries.
    """
    import pickle
    steps_m = pickle.load(open(
        f"{R}/drift_slow_cartpole/pt_ppo_seed_0_scalars.pkl", "rb"))["drift/multiplier"]
    m = np.asarray(steps_m, dtype=float)
    mat, steps = _curves(f"{R}/drift_slow_cartpole/pt_ppo_seed_*_returns.pkl")

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes[0].plot(m[:, 0], m[:, 1], color="#898781", linewidth=2.0)
    _style(axes[0], "", "Physics multiplier", None)
    axes[1].plot(steps, np.median(mat, axis=0), color=PT_BLUE, linewidth=2.0)
    _style(axes[1], "Environment steps", "Episodic return", None)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join("src_continuous_control/plots/figures_drift", f"drift_is_smooth.{ext}")
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        print(f"[fig] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
