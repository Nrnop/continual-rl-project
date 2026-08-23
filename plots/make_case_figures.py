"""Return curves for every experimental setup we have, as one small-multiple grid.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.plots.make_case_figures

Each panel is one SETUP, named by what it is rather than by when it was run:

    Flip · sigma X [· reset] [· wide PT]      reward flips sign at each boundary
    Physics · sigma X [· wide PT]             damping/friction/mass/armature change at each boundary

"wide PT" = permanent [256,256] + transient [64,64] (154k params, 13.9x vanilla). Otherwise PT is
parameter-matched to vanilla: permanent [51,51] + transient [32,32] (11k, 0.99x).

Every panel shows the same four arms in the same colours, so a missing line means that arm was
never run in that setup — which is itself information, and is listed in the report.
"""
import os
import pickle
import glob

import numpy as np

VAN, PT, EWC, FROZEN = "#eb6834", "#2a78d6", "#1baf7a", "#86b6ef"
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#14161a", "#79828d", "#e3e6e9", "#c8ced4"

P1 = ""                                          # Phase 1 trees live at the repo parent
P2 = "src_continuous_control/results"

# title, {arm: glob}
SETUPS = [
    ("Flip · σ 1.0", {
        "Vanilla PPO": "stage12_results/van/*_returns.pkl",
        "PT": "stage12_results/pt/*_returns.pkl",
        "PT, permanent frozen": "stage12_results/inert/*_returns.pkl"}),
    ("Flip · σ 0.37", {
        "Vanilla PPO": "stage14_results/van/*_returns.pkl",
        "PT": "stage14_results/pt/*_returns.pkl",
        "PT, permanent frozen": "stage14_results/inert/*_returns.pkl",
        "Online EWC": f"{P2}/ewc_flip_s037/*_returns.pkl"}),
    # Same setup as the panel above, so it reuses the same EWC runs -- only PT's width differs.
    ("Flip · σ 0.37 · wide PT", {
        "Vanilla PPO": "stage18_results/van/*_returns.pkl",
        "PT": "stage18_results/pt_hisexact/*_returns.pkl",
        "Online EWC": f"{P2}/ewc_flip_s037/*_returns.pkl"}),
    ("Flip · σ 0.37 · reset", {
        "Vanilla PPO": f"{P2}/s14reset/van/*_returns.pkl",
        "PT": f"{P2}/s14reset/pt/*_returns.pkl",
        "PT, permanent frozen": f"{P2}/s14reset/inert/*_returns.pkl",
        "Online EWC": f"{P2}/s14reset/ewc/*_returns.pkl"}),
    ("Flip · σ 0.37 · reset · TASK LABEL", {
        "Vanilla PPO": f"{P2}/taskid/van/*_returns.pkl",
        "PT": f"{P2}/taskid/pt/*_returns.pkl",
        "Online EWC": f"{P2}/taskid/ewc/*_returns.pkl"}),
    ("Flip · σ learned · reset", {
        "Vanilla PPO": f"{P2}/rewardflip/vanilla/*_returns.pkl",
        "PT": f"{P2}/rewardflip/pt/*_returns.pkl",
        "PT, permanent frozen": f"{P2}/rewardflip/pt_frozen/*_returns.pkl",
        "Online EWC": f"{P2}/rewardflip/ewc/*_returns.pkl"}),
    ("Physics · σ learned", {
        "Vanilla PPO": f"{P2}/clean/vanilla/*_returns.pkl",
        "PT": f"{P2}/clean/pt/*_returns.pkl",
        "PT, permanent frozen": f"{P2}/clean/pt_frozen/*_returns.pkl",
        "Online EWC": f"{P2}/clean/ewc_fixed/*_returns.pkl"}),
    ("Physics · σ learned · wide PT", {
        "Vanilla PPO": f"{P2}/clean/vanilla/*_returns.pkl",
        "PT": f"{P2}/clean/pt_sup/*_returns.pkl",
        "PT, permanent frozen": f"{P2}/clean/pt_sup_frozen/*_returns.pkl",
        "Online EWC": f"{P2}/clean/ewc_fixed/*_returns.pkl"}),
    # --- queued overnight; these panels stay empty until the runs land, which is itself the
    # --- clearest signal of what is still missing.
    ("Physics · σ 0.37", {
        "Vanilla PPO": f"{P2}/van_physics_s037/*_returns.pkl",
        "PT": f"{P2}/pt_physics_s037/*_returns.pkl",
        "Online EWC": f"{P2}/ewc_physics_s037/*_returns.pkl"}),
    ("Flip · σ 0.55 · reset", {
        "Vanilla PPO": f"{P2}/sigma_sweep/s055_van/*_returns.pkl",
        "PT": f"{P2}/sigma_sweep/s055_pt/*_returns.pkl",
        "Online EWC": f"{P2}/sigma_sweep/s055_ewc/*_returns.pkl"}),
    ("Flip · σ 0.20 · reset", {
        "Vanilla PPO": f"{P2}/sigma_sweep/s020_van/*_returns.pkl",
        "PT": f"{P2}/sigma_sweep/s020_pt/*_returns.pkl",
        "Online EWC": f"{P2}/sigma_sweep/s020_ewc/*_returns.pkl"}),
]
COLOUR = {"Vanilla PPO": VAN, "PT": PT, "Online EWC": EWC, "PT, permanent frozen": FROZEN}


def load(pattern):
    curves, steps = [], None
    for path in sorted(glob.glob(pattern)):
        if "ep_returns" in path or "eval_returns" in path:
            continue
        arr = np.asarray(pickle.load(open(path, "rb")), dtype=float)
        if arr.ndim == 2 and arr[-1, 0] >= 3_000_000:
            curves.append(arr[:, 1])
            if steps is None or len(arr) < len(steps):
                steps = arr[:, 0]
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    return steps[:n], np.stack([c[:n] for c in curves])


def smooth(y, w=15):
    if w < 2 or len(y) < w:
        return y
    pad = w // 2
    return np.convolve(np.concatenate([np.full(pad, y[0]), y, np.full(w - pad - 1, y[-1])]),
                       np.ones(w) / w, mode="valid")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = "src_continuous_control/plots/figures_phase2"
    os.makedirs(out_dir, exist_ok=True)
    rows = (len(SETUPS) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.9 * rows), sharex=True)
    axes = axes.ravel()
    seen = {}

    for ax, (title, arms) in zip(axes, SETUPS):
        drawn = 0
        for label, pattern in arms.items():
            steps, mat = load(pattern)
            if mat is None:
                continue
            med = smooth(np.median(mat, axis=0))
            lo, hi = smooth(np.percentile(mat, 25, axis=0)), smooth(np.percentile(mat, 75, axis=0))
            line, = ax.plot(steps, med, color=COLOUR[label], lw=1.8)
            ax.fill_between(steps, lo, hi, color=COLOUR[label], alpha=0.15, lw=0)
            seen.setdefault(label, line)
            drawn += 1
        missing = [a for a in ("Vanilla PPO", "PT", "Online EWC") if a not in arms or
                   load(arms.get(a, "___none___"))[1] is None]
        for bx in range(614400, 3_072_000, 614400):
            ax.axvline(bx, color=AXIS, ls=(0, (4, 3)), lw=0.8)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        sub = f"   (no EWC run)" if "Online EWC" in missing else ""
        ax.set_title(title + sub, color=INK, fontsize=10.5, loc="left", pad=6)

    for ax in axes[len(SETUPS):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(SETUPS) - 2):len(SETUPS)]:
        ax.set_xlabel("Environment steps", color=INK, fontsize=9)
    for i in range(0, len(SETUPS), 2):
        axes[i].set_ylabel("Return (median, IQR)", color=INK, fontsize=9)

    order = ["Vanilla PPO", "PT", "Online EWC", "PT, permanent frozen"]
    handles = [seen[k] for k in order if k in seen]
    labels = [k for k in order if k in seen]
    leg = fig.legend(handles, labels, frameon=False, fontsize=10, ncol=4,
                     loc="lower center", bbox_to_anchor=(0.5, -0.015))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"all_setups.{ext}")
        fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] {out_dir}/all_setups.png  ({len(SETUPS)} setups)")
    plt.close(fig)


if __name__ == "__main__":
    main()
