"""Six mechanism figures built from per-seed logged scalars ((step, value) pairs); no new runs.

    # from the parent directory of the package; optional args: function names to run
    python -m src_continuous_control.plots.make_mechanism_figures [fig_detachment ...]

    fig6_components            the two components through a run           Section 6.2.5
    fig4_2_detachment          the permanent gets no online gradient      Section 4.3.2
    fig4_4_consolidation       consolidation does not move behavior       Section 4.4
    fig5_3_transfer_matrix     what row i, column j of the matrix means   Section 5.5.2
    fig5_4_retention           the retention family, against each other   Section 5.5.4
    fig6_2_boundary_drop       what each arm loses at a boundary          Section 6.2.6
"""
import glob
import os
import pickle

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
PT, VANILLA, EWC = "#2a78d6", "#eb6834", "#1baf7a"
ARM_COLOR = {"pt": PT, "vanilla": VANILLA, "ewc": EWC}
ARM_LABEL = {"pt": "PT", "vanilla": "vanilla", "ewc": "EWC"}

# HalfCheetah, piecewise physics change, sigma learned, ten seeds per arm (Table 6.1).
# Not results/hard, which is the sigma = 1.0 condition.
ARM_DIR = {
    "vanilla": "src_continuous_control/results/clean/vanilla",
    "pt": "src_continuous_control/results/clean/pt",
    "ewc": "src_continuous_control/results/clean/ewc_fixed",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_thesis")
SWITCH = 614_400


def arm_files(arm, suffix):
    """Per-seed files; the pattern is anchored so 'returns' does not also match ep_/eval_returns."""
    import re
    pat = re.compile(rf"{arm}_ppo_seed_\d+_{suffix}\.pkl$")
    return sorted(p for p in glob.glob(os.path.join(ARM_DIR[arm], f"{arm}_ppo_seed_*_{suffix}.pkl"))
                  if pat.search(os.path.basename(p)))


def series(arm, key):
    """Every seed's (step, value) pairs for one logged scalar."""
    out = []
    for p in arm_files(arm, "scalars"):
        d = pickle.load(open(p, "rb"))
        if key in d and len(d[key]):
            a = np.asarray(d[key], dtype=float)
            if a.ndim == 2 and a.shape[1] == 2:
                out.append(a)
    return out


def style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=9)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    print(f"  wrote {name}.png / .pdf")
    plt.close(fig)


def boundaries(ax, n=4):
    for b in range(1, n + 1):
        ax.axvline(b * SWITCH / 1e6, color=AXIS, linewidth=0.7, linestyle=(0, (4, 3)))


# Section 4.2.2: the cancellation
def fig_cancellation():
    runs = series("pt", "diag/actor_perm_trans_corr")
    if not runs:
        print("  [skip] no correlation series"); return
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for a in runs:
        ax.plot(a[:, 0] / 1e6, a[:, 1], color=PT, linewidth=0.9, alpha=0.35)
    grid = runs[0][:, 0]
    stack = np.vstack([np.interp(grid, a[:, 0], a[:, 1]) for a in runs])
    ax.plot(grid / 1e6, np.median(stack, axis=0), color=PT, linewidth=2.2,
            solid_capstyle="round")
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.axhline(-1, color=MUTED, linewidth=0.9, linestyle=(0, (1, 2)))
    ax.text(grid[-1] / 1e6, -1, " perfect cancellation", color=MUTED, fontsize=8.5, va="center")
    boundaries(ax)
    ax.set_ylim(-1.15, 1.05)
    style(ax, "million env steps", "corr(permanent, transient)")
    save(fig, "fig6_components")


# Section 4.3.2: the detachment
def fig_detachment():
    """Transient and permanent actor gradient norms, side by side on a shared linear axis.

    Linear rather than log, because the permanent's gradient is exactly zero.
    """
    perm = series("pt", "train/grad_norm_perm_actor")
    trans = series("pt", "train/grad_norm_trans_actor")
    if not perm or not trans:
        print("  [skip] no gradient-norm series"); return
    top = max(a[:, 1].max() for a in trans) * 1.12
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
    for ax, runs, colour, name in ((axes[0], trans, PT, "transient actor"),
                                   (axes[1], perm, VANILLA, "permanent actor")):
        for a in runs:
            ax.plot(a[:, 0] / 1e6, a[:, 1], color=colour, linewidth=0.9, alpha=0.5)
        boundaries(ax)
        ax.set_ylim(-0.03 * top, top)
        style(ax, "million env steps",
              "gradient norm reaching the actor" if name.startswith("trans") else "", name)
    mx = max(a[:, 1].max() for a in perm)
    print(f"     permanent actor gradient: max {mx:.3g}")
    save(fig, "fig4_2_detachment")


# Section 4.4: consolidation
def fig_consolidation():
    dv, dpi = series("pt", "consol/delta_v"), series("pt", "consol/delta_pi")
    if not dv or not dpi:
        print("  [skip] no consolidation deltas"); return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, runs, lab in ((axes[0], dv, "value"), (axes[1], dpi, "policy mean")):
        for a in runs:
            ax.plot(a[:, 0] / 1e6, np.abs(a[:, 1]), color=PT, linewidth=0.9, alpha=0.35)
        grid = runs[0][:, 0]
        stack = np.vstack([np.interp(grid, a[:, 0], np.abs(a[:, 1])) for a in runs])
        ax.plot(grid / 1e6, np.median(stack, axis=0), color=PT, linewidth=2.0)
        ax.set_yscale("log")
        boundaries(ax)
        # delta_v: relative change ||dV||/||V||. delta_pi: KL between the policy before and
        # after, sum (d mu)^2 / (2 sigma^2), which grows as sigma shrinks.
        style(ax, "million env steps",
              "relative change in value" if lab == "value" else "KL(before, after), nats")
    save(fig, "fig4_4_consolidation")


# Section 5.5.2: transfer matrix
def fig_transfer_matrix():
    ps = arm_files("pt", "transfer_matrix")
    if not ps:
        print("  [skip] no transfer matrix"); return
    mats = []
    for p in ps:
        d = pickle.load(open(p, "rb"))
        mats.append(np.asarray(d["transfer_matrix"], dtype=float))
    M = np.median(np.stack(mats), axis=0)
    n = M.shape[0]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    v = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu", vmin=-v, vmax=v)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color=INK if abs(M[i, j]) < 0.6 * v else SURFACE)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"task {j+1}" for j in range(n)], fontsize=9, color=SECONDARY)
    ax.set_yticklabels([f"after task {i+1}" for i in range(n)], fontsize=9, color=SECONDARY)
    ax.set_xlabel("evaluated on", color=INK, fontsize=10)
    ax.set_ylabel("trained up to", color=INK, fontsize=10)
    for s in ax.spines.values():
        s.set_color(AXIS)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=8)
    cb.outline.set_edgecolor(AXIS)
    save(fig, "fig5_3_transfer_matrix")


# Section 5.5.4: retention family
def fig_retention():
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    drew = False
    for key, c, lab in (("retention/mse_zero", MUTED, "predict zero everywhere"),
                        ("retention/mse_perm", EWC, "permanent component alone"),
                        ("retention/mse_full", PT, "the composed estimate")):
        runs = series("pt", key)
        if not runs:
            continue
        grid = runs[0][:, 0]
        stack = np.vstack([np.interp(grid, a[:, 0], a[:, 1]) for a in runs])
        ax.plot(grid / 1e6, np.median(stack, axis=0), color=c, linewidth=1.9, label=lab)
        drew = True
    if not drew:
        print("  [skip] no retention series"); plt.close(fig); return
    ax.set_yscale("log")
    boundaries(ax)
    style(ax, "million env steps", "retention error on earlier tasks (MSE, log)")
    # legend below the axes, clear of the curves
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=3,
                    bbox_to_anchor=(0.5, -0.16))
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, "fig5_4_retention")


# Section 6.2.6: the boundary drop
def fig_boundary_drop():
    """Post-boundary trough as a percentage change on pre-boundary return, per arm and boundary.

    Relative rather than absolute, because the arms reach very different return levels.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    width, arms = 0.26, ("vanilla", "ewc", "pt")
    drawn = False
    # Computed from the return curve (one point per 2,048-step update). The logged
    # boundary/post_trough uses a 10,240-step window, too short for slow collapses.
    PRE, POST = 15, 150          # updates: ~30k steps before, half a task after
    for k, arm in enumerate(arms):
        rows = []
        for p in arm_files(arm, "returns"):
            c = np.asarray(pickle.load(open(p, "rb")), float)
            steps, ret = c[:, 0], c[:, 1]
            r = []
            for b in range(1, 5):
                i = int(np.searchsorted(steps, b * SWITCH))
                pre = ret[max(0, i - PRE):i].mean()
                trough = ret[i:i + POST].min()
                r.append(100.0 * (trough / pre - 1.0) if pre > 0 else np.nan)
            rows.append(np.asarray(r))
        if not rows:
            continue
        m = min(len(r) for r in rows)
        vals = np.vstack([r[:m] for r in rows])
        med = np.nanmedian(vals, axis=0)
        x = np.arange(m) + (k - 1) * width
        ax.bar(x, med, width=width, color=ARM_COLOR[arm], label=ARM_LABEL[arm],
               edgecolor=SURFACE, linewidth=0.8)
        for xi, col in zip(x, vals.T):
            ax.scatter(np.full(len(col), xi), col, s=11, facecolor=SURFACE,
                       edgecolor=MUTED, linewidth=0.7, zorder=3)
        print(f"     {arm:<8} per boundary: {np.round(med, 1)}  median {np.nanmedian(vals):+.1f}%")
        drawn = True
    if not drawn:
        print("  [skip] no boundary series"); plt.close(fig); return
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"boundary {b+1}" for b in range(4)])
    ax.axhline(0, color=INK, linewidth=1.0)
    style(ax, "", "trough as a change on pre-boundary return (%)")
    leg = ax.legend(frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, "fig6_2_boundary_drop")


def main():
    os.makedirs(OUT, exist_ok=True)
    import sys
    only = set(sys.argv[1:])
    for fn in (fig_cancellation, fig_detachment, fig_consolidation,
               fig_transfer_matrix, fig_retention, fig_boundary_drop):
        if only and fn.__name__ not in only:
            continue
        print(f"- {fn.__name__}")
        fn()


if __name__ == "__main__":
    main()
