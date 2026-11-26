"""Figures for the six-environment dm_control piecewise study, each with its CSV.

    python -m src_continuous_control.plots.make_multienv_figures

Writes a_returns_per_environment, b_family_summary, c_carryover_vs_advantage, d_walker_pair and
e_transfer_vs_learning. Bars show medians across seeds with per-seed dots.
"""
import argparse
import os

import numpy as np

from ..scripts.report_multienv import (
    ARMS,
    CEILING,
    ENV_DIRS,
    arm_returns,
    carry_over,
    load_pickle,
)
from .make_phase2_figures import (
    AXIS,
    EWC_AQUA,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PT_BLUE,
    SURFACE,
    VANILLA_ORANGE,
    _style,
)

ARM_COLOR = {"vanilla": VANILLA_ORANGE, "ewc": EWC_AQUA, "pt": PT_BLUE}
ARM_LABEL = {"vanilla": "vanilla", "ewc": "EWC", "pt": "pt"}

# Display names for the result directory names.
PRETTY = {
    "cartpole_swingup": "cartpole-swingup",
    "reacher_easy": "reacher-easy",
    "ball_in_cup_catch": "ball_in_cup-catch",
    "walker_stand": "walker-stand",
    "walker_walk": "walker-walk",
    "cheetah_run": "cheetah-run",
}
SWITCH = 614400


def _out_dir(root):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), root)
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, out_dir, name):
    """Write the figure as .png and .pdf and return the .png path."""
    import matplotlib.pyplot as plt
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    pdf_path = os.path.join(out_dir, name + ".pdf")
    fig.savefig(pdf_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, "and", pdf_path)
    return path


def _write_csv(out_dir, name, header, rows):
    """Write the numbers behind a figure as CSV."""
    path = os.path.join(out_dir, name + ".csv")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print("wrote", path)


def _curves(root, env, arm):
    """(steps, per-seed return matrix) on a common 400-point grid; runs under 3M steps skipped."""
    import glob
    curves = []
    for p in sorted(glob.glob(os.path.join(root, env, arm, f"{arm}_ppo_seed_*_returns.pkl"))):
        if any(s in p for s in ("ep_returns", "eval_returns")):
            continue
        a = np.asarray(load_pickle(p), dtype=float)
        if a.ndim == 2 and a[-1, 0] >= 3_000_000:
            curves.append(a)
    if not curves:
        return None, None
    grid = np.linspace(0, min(c[-1, 0] for c in curves), 400)
    mat = np.vstack([np.interp(grid, c[:, 0], c[:, 1]) for c in curves])
    return grid, mat


def _peak_and_bwt(root, env, arm):
    """Per-seed peak return and BWT, truncated to the same length."""
    import glob
    peaks, bwts = [], []
    for p in sorted(glob.glob(os.path.join(root, env, arm, f"{arm}_ppo_seed_*_returns.pkl"))):
        if any(s in p for s in ("ep_returns", "eval_returns")):
            continue
        peaks.append(float(np.asarray(load_pickle(p), dtype=float)[:, 1].max()))
    for p in sorted(glob.glob(os.path.join(root, env, arm, "*_transfer_matrix.pkl"))):
        d = load_pickle(p)
        if d.get("bwt") is not None:
            bwts.append(float(d["bwt"]))
    n = min(len(peaks), len(bwts))
    return np.array(peaks[:n]), np.array(bwts[:n])


def fig_returns(root, out_dir, envs):
    """(a) Median return with IQR band through the task sequence, one panel per environment."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), facecolor=SURFACE)
    rows = []
    for ax, env in zip(axes.ravel(), envs):
        for arm in ARMS:
            grid, mat = _curves(root, env, arm)
            if grid is None:
                continue
            med = np.median(mat, axis=0)
            lo, hi = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
            ax.fill_between(grid / 1e6, lo, hi, color=ARM_COLOR[arm], alpha=0.14, linewidth=0)
            ax.plot(grid / 1e6, med, color=ARM_COLOR[arm], linewidth=2.0,
                    label=ARM_LABEL[arm], solid_capstyle="round")
            # Direct label at the right edge.
            ax.annotate(ARM_LABEL[arm], xy=(grid[-1] / 1e6, med[-1]), xytext=(4, 0),
                        textcoords="offset points", color=ARM_COLOR[arm], fontsize=8,
                        va="center", fontweight="bold")
            for g, m in zip(grid[::40], med[::40]):
                rows.append((env, arm, int(g), round(float(m), 2)))
        for b in range(1, 5):                      # observable boundaries
            ax.axvline(b * SWITCH / 1e6, color=AXIS, linewidth=0.7, linestyle=(0, (4, 3)))
        ax.set_ylim(0, CEILING)
        _style(ax, "million env steps", "return (ceiling 1000)", PRETTY[env])
        ax.margins(x=0.10)
    # No figure title; the explanation is in the thesis caption.
    fig.tight_layout()
    _write_csv(out_dir, "a_returns_per_environment",
               ["environment", "arm", "env_step", "median_return"], rows)
    return _save(fig, out_dir, "a_returns_per_environment")


def fig_family_summary(root, out_dir, envs):
    """(b) pt's percent advantage over vanilla per environment, with the family mean.

    Advantage is on median final-20% return; p is the exact permutation test.
    """
    import matplotlib.pyplot as plt
    from ..scripts.report_tables import perm_p
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    labels, advs, ps, rows = [], [], [], []
    for env in envs:
        v, p_ = arm_returns(root, env, "vanilla"), arm_returns(root, env, "pt")
        if len(v) == 0 or len(p_) == 0:
            continue
        mv, mp = float(np.median(v[:, 1])), float(np.median(p_[:, 1]))
        adv = 100.0 * (mp - mv) / abs(mv)
        pval = perm_p(p_[:, 1], v[:, 1])
        labels.append(PRETTY[env])
        advs.append(adv)
        ps.append(pval)
        rows.append((env, round(mv, 1), round(mp, 1), round(adv, 1), round(pval, 4)))
    order = np.argsort(advs)
    labels = [labels[i] for i in order]
    advs = [advs[i] for i in order]
    ps = [ps[i] for i in order]

    y = np.arange(len(labels))
    # Colour by sign: blue where pt helps, orange where it hurts.
    colors = [PT_BLUE if a >= 0 else VANILLA_ORANGE for a in advs]
    ax.barh(y, advs, color=colors, height=0.6, linewidth=0)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1.0)
    fam = float(np.mean(advs))
    ax.axvline(fam, color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 3)))
    # Anchored in axes coordinates so the label is not clipped by the frame.
    ax.annotate(f"family mean {fam:+.1f}%", xy=(fam, 0.015), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points",
                color=INK_SECONDARY, fontsize=9, va="bottom")
    for i, (a, pv) in enumerate(zip(advs, ps)):
        star = "  p=%.3f" % pv + (" *" if pv < 0.05 else "")
        ax.annotate(f"{a:+.1f}%{star}", xy=(a, i),
                    xytext=(6 if a >= 0 else -6, 0), textcoords="offset points",
                    ha="left" if a >= 0 else "right", va="center",
                    color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    _style(ax, "pt's advantage over vanilla, final-20% return (%)", "", None)
    ax.margins(x=0.22)
    fig.tight_layout()
    _write_csv(out_dir, "b_family_summary",
               ["environment", "vanilla_median", "pt_median", "pt_advantage_pct", "p_exact"], rows)
    return _save(fig, out_dir, "b_family_summary")


def fig_carryover(root, out_dir, envs):
    """(c) Carry-over (x) against pt's percent advantage over vanilla (y), one point per env."""
    import matplotlib.pyplot as plt
    from ..scripts.report_tables import perm_p
    import glob
    fig, ax = plt.subplots(figsize=(8.4, 6.0), facecolor=SURFACE)
    xs, ys, names, rows = [], [], [], []
    for env in envs:
        hits = glob.glob(os.path.join(root, env, "vanilla", "*_transfer_matrix.pkl"))
        v, p_ = arm_returns(root, env, "vanilla"), arm_returns(root, env, "pt")
        if not hits or len(v) == 0 or len(p_) == 0:
            continue
        # Carry-over is a property of the environment, so it is read from the vanilla arm.
        cs = []
        for h in hits:
            d = load_pickle(h)
            c = carry_over(d["transfer_matrix"], d["baselines"])
            if np.isfinite(c):
                cs.append(c)
        mv, mp = float(np.median(v[:, 1])), float(np.median(p_[:, 1]))
        adv = 100.0 * (mp - mv) / abs(mv)
        xs.append(float(np.median(cs)))
        ys.append(adv)
        names.append(PRETTY[env])
        rows.append((env, round(float(np.median(cs)), 3), round(adv, 1),
                     round(perm_p(p_[:, 1], v[:, 1]), 4)))

    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.scatter(xs, ys, s=110, color=PT_BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=2.0)     # 2px surface ring on overlapping marks
    # Alternate label offsets above and below where points are close in x.
    order = np.argsort(xs)
    offsets = {}
    for rank, i in enumerate(order):
        near = [j for j in order if j != i and abs(xs[j] - xs[i]) < 0.06 * (max(xs) - min(xs) + 1e-9)]
        offsets[i] = 12 if (not near or rank % 2 == 0) else -18
    for i, (x, y, n) in enumerate(zip(xs, ys, names)):
        ax.annotate(n, xy=(x, y), xytext=(0, offsets[i]), textcoords="offset points",
                    ha="center", va="bottom" if offsets[i] > 0 else "top",
                    color=INK_SECONDARY, fontsize=9)
    if len(xs) >= 3:
        r = float(np.corrcoef(xs, ys)[0, 1])
        _style(ax, "carry-over  (fraction of task-0 competence kept on the other physics)",
           "pt's advantage over vanilla (%)", None)
    ax.margins(0.18)
    fig.tight_layout()
    _write_csv(out_dir, "c_carryover_vs_advantage",
               ["environment", "carry_over", "pt_advantage_pct", "p_exact"], rows)
    return _save(fig, out_dir, "c_carryover_vs_advantage")


def fig_walker_pair(root, out_dir):
    """(d) Final-20% return of each arm on walker-stand and walker-walk (same body)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
    envs = ["walker_stand", "walker_walk"]
    width, rows = 0.24, []
    for gi, env in enumerate(envs):
        for ai, arm in enumerate(ARMS):
            vals = arm_returns(root, env, arm)
            if len(vals) == 0:
                continue
            v = vals[:, 1]
            pos = gi + (ai - 1) * width
            ax.bar(pos, np.median(v), width=width * 0.88, color=ARM_COLOR[arm], linewidth=0)
            # Per-seed dots show the bimodal walker-stand distribution the median hides.
            ax.scatter(np.full(len(v), pos) + np.linspace(-0.05, 0.05, len(v)), v,
                       s=13, color=SURFACE, edgecolor=INK_SECONDARY, linewidth=0.8, zorder=3)
            ax.annotate(f"{np.median(v):.0f}", xy=(pos, np.median(v)), xytext=(0, -14),
                        textcoords="offset points", ha="center", color=SURFACE,
                        fontsize=9, fontweight="bold")
            rows += [(env, arm, i, round(float(x), 1)) for i, x in enumerate(v)]
            if gi == 0:
                ax.bar(pos, 0, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(["walker-stand", "walker-walk"])
    ax.set_ylim(0, CEILING)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    _style(ax, "", "final-20% return (ceiling 1000)", None)
    fig.tight_layout()
    _write_csv(out_dir, "d_walker_pair", ["environment", "arm", "seed_index", "final_return"], rows)
    return _save(fig, out_dir, "d_walker_pair")


def fig_transfer_vs_learning(root, out_dir, envs):
    """(e) Median BWT against median peak return, per environment and arm.

    Retention metrics improve when an agent learns less, so BWT is shown against peak return.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor=SURFACE)
    allx, ally, rows = [], [], []
    for arm in ARMS:
        xs, ys = [], []
        for env in envs:
            peaks, bwts = _peak_and_bwt(root, env, arm)
            if len(peaks) == 0:
                continue
            xs.append(float(np.median(peaks)))
            ys.append(float(np.median(bwts)))
            rows.append((env, arm, round(float(np.median(peaks)), 1),
                         round(float(np.median(bwts)), 1)))
        ax.scatter(xs, ys, s=90, color=ARM_COLOR[arm], label=ARM_LABEL[arm],
                   edgecolor=SURFACE, linewidth=2.0, zorder=3)
        allx += xs
        ally += ys
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
    if len(allx) >= 3:
        r = float(np.corrcoef(allx, ally)[0, 1])
        leg = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    _style(ax, "peak return", "backward transfer", None)
    ax.margins(0.16)
    fig.tight_layout()
    _write_csv(out_dir, "e_transfer_vs_learning",
               ["environment", "arm", "median_peak_return", "median_bwt"], rows)
    return _save(fig, out_dir, "e_transfer_vs_learning")


def main():
    p = argparse.ArgumentParser(description="the dm_control family's figures")
    p.add_argument("--results-dir", default="src_continuous_control/results/multienv")
    p.add_argument("--out", default="figures_multienv")
    p.add_argument("--envs", nargs="+", default=list(ENV_DIRS))
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    out = _out_dir(args.out)
    fig_returns(args.results_dir, out, args.envs)
    fig_family_summary(args.results_dir, out, args.envs)
    fig_carryover(args.results_dir, out, args.envs)
    fig_walker_pair(args.results_dir, out)
    fig_transfer_vs_learning(args.results_dir, out, args.envs)
    print("\nAll figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
