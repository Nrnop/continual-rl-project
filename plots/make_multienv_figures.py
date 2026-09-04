"""The dm_control family's figures, and the CSV behind each one.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.plots.make_multienv_figures

Five figures, in the order MULTIENV_TASK.md section 6 asks for them:

  a_returns_per_environment  return through the task sequence, one panel per environment
  b_family_summary           pt's advantage per environment, and the family mean -- an average
                             that hides a win and a loss cancelling out is worse than no plot,
                             so the spread is drawn beside it and never replaced by it
  c_carryover_vs_advantage   THE HEADLINE: carry-over on x, pt's advantage on y, six points.
                             The plot the study exists to produce
  d_walker_pair              the study's one controlled comparison, read on its own
  e_transfer_vs_learning     backward transfer against PEAK RETURN, because every
                             retention-flavoured metric improves when an agent simply learns less

COLOR. The three arm hues are the ones `make_phase2_figures.py` already uses, so a reader moving
between the HalfCheetah, cartpole and family write-ups sees one system rather than three. They were
re-validated for this study: all three sit inside the lightness band, clear the chroma floor, and
the worst adjacent colorblind separation is dE 9.2 (deutan), above the 8 threshold. The one warning
is EWC's aqua at 2.74:1 against the surface, below 3:1 -- which is why every figure here carries
direct labels and writes its CSV. Identity is never color-alone.

MEDIANS, NOT MEANS, and per-seed dots wherever a bar is drawn, because at 10 seeds a median hides
distribution shape -- and on walker-stand the distribution is bimodal, which is the single most
interesting thing in the study and is invisible in any summary statistic.
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

# Display names. The directory names carry underscores so a folder can be traced back to a
# dm_control task; the figures should read as English.
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
    import matplotlib.pyplot as plt
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def _write_csv(out_dir, name, header, rows):
    """Every figure ships its numbers. A figure whose CSV is missing cannot be checked."""
    path = os.path.join(out_dir, name + ".csv")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print("wrote", path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _curves(root, env, arm):
    """(steps, matrix of per-seed returns) resampled onto a common grid."""
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


# ---------------------------------------------------------------------------
# (a) return through the task sequence, per environment
# ---------------------------------------------------------------------------
def fig_returns(root, out_dir, envs):
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
            # Direct label at the right edge — identity is never color-alone.
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
    fig.suptitle("Return through the task sequence — median of 10 seeds, IQR shaded; "
                 "dashed lines are observable physics boundaries",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _write_csv(out_dir, "a_returns_per_environment",
               ["environment", "arm", "env_step", "median_return"], rows)
    return _save(fig, out_dir, "a_returns_per_environment")


# ---------------------------------------------------------------------------
# (b) pt's advantage per environment, with the family mean beside it
# ---------------------------------------------------------------------------
def fig_family_summary(root, out_dir, envs):
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
    # Diverging by SIGN, which is the polarity the reader needs: did pt help or hurt.
    colors = [PT_BLUE if a >= 0 else VANILLA_ORANGE for a in advs]
    ax.barh(y, advs, color=colors, height=0.6, linewidth=0)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1.0)
    fam = float(np.mean(advs))
    ax.axvline(fam, color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 3)))
    # Anchored to the axes, not to a data row: at the top row it was clipped by the frame.
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
    _style(ax, "pt's advantage over vanilla, final-20% return (%)", "",
           "What pt buys, per environment — the mean alone would hide a win and a loss cancelling")
    ax.margins(x=0.22)
    fig.tight_layout()
    _write_csv(out_dir, "b_family_summary",
               ["environment", "vanilla_median", "pt_median", "pt_advantage_pct", "p_exact"], rows)
    return _save(fig, out_dir, "b_family_summary")


# ---------------------------------------------------------------------------
# (c) THE HEADLINE — carry-over vs pt's advantage
# ---------------------------------------------------------------------------
def fig_carryover(root, out_dir, envs):
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
        # Carry-over is a property of the ENVIRONMENT, so it is read off the baseline arm.
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
    # Nudge labels apart where points are close in x: two overlapping labels are a
    # collision the palette validator cannot catch, only looking at the render can.
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
        ax.annotate(f"Pearson r = {r:.2f}  (n = {len(xs)} — a description, not a test)",
                    xy=(0.02, 0.03), xycoords="axes fraction",
                    color=INK_MUTED, fontsize=9)
    _style(ax, "carry-over  (fraction of task-0 competence kept on the other physics)",
           "pt's advantage over vanilla (%)",
           "The plot the study exists to produce")
    ax.margins(0.18)
    fig.tight_layout()
    _write_csv(out_dir, "c_carryover_vs_advantage",
               ["environment", "carry_over", "pt_advantage_pct", "p_exact"], rows)
    return _save(fig, out_dir, "c_carryover_vs_advantage")


# ---------------------------------------------------------------------------
# (d) the walker pair — the one controlled comparison
# ---------------------------------------------------------------------------
def fig_walker_pair(root, out_dir):
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
            # Per-seed dots: walker-stand is BIMODAL and the median alone hides it entirely.
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
    _style(ax, "", "final-20% return (ceiling 1000)",
           "The same robot, the same physics change — only the GOAL differs.\n"
           "Dots are individual seeds: walker-stand is bimodal, which no median can show.")
    fig.tight_layout()
    _write_csv(out_dir, "d_walker_pair", ["environment", "arm", "seed_index", "final_return"], rows)
    return _save(fig, out_dir, "d_walker_pair")


# ---------------------------------------------------------------------------
# (e) transfer against how much the arm actually learned
# ---------------------------------------------------------------------------
def fig_transfer_vs_learning(root, out_dir, envs):
    """Constraint 4: never a retention metric without how much that arm learned beside it.

    On HalfCheetah corr(peak return, BWT) was -0.745 and a frozen arm scored BWT ~ 0 with a peak
    return of exactly 0.0. Plotting the two against each other makes that failure mode visible
    instead of leaving it to a footnote.
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
        ax.annotate(f"corr(peak, BWT) = {r:+.2f}   (HalfCheetah measured -0.75)",
                    xy=(0.02, 0.04), xycoords="axes fraction", color=INK_MUTED, fontsize=9)
    leg = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    _style(ax, "peak return — how much this arm actually learned", "backward transfer",
           "Retention beside learning: a retention metric improves when an agent learns less")
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
