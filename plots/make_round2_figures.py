"""Figures for the k/rho sweep on Lipschitz2 and the stationary control.

    python -m src_continuous_control.plots.make_round2_figures

Covers cartpole-swingup and cheetah-run, each figure with its CSV. The k sweep is compared
against EWC, and the sweep and its baselines were run on the same machine, so they are compared
only with each other. Final-20% return is `report_multienv.run_returns` (mean of the last fifth
of a run's logged returns), then the median across seeds.
"""
import argparse
import glob
import os

import numpy as np

from ..scripts.report_multienv import CEILING, load_pickle, run_returns
from ..scripts.report_tables import perm_p
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

ENVS = ("cartpole_swingup", "cheetah_run")
PRETTY = {"cartpole_swingup": "cartpole-swingup", "cheetah_run": "cheetah-run"}

KSWEEP = "src_continuous_control/results/ksweep"
STATIONARY = "src_continuous_control/results/stationary"

# The four rho = 0.5 arms in k order, plus the rho = 1.0 control. k = 10 is the default
# Lipschitz2 configuration.
K_ARMS = (("pt_k10_rho050", 10, 0.5), ("pt_k20_rho050", 20, 0.5),
          ("pt_k30_rho050", 30, 0.5), ("pt_k60_rho050", 60, 0.5))
RHO_ARM = ("pt_k20_rho100", 20, 1.0)

STATIONARY_ARMS = (("vanilla", VANILLA_ORANGE), ("ewc", EWC_AQUA), ("pt", PT_BLUE))
SWITCH = 614400


def _out_dir(name="figures_round2"):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, out_dir, name):
    """Write the figure as .png and .pdf."""
    import matplotlib.pyplot as plt
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    pdf_path = os.path.join(out_dir, name + ".pdf")
    fig.savefig(pdf_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, "and", pdf_path)


def _write_csv(out_dir, name, header, rows):
    path = os.path.join(out_dir, name + ".csv")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print("wrote", path)


def final20(cell_dir, arm):
    """Per-seed final-20% return via `report_multienv.run_returns`; runs under 3M steps skipped."""
    out = []
    for path in sorted(glob.glob(os.path.join(cell_dir, arm, "*_returns.pkl"))):
        got = run_returns(path)
        if got is not None and got[2] >= 3_000_000:
            out.append(got[1])
    return np.asarray(out, dtype=float)


def curves(cell_dir, arm, n_grid=400):
    """(steps, per-seed matrix) on a common grid."""
    out = []
    for p in sorted(glob.glob(os.path.join(cell_dir, arm, f"*_returns.pkl"))):
        if any(s in p for s in ("ep_returns", "eval_returns")):
            continue
        a = np.asarray(load_pickle(p), dtype=float)
        if a.ndim == 2 and a[-1, 0] >= 3_000_000:
            out.append(a)
    if not out:
        return None, None
    grid = np.linspace(0, min(c[-1, 0] for c in out), n_grid)
    return grid, np.vstack([np.interp(grid, c[:, 0], c[:, 1]) for c in out])


def absorbed(cell_dir, arm):
    """Median over seeds of each run's median actor_absorbed_frac."""
    import pickle
    per = []
    for p in sorted(glob.glob(os.path.join(cell_dir, arm, "*_consolidation_records.pkl"))):
        with open(p, "rb") as f:
            recs = pickle.load(f)
        vals = [r["actor_absorbed_frac"] for r in recs
                if r.get("actor_absorbed_frac") is not None]
        if vals:
            per.append(float(np.median(vals)))
    return float(np.median(per)) if per else float("nan")


def _fmt_p(p):
    return "p<.001" if p < 0.001 else ("p=%.3f" % p).replace("0.", ".", 1)


def _boot(a, b, n=4000, seed=0):
    """95% bootstrap interval of median(a) - median(b)."""
    rng = np.random.RandomState(seed)
    d = [np.median(a[rng.randint(0, len(a), len(a))]) - np.median(b[rng.randint(0, len(b), len(b))])
         for _ in range(n)]
    return tuple(float(v) for v in np.percentile(d, [2.5, 97.5]))


def fig_k_levels(out_dir):
    """Final-20% return against k, with per-seed dots and the EWC median and IQR as reference."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), facecolor=SURFACE)
    rows = []
    for ax, env in zip(axes, ENVS):
        cell = os.path.join(KSWEEP, env)
        ew = final20(cell, "ewc")
        ax.axhspan(np.percentile(ew, 25), np.percentile(ew, 75), color=EWC_AQUA, alpha=0.13,
                   linewidth=0, zorder=0)
        ax.axhline(np.median(ew), color=EWC_AQUA, linewidth=2.0, zorder=1)
        ax.annotate("EWC  %.1f" % np.median(ew), xy=(0.985, np.median(ew)),
                    xycoords=("axes fraction", "data"), xytext=(0, 6),
                    textcoords="offset points", ha="right", va="bottom",
                    color=EWC_AQUA, fontsize=9, fontweight="bold")
        rows.append((env, "ewc", 0, 0.0, round(float(np.median(ew)), 1), len(ew), "", ""))

        xs = [k for _, k, _ in K_ARMS]
        meds = []
        for arm, k, rho in K_ARMS:
            v = final20(cell, arm)
            meds.append(float(np.median(v)))
            ax.scatter(np.full(len(v), k) + np.linspace(-1.4, 1.4, len(v)), v, s=16,
                       color=PT_BLUE, alpha=0.30, linewidth=0, zorder=2)
            rows.append((env, arm, k, rho, round(float(np.median(v)), 1), len(v),
                         round(float(np.median(v) - np.median(ew)), 1),
                         round(float(perm_p(v, ew)), 4)))
        ax.plot(xs, meds, color=PT_BLUE, linewidth=2.2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3, label="pt, rho = 0.5")

        arm, k, rho = RHO_ARM
        v = final20(cell, arm)
        ax.scatter([k], [np.median(v)], s=90, facecolor=SURFACE, edgecolor=VANILLA_ORANGE,
                   linewidth=2.2, zorder=4, label="pt, rho = 1.0")
        ax.scatter(np.full(len(v), k) + np.linspace(-1.4, 1.4, len(v)), v, s=16,
                   color=VANILLA_ORANGE, alpha=0.30, linewidth=0, zorder=2)
        rows.append((env, arm, k, rho, round(float(np.median(v)), 1), len(v),
                     round(float(np.median(v) - np.median(ew)), 1),
                     round(float(perm_p(v, ew)), 4)))

        ax.set_xticks(xs)
        ax.set_xlim(4, 66)
        _style(ax, "k  (PPO updates between consolidations)", "final-20% return (ceiling 1000)",
               PRETTY[env])
        leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)
    fig.tight_layout()
    _write_csv(out_dir, "k_sweep_levels",
               ["environment", "arm", "k", "rho", "median_final20", "n",
                "minus_ewc", "p_exact_ranksum"], rows)
    _save(fig, out_dir, "k_sweep_levels")


def fig_k_differences(out_dir):
    """Each k-sweep arm minus EWC and minus pt at k = 10, as a dot plot.

    The grey bar is a 95% bootstrap interval and shows spread only; the dot colour and printed p
    come from the exact rank-sum test.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    arms = list(K_ARMS[1:]) + [RHO_ARM]          # k = 10 is a reference, not a row
    labels = ["k=%d, rho=%.1f" % (k, rho) for _, k, rho in arms]
    n = len(arms)
    band_alpha = 0.45
    band_bg = tuple(band_alpha * g + (1 - band_alpha) * s
                    for g, s in zip(mcolors.to_rgb(GRID), mcolors.to_rgb(SURFACE)))
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.2), facecolor=SURFACE, sharex="col")
    rows = []
    for col, env in enumerate(ENVS):
        cell = os.path.join(KSWEEP, env)
        refs = (("ewc", final20(cell, "ewc"), "minus EWC"),
                ("pt_k10_rho050", final20(cell, "pt_k10_rho050"), "minus pt at k = 10"))
        for row, (rname, ref, title) in enumerate(refs):
            ax = axes[row][col]
            for gi in range(0, n, 2):
                ax.axhspan(n - 1 - gi - 0.5, n - 1 - gi + 0.5, color=GRID, alpha=band_alpha,
                           linewidth=0, zorder=0)
            ax.axvline(0, color=INK_SECONDARY, linewidth=1.1, zorder=1)
            for gi, (arm, k, rho) in enumerate(arms):
                y = n - 1 - gi
                v = final20(cell, arm)
                obs = float(np.median(v) - np.median(ref))
                lo, hi = _boot(v, ref)
                pv = float(perm_p(v, ref))
                sig = pv < 0.05
                color = (PT_BLUE if obs > 0 else VANILLA_ORANGE) if sig else INK_MUTED
                ax.plot([lo, hi], [y, y], color=AXIS, linewidth=2.4, solid_capstyle="round",
                        zorder=2)
                ax.scatter([obs], [y], s=95, color=color, zorder=3, edgecolor=SURFACE,
                           linewidth=1.8)
                ax.annotate(("%+.1f   %s" % (obs, _fmt_p(pv))).replace("-", "\u2212"),
                            xy=(obs, y), xytext=(0, 11), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5,
                            fontweight="bold" if sig else "normal",
                            color=INK if sig else INK_MUTED, zorder=4,
                            bbox=dict(boxstyle="square,pad=0.14",
                                      facecolor=band_bg if gi % 2 == 0 else SURFACE,
                                      edgecolor="none"))
                rows.append((env, arm, rname, round(obs, 1), round(lo, 1), round(hi, 1),
                             round(pv, 4), "p<0.05" if sig else "ns"))
            ax.set_yticks(range(n))
            ax.set_yticklabels(list(reversed(labels)))
            ax.set_ylim(-0.6, n - 0.35)
            _style(ax, "", "", "%s  \u2014  %s" % (PRETTY[env], title))
            ax.grid(False, axis="y")
            ax.set_xlim(-140, 140)
    for ax in axes[1]:
        ax.set_xlabel("difference in final-20% return (points; ceiling 1000)",
                      color=INK, fontsize=10)
    key = [plt.Line2D([], [], marker="o", linestyle="none", markersize=9, color=PT_BLUE,
                      label="pt better, exact rank-sum p < 0.05"),
           plt.Line2D([], [], marker="o", linestyle="none", markersize=9, color=INK_MUTED,
                      label="not distinguishable from no difference"),
           plt.Line2D([], [], marker="o", linestyle="none", markersize=9, color=VANILLA_ORANGE,
                      label="pt worse, p < 0.05"),
           plt.Line2D([], [], color=AXIS, linewidth=2.4,
                      label="grey bar: 95% bootstrap spread (not the test)")]
    leg = fig.legend(handles=key, frameon=False, loc="lower center", ncol=4, fontsize=9,
                     bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    _write_csv(out_dir, "k_sweep_differences",
               ["environment", "arm", "reference", "difference", "ci_lo", "ci_hi",
                "p_exact_ranksum", "verdict"], rows)
    _save(fig, out_dir, "k_sweep_differences")


def fig_stationary_arms(out_dir):
    """Stationary control: median return of all three arms over training."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor=SURFACE)
    rows = []
    for ax, env in zip(axes, ENVS):
        cell = os.path.join(STATIONARY, env)
        for arm, color in STATIONARY_ARMS:
            grid, mat = curves(cell, arm)
            med = np.median(mat, axis=0)
            ax.fill_between(grid / 1e6, np.percentile(mat, 25, axis=0),
                            np.percentile(mat, 75, axis=0), color=color, alpha=0.13, linewidth=0)
            ax.plot(grid / 1e6, med, color=color, linewidth=2.0, solid_capstyle="round",
                    label=arm)
            rows += [(env, arm, int(g), round(float(m), 2))
                     for g, m in zip(grid[::40], med[::40])]
        ax.set_ylim(0, CEILING)
        _style(ax, "million env steps", "return (ceiling 1000)", PRETTY[env])
        leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)
    fig.tight_layout()
    _write_csv(out_dir, "stationary_arms", ["environment", "arm", "env_step", "median_return"],
               rows)
    _save(fig, out_dir, "stationary_arms")


def fig_stationary_windows(out_dir):
    """Stationary control: pt minus vanilla mean return in each of five task-length windows."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.6, 5.0), facecolor=SURFACE)
    rows = []
    width = 0.36
    for i, env in enumerate(ENVS):
        cell = os.path.join(STATIONARY, env)
        per = {}
        for arm in ("vanilla", "pt"):
            runs = []
            for p in sorted(glob.glob(os.path.join(cell, arm, "*_returns.pkl"))):
                if any(s in p for s in ("ep_returns", "eval_returns")):
                    continue
                a = np.asarray(load_pickle(p), dtype=float)
                if a.ndim == 2 and a[-1, 0] >= 3_000_000:
                    runs.append(a)
            per[arm] = [float(np.median([r[(r[:, 0] >= t * SWITCH) & (r[:, 0] < (t + 1) * SWITCH),
                                           1].mean() for r in runs])) for t in range(5)]
        diff = [b - a for a, b in zip(per["vanilla"], per["pt"])]
        x = np.arange(5) + (i - 0.5) * width
        ax.bar(x, diff, width=width * 0.92, color=PT_BLUE if i == 0 else EWC_AQUA,
               label=PRETTY[env], zorder=2)
        for xi, d in zip(x, diff):
            ax.annotate(("%+.1f" % d).replace("-", "\u2212"), xy=(xi, d),
                        xytext=(0, 4 if d >= 0 else -13), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_SECONDARY)
        rows += [(env, t + 1, round(per["vanilla"][t], 1), round(per["pt"][t], 1),
                  round(diff[t], 1)) for t in range(5)]
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.1, zorder=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["window %d" % (t + 1) for t in range(5)])
    _style(ax, "", "pt minus vanilla (return points)", None)
    ax.grid(False, axis="x")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    fig.tight_layout()
    _write_csv(out_dir, "stationary_windows",
               ["environment", "window", "vanilla", "pt", "pt_minus_vanilla"], rows)
    _save(fig, out_dir, "stationary_windows")


def main():
    p = argparse.ArgumentParser(description="k sweep and stationary control figures")
    p.add_argument("--out", default="figures_round2")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    out = _out_dir(args.out)
    fig_k_levels(out)
    fig_k_differences(out)
    fig_stationary_arms(out)
    fig_stationary_windows(out)
    print("\nAll figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
