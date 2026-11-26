"""Return-curve, pt-advantage and transfer figures for the three dm_control setups.

Run from the parent directory of the package:

    python -m src_continuous_control.plots.make_setup_figures

`SOURCES` maps each setup and environment to its result directory.
"""
import argparse
import glob
import os

import numpy as np

from ..scripts.report_multienv import CEILING, load_pickle, run_returns
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

ARMS = ("vanilla", "ewc", "pt")
ARM_COLOR = {"vanilla": VANILLA_ORANGE, "ewc": EWC_AQUA, "pt": PT_BLUE}
ARM_LABEL = {"vanilla": "vanilla", "ewc": "EWC", "pt": "pt"}

PRETTY = {
    "cartpole_swingup": "cartpole-swingup",
    "reacher_easy": "reacher-easy",
    "ball_in_cup_catch": "ball_in_cup-catch",
    "walker_stand": "walker-stand",
    "walker_walk": "walker-walk",
    "cheetah_run": "cheetah-run",
}

# Ordered by observation dimension, so the legend reads small-to-large body.
ENV_ORDER = ("cartpole_swingup", "reacher_easy", "ball_in_cup_catch",
             "cheetah_run", "walker_stand", "walker_walk")

# One hue per environment; lines are also labelled directly.
ENV_COLOR = {
    "cartpole_swingup": "#2a78d6",
    "reacher_easy": "#eb6834",
    "ball_in_cup_catch": "#1baf7a",
    "cheetah_run": "#8a4fbd",
    "walker_stand": "#c99700",
    "walker_walk": "#0f9bb5",
}

MD = "src_continuous_control/results/multienv"
DR = "src_continuous_control/results/multienv_drift"

# setup -> {environment: directory}. lipschitz1 cheetah-run lives in a separate result tree.
SOURCES = {
    "piecewise": {e: f"{MD}/{e}" for e in ENV_ORDER},
    "lipschitz1": {**{e: f"{DR}/lipschitz1/{e}" for e in
                      ("cartpole_swingup", "reacher_easy", "walker_stand", "walker_walk")},
                   "cheetah_run": f"{DR}/lipschitz1_boxB/cheetah_run"},
    "lipschitz2": {e: f"{DR}/lipschitz2/{e}" for e in
                   ("cartpole_swingup", "reacher_easy", "walker_stand", "walker_walk",
                    "cheetah_run")},
}

SETUP_TITLE = {
    "piecewise": "piecewise - physics change at 4 observable boundaries",
    "lipschitz1": "slow drift - one rate (2.5 cycles per run)",
    "lipschitz2": "slow + fast drift - slow drift plus a fast ripple (100 cycles per run)",
}
SWITCH = 614400


def _out_dir(name="figures_setups"):
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


def curves(cell_dir, arm, n_grid=400):
    """(steps, per-seed matrix) on a common grid, or (None, None) if the cell is absent."""
    out = []
    for p in sorted(glob.glob(os.path.join(cell_dir, arm, f"{arm}_ppo_seed_*_returns.pkl"))):
        if any(s in p for s in ("ep_returns", "eval_returns")):
            continue
        a = np.asarray(load_pickle(p), dtype=float)
        if a.ndim == 2 and a[-1, 0] >= 3_000_000:
            out.append(a)
    if not out:
        return None, None
    grid = np.linspace(0, min(c[-1, 0] for c in out), n_grid)
    return grid, np.vstack([np.interp(grid, c[:, 0], c[:, 1]) for c in out])


def final20(cell_dir, arm):
    """Per-seed final-20% return, as computed by `report_multienv.run_returns`.

    Uses the same definition as the result tables rather than the interpolated curves, so
    figure and tables agree. Runs shorter than 3M steps are skipped.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(cell_dir, arm, "*_returns.pkl"))):
        got = run_returns(path)
        if got is not None and got[2] >= 3_000_000:
            out.append(got[1])
    return np.asarray(out, dtype=float)


def fig_arms(setup, out_dir):
    """All three arms, one panel per environment."""
    import matplotlib.pyplot as plt
    envs = [e for e in ENV_ORDER if e in SOURCES[setup]]
    ncol = 3
    nrow = int(np.ceil(len(envs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.7 * nrow), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for ax, env in zip(axes, envs):
        ends = []
        for arm in ARMS:
            grid, mat = curves(SOURCES[setup][env], arm)
            if grid is None:
                continue
            med = np.median(mat, axis=0)
            ax.fill_between(grid / 1e6, np.percentile(mat, 25, axis=0),
                            np.percentile(mat, 75, axis=0),
                            color=ARM_COLOR[arm], alpha=0.13, linewidth=0)
            ax.plot(grid / 1e6, med, color=ARM_COLOR[arm], linewidth=2.0,
                    solid_capstyle="round")
            ends.append((float(med[-1]), arm, grid[-1] / 1e6))
            rows += [(setup, env, arm, int(g), round(float(m), 2))
                     for g, m in zip(grid[::40], med[::40])]
        # Offset end labels in display space so close curves do not overlap their labels.
        ends.sort()
        last = -1e9
        for y, arm, x in ends:
            y_lab = max(y, last + 52)
            last = y_lab
            ax.annotate(ARM_LABEL[arm], xy=(x, y), xytext=(5, y_lab - y),
                        textcoords="offset points", color=ARM_COLOR[arm],
                        fontsize=8, fontweight="bold", va="center")
        if setup == "piecewise":
            for b in range(1, 5):
                ax.axvline(b * SWITCH / 1e6, color=AXIS, linewidth=0.7, linestyle=(0, (4, 3)))
        ax.set_ylim(0, CEILING)
        ax.margins(x=0.12)
        _style(ax, "million env steps", "return (ceiling 1000)", PRETTY[env])
    for ax in axes[len(envs):]:
        ax.set_visible(False)
    sub = ("dashed lines are observable boundaries" if setup == "piecewise"
           else "no boundaries - physics move every step")
    # No figure title; the explanation is in the thesis caption.
    fig.tight_layout()
    _write_csv(out_dir, f"{setup}_arms",
               ["setup", "environment", "arm", "env_step", "median_return"], rows)
    _save(fig, out_dir, f"{setup}_arms")


def fig_pt_across_envs(setup, out_dir):
    """pt alone, every environment on one axis (all tasks share the 1000 ceiling)."""
    import matplotlib.pyplot as plt
    envs = [e for e in ENV_ORDER if e in SOURCES[setup]]
    fig, ax = plt.subplots(figsize=(10.5, 6.0), facecolor=SURFACE)
    rows, finals = [], []
    for env in envs:
        grid, mat = curves(SOURCES[setup][env], "pt")
        if grid is None:
            continue
        med = np.median(mat, axis=0)
        ax.plot(grid / 1e6, med, color=ENV_COLOR[env], linewidth=2.0,
                solid_capstyle="round", label=PRETTY[env])
        finals.append((med[-1], env, grid[-1] / 1e6))
        rows += [(setup, env, int(g), round(float(m), 2))
                 for g, m in zip(grid[::40], med[::40])]
    # Direct labels at the right edge, nudged apart where curves end close together.
    finals.sort()
    last = -1e9
    for y, env, x in finals:
        y_lab = max(y, last + 34)
        last = y_lab
        ax.annotate(PRETTY[env], xy=(x, y), xytext=(6, y_lab - y), textcoords="offset points",
                    color=ENV_COLOR[env], fontsize=9, fontweight="bold", va="center")
    if setup == "piecewise":
        for b in range(1, 5):
            ax.axvline(b * SWITCH / 1e6, color=AXIS, linewidth=0.7, linestyle=(0, (4, 3)))
    ax.set_ylim(0, CEILING)
    ax.margins(x=0.16)
    _style(ax, "million env steps", "pt return (ceiling 1000)",
           f"pt across environments - {SETUP_TITLE[setup]}\n"
           f"comparable only because every task is scored in [0,1] over exactly 1000 steps")
    _write_csv(out_dir, f"{setup}_pt", ["setup", "environment", "env_step", "median_return"], rows)
    _save(fig, out_dir, f"{setup}_pt")


def _transfer(cell_dir, arm):
    """Per-seed (FWT, BWT, peak return) arrays for one cell."""
    fwt, bwt, peak = [], [], []
    for p in sorted(glob.glob(os.path.join(cell_dir, arm, "*_transfer_matrix.pkl"))):
        d = load_pickle(p)
        if d.get("fwt") is not None:
            fwt.append(float(d["fwt"]))
        if d.get("bwt") is not None:
            bwt.append(float(d["bwt"]))
    for p in sorted(glob.glob(os.path.join(cell_dir, arm, f"{arm}_ppo_seed_*_returns.pkl"))):
        if any(s in p for s in ("ep_returns", "eval_returns")):
            continue
        peak.append(float(np.asarray(load_pickle(p), dtype=float)[:, 1].max()))
    return np.array(fwt), np.array(bwt), np.array(peak)


def fig_transfer(out_dir):
    """FWT, BWT and peak return side by side, piecewise setup only.

    Peak return is shown because retention metrics also improve when an agent learns less.
    """
    import matplotlib.pyplot as plt
    envs = [e for e in ENV_ORDER if e in SOURCES["piecewise"]]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), facecolor=SURFACE)
    panels = [("forward transfer (FWT)", 0), ("backward transfer (BWT)", 1),
              ("peak return - how much the arm learned", 2)]
    rows = []
    width = 0.26
    for ax, (title, which) in zip(axes, panels):
        for ai, arm in enumerate(ARMS):
            xs, ys, allv = [], [], []
            for gi, env in enumerate(envs):
                f, b, pk = _transfer(SOURCES["piecewise"][env], arm)
                vals = (f, b, pk)[which]
                if not len(vals):
                    continue
                pos = gi + (ai - 1) * width
                xs.append(pos)
                ys.append(float(np.median(vals)))
                allv.append((pos, vals))
            ax.bar(xs, ys, width=width * 0.9, color=ARM_COLOR[arm], linewidth=0,
                   label=ARM_LABEL[arm] if which == 0 else None)
            for pos, vals in allv:
                ax.scatter(np.full(len(vals), pos) + np.linspace(-0.05, 0.05, len(vals)), vals,
                           s=9, color=SURFACE, edgecolor=INK_SECONDARY, linewidth=0.6, zorder=3)
                rows += [("piecewise", envs[int(round(pos))], arm, title, round(float(v), 2))
                         for v in vals]
        ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
        ax.set_xticks(range(len(envs)))
        ax.set_xticklabels([PRETTY[e] for e in envs], rotation=28, ha="right", fontsize=8)
        _style(ax, "", "", title)
    leg = axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    fig.tight_layout()
    _write_csv(out_dir, "piecewise_transfer",
               ["setup", "environment", "arm", "metric", "value"], rows)
    _save(fig, out_dir, "piecewise_transfer")


def fig_transfer_split(out_dir):
    """The three panels of fig_transfer as separate images, without titles."""
    import matplotlib.pyplot as plt
    envs = [e for e in ENV_ORDER if e in SOURCES["piecewise"]]
    width = 0.26
    for name, ylabel, which in (("piecewise_transfer_fwt", "forward transfer", 0),
                                ("piecewise_transfer_bwt", "backward transfer", 1),
                                ("piecewise_transfer_peak", "peak return (ceiling 1000)", 2)):
        fig, ax = plt.subplots(figsize=(9.0, 4.4), facecolor=SURFACE)
        for ai, arm in enumerate(ARMS):
            xs, ys, allv = [], [], []
            for gi, env in enumerate(envs):
                vals = _transfer(SOURCES["piecewise"][env], arm)[which]
                if not len(vals):
                    continue
                pos = gi + (ai - 1) * width
                xs.append(pos)
                ys.append(float(np.median(vals)))
                allv.append((pos, vals))
            ax.bar(xs, ys, width=width * 0.9, color=ARM_COLOR[arm], linewidth=0,
                   label=ARM_LABEL[arm])
            for pos, vals in allv:
                ax.scatter(np.full(len(vals), pos) + np.linspace(-0.05, 0.05, len(vals)), vals,
                           s=11, color=SURFACE, edgecolor=INK_SECONDARY, linewidth=0.6, zorder=3)
        ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
        ax.set_xticks(range(len(envs)))
        ax.set_xticklabels([PRETTY[e] for e in envs], fontsize=9)
        _style(ax, "", ylabel, None)
        # Legend below the axes so it never covers a bar.
        leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                        ncol=3, fontsize=9)
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)
        fig.tight_layout()
        _save(fig, out_dir, name)


def _pt_panel(ax, setup, envs, rows, label_gap=34):
    """Draw pt's median curve for each environment of one setup on `ax`."""
    finals = []
    for env in envs:
        grid, mat = curves(SOURCES[setup][env], "pt")
        if grid is None:
            continue
        med = np.median(mat, axis=0)
        ax.plot(grid / 1e6, med, color=ENV_COLOR[env], linewidth=1.9, solid_capstyle="round")
        finals.append((float(med[-1]), env, grid[-1] / 1e6))
        rows += [(setup, env, int(g), round(float(m), 2)) for g, m in zip(grid[::40], med[::40])]
    finals.sort()
    last = -1e9
    for y, env, x in finals:
        y_lab = max(y, last + label_gap)
        last = y_lab
        ax.annotate(PRETTY[env], xy=(x, y), xytext=(5, y_lab - y), textcoords="offset points",
                    color=ENV_COLOR[env], fontsize=8, fontweight="bold", va="center")
    if setup == "piecewise":
        for b in range(1, 5):
            ax.axvline(b * SWITCH / 1e6, color=AXIS, linewidth=0.7, linestyle=(0, (4, 3)))
    ax.set_ylim(0, CEILING)
    ax.margins(x=0.30)


def fig_pt_all_setups(out_dir):
    """pt's curves for the three setups in one image.

    The y axis is shared, but the attainable return differs per environment, so vertical order
    reflects environment difficulty. fig_pt_advantage_all_setups compares pt against vanilla.
    """
    import matplotlib.pyplot as plt
    setups = ("piecewise", "lipschitz1", "lipschitz2")
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), facecolor=SURFACE)
    rows = []
    for ax, setup in zip(axes, setups):
        envs = [e for e in ENV_ORDER if e in SOURCES[setup]]
        _pt_panel(ax, setup, envs, rows)
        _style(ax, "million env steps",
               "pt return (ceiling 1000)" if setup == "piecewise" else "",
               SETUP_TITLE[setup].split(" - ")[0])
    fig.tight_layout()
    _write_csv(out_dir, "pt_across_environments",
               ["setup", "environment", "env_step", "median_return"], rows)
    _save(fig, out_dir, "pt_across_environments")


def fig_pt_advantage_all_setups(out_dir):
    """Median pt return minus median vanilla return, per environment, over training.

    The within-environment difference removes each environment's difficulty and reward scale.
    No uncertainty is drawn.
    """
    import matplotlib.pyplot as plt
    setups = ("piecewise", "lipschitz1", "lipschitz2")
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), facecolor=SURFACE)
    rows = []
    for ax, setup in zip(axes, setups):
        envs = [e for e in ENV_ORDER if e in SOURCES[setup]]
        finals = []
        for env in envs:
            g_pt, m_pt = curves(SOURCES[setup][env], "pt")
            g_v, m_v = curves(SOURCES[setup][env], "vanilla")
            if g_pt is None or g_v is None:
                continue
            n = min(len(g_pt), len(g_v))
            diff = np.median(m_pt, axis=0)[:n] - np.median(m_v, axis=0)[:n]
            ax.plot(g_pt[:n] / 1e6, diff, color=ENV_COLOR[env], linewidth=1.9,
                    solid_capstyle="round")
            finals.append((float(diff[-1]), env, g_pt[n - 1] / 1e6))
            rows += [(setup, env, int(g), round(float(d), 2))
                     for g, d in zip(g_pt[:n:40], diff[::40])]
        ax.axhline(0, color=INK_SECONDARY, linewidth=1.1)
        finals.sort()
        last = -1e9
        for y, env, x in finals:
            y_lab = max(y, last + 26)
            last = y_lab
            ax.annotate(PRETTY[env], xy=(x, y), xytext=(5, y_lab - y),
                        textcoords="offset points", color=ENV_COLOR[env],
                        fontsize=8, fontweight="bold", va="center")
        if setup == "piecewise":
            for b in range(1, 5):
                ax.axvline(b * SWITCH / 1e6, color=AXIS, linewidth=0.7, linestyle=(0, (4, 3)))
        ax.margins(x=0.30)
        _style(ax, "million env steps",
               "pt return minus vanilla return" if setup == "piecewise" else "",
               SETUP_TITLE[setup].split(" - ")[0])
    fig.tight_layout()
    _write_csv(out_dir, "pt_advantage_over_vanilla",
               ["setup", "environment", "env_step", "pt_minus_vanilla"], rows)
    _save(fig, out_dir, "pt_advantage_over_vanilla")


def _fmt_p(p):
    """Format a p-value without the leading zero (p=.005)."""
    return "p<.001" if p < 0.001 else ("p=%.3f" % p).replace("0.", ".", 1)


def fig_advantage_endpoints(out_dir):
    """Dot plot of the final-20% return difference, pt minus vanilla, per setup and environment.

    One panel per setup. The dot colour and printed p come from the exact rank-sum test; the grey
    bar is a 95% bootstrap interval of the difference of medians and shows spread only.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from ..scripts.report_tables import perm_p

    setups = ("piecewise", "lipschitz1", "lipschitz2")
    col_title = {"piecewise": ("piecewise", "physics jump at 4 observable boundaries"),
                 "lipschitz1": ("slow drift", "one rate, 2.5 cycles per run"),
                 "lipschitz2": ("slow + fast drift", "plus a fast ripple, 100 cycles")}
    # Polarity encoding: two poles and a neutral grey for "no significant difference".
    BETTER, WORSE, NEITHER = PT_BLUE, VANILLA_ORANGE, INK_MUTED

    envs = list(ENV_ORDER)
    n = len(envs)
    # Blended colour of a shaded row, used as the label background so labels hide the zero line.
    band_alpha = 0.45
    band_bg = tuple(band_alpha * g + (1 - band_alpha) * su for g, su in
                    zip(mcolors.to_rgb(GRID), mcolors.to_rgb(SURFACE)))
    row_bg = {gi: (band_bg if gi % 2 == 0 else SURFACE) for gi in range(n)}
    # Rows are drawn top to bottom in ENV_ORDER.
    xlo, xhi = -300.0, 420.0

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 7.0), facecolor=SURFACE,
                             sharex=True, sharey=True)
    rows = []
    rng = np.random.RandomState(0)
    for ax, setup in zip(axes, setups):
        # Alternating row bands, drawn first, below everything.
        for gi in range(0, n, 2):
            ax.axhspan(n - 1 - gi - 0.5, n - 1 - gi + 0.5, color=GRID, alpha=band_alpha,
                       linewidth=0, zorder=0)
        ax.axvline(0, color=INK_SECONDARY, linewidth=1.1, zorder=1)

        for gi, env in enumerate(envs):
            y = n - 1 - gi
            if env not in SOURCES[setup]:
                ax.text(0, y, "  not run", color=INK_MUTED, fontsize=8.5, style="italic",
                        va="center", ha="left", zorder=4)
                continue
            f_pt = final20(SOURCES[setup][env], "pt")
            f_v = final20(SOURCES[setup][env], "vanilla")
            if len(f_pt) == 0 or len(f_v) == 0:
                continue
            obs = float(np.median(f_pt) - np.median(f_v))
            boot = [np.median(f_pt[rng.randint(0, len(f_pt), len(f_pt))])
                    - np.median(f_v[rng.randint(0, len(f_v), len(f_v))]) for _ in range(4000)]
            lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
            pv = float(perm_p(f_pt, f_v))
            sig = pv < 0.05
            color = (BETTER if obs > 0 else WORSE) if sig else NEITHER

            # Bootstrap interval in neutral grey (spread only, not the test).
            ax.plot([max(lo, xlo), min(hi, xhi)], [y, y], color=AXIS, linewidth=2.4,
                    solid_capstyle="round", zorder=2)
            # Intervals beyond the axis limits are clipped and marked with an arrowhead.
            if hi > xhi:
                ax.plot([xhi], [y], marker=">", markersize=5, color=AXIS, zorder=2)
            if lo < xlo:
                ax.plot([xlo], [y], marker="<", markersize=5, color=AXIS, zorder=2)
            ax.scatter([obs], [y], s=95, color=color, zorder=3,
                       edgecolor=SURFACE, linewidth=1.8)
            # Effect and p above the dot, to one decimal to match the result tables.
            label = ("%+.1f   %s" % (obs, _fmt_p(pv))).replace("-", "−")
            ax.annotate(label, xy=(obs, y),
                        xytext=(0, 11), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold" if sig else "normal",
                        color=INK if sig else INK_MUTED, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.14", facecolor=row_bg[gi],
                                  edgecolor="none"))
            rows.append((setup, env, round(obs, 1), round(lo, 1), round(hi, 1),
                         round(pv, 4), "p<0.05" if sig else "ns"))

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(range(n))
        ax.set_yticklabels([PRETTY[e] for e in reversed(envs)])
        _style(ax, "", "", None)
        ax.grid(False, axis="y")
        ax.set_xticks([-200, 0, 200, 400])
        # Panel title is the setup name only.
        head, _sub = col_title[setup]
        ax.set_title(head, color=INK, fontsize=10.5, loc="left", pad=10,
                     fontweight="bold")
        for t in ax.get_yticklabels():
            t.set_fontsize(9.5)

    axes[1].set_xlabel("pt final-20% return minus vanilla, same environment "
                       "(return points; every task ceiling is 1000)",
                       color=INK, fontsize=10, labelpad=8)
    # Direction labels for the zero line.
    axes[0].annotate("← vanilla better", xy=(0.0, -0.10), xycoords="axes fraction",
                     ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    axes[2].annotate("pt better →", xy=(1.0, -0.10), xycoords="axes fraction",
                     ha="right", va="top", fontsize=8.5, color=INK_MUTED)

    key = [plt.Line2D([], [], marker="o", linestyle="none", markersize=9, color=BETTER,
                      label="pt better, exact rank-sum p < 0.05"),
           plt.Line2D([], [], marker="o", linestyle="none", markersize=9, color=NEITHER,
                      label="not distinguishable from no difference"),
           plt.Line2D([], [], marker="o", linestyle="none", markersize=9, color=WORSE,
                      label="pt worse, p < 0.05"),
           plt.Line2D([], [], color=AXIS, linewidth=2.4,
                      label="grey bar: 95% bootstrap spread of the difference (not the test)")]
    leg = fig.legend(handles=key, frameon=False, loc="lower center", ncol=4, fontsize=9,
                     bbox_to_anchor=(0.5, 0.0), handletextpad=0.6, columnspacing=2.2)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    # tight_layout ignores a figure legend, so reserve a strip at the bottom for it.
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    _write_csv(out_dir, "pt_advantage_endpoints",
               ["setup", "environment", "pt_minus_vanilla", "ci_lo", "ci_hi",
                "p_exact_ranksum", "verdict"], rows)
    _save(fig, out_dir, "pt_advantage_endpoints")


def main():
    p = argparse.ArgumentParser(description="per-setup return and transfer figures")
    p.add_argument("--out", default="figures_setups")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    out = _out_dir(args.out)
    for setup in ("piecewise", "lipschitz1", "lipschitz2"):
        fig_arms(setup, out)
        fig_pt_across_envs(setup, out)
    fig_pt_all_setups(out)
    fig_pt_advantage_all_setups(out)
    fig_advantage_endpoints(out)
    fig_transfer(out)
    fig_transfer_split(out)
    print("\nAll figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
