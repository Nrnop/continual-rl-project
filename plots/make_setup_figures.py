"""Return curves and transfer figures, one set per non-stationarity setup.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.plots.make_setup_figures

Eight figures, in three groups:

  <setup>_arms      one panel per environment, all three arms on each. "Does pt beat the
                    baselines, and where" -- read one panel at a time.
  <setup>_pt        pt alone, every environment on ONE axis. Legitimate only because every
                    dm_control task here is scored in [0,1] over exactly 1000 steps, so the
                    ceiling is 1000 everywhere and the curves are directly comparable. This is
                    the plot for "which environments does pt do well in", which the per-panel
                    view cannot answer.
  piecewise_transfer  forward and backward transfer, per environment, per arm -- and PEAK RETURN
                    beside them, because every retention metric improves when an agent simply
                    learns less. Only defined for the piecewise setup: FWT/BWT are indexed by
                    task number and the drift setups have no tasks.

WHERE THE DATA COMES FROM. The drift setups were run across two machines: four environments on
the Ryzen box and cheetah-run on an EPYC box. `SOURCES` records that per environment rather than
hiding it, because a curve's provenance is part of what it means.

COLOR is the project's existing validated three-hue set (see make_multienv_figures.py). For the
pt-across-environments figures, environments need six distinguishable colours, which is a
different job -- that uses a sequential ordering by observation dimension, with every line
directly labelled so identity is never colour-alone.
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

# Six distinguishable hues for the per-environment figures. Every line is also directly
# labelled, so colour is never the only carrier of identity.
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

# setup -> {environment: directory}. cheetah's drift cells live under the EPYC box's tree.
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
    "lipschitz1": "Lipschitz1 - smooth drift, one rate (2.5 cycles per run)",
    "lipschitz2": "Lipschitz2 - slow drift plus a fast ripple (100 cycles per run)",
}
SWITCH = 614400


def _out_dir(name="figures_setups"):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, out_dir, name):
    import matplotlib.pyplot as plt
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


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
    """Per-seed final-20% return, using the PROJECT'S STANDING DEFINITION.

    This deliberately calls `report_multienv.run_returns` rather than slicing the interpolated
    curves above. The endpoint figure originally took the last 20% of a common 400-point grid,
    which is a third definition of "final 20%" and disagreed with every table in the study by up
    to 0.7 return points -- small, but enough that a reader checking the figure against section 5
    finds two different numbers for one cell. One definition, one number.

    Keeps `curves`'s guard on run length so an aborted run cannot enter a median.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(cell_dir, arm, "*_returns.pkl"))):
        got = run_returns(path)
        if got is not None and got[2] >= 3_000_000:
            out.append(got[1])
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Group 1: all three arms, one panel per environment
# ---------------------------------------------------------------------------
def fig_arms(setup, out_dir):
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
        # Arms often finish within a few points of each other, which stacks the labels on top
        # of one another. Push them apart in DISPLAY space, keeping each anchored to its own
        # curve, so identity stays readable without moving the data.
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
    fig.suptitle(f"{SETUP_TITLE[setup]}\nmedian of 10 seeds, IQR shaded; {sub}",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _write_csv(out_dir, f"{setup}_arms",
               ["setup", "environment", "arm", "env_step", "median_return"], rows)
    _save(fig, out_dir, f"{setup}_arms")


# ---------------------------------------------------------------------------
# Group 2: pt alone, every environment on one axis
# ---------------------------------------------------------------------------
def fig_pt_across_envs(setup, out_dir):
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


# ---------------------------------------------------------------------------
# Group 3: forward / backward transfer, piecewise only
# ---------------------------------------------------------------------------
def _transfer(cell_dir, arm):
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
    """FWT, BWT and PEAK RETURN side by side.

    The third panel is not decoration. Every retention-flavoured metric improves when an agent
    simply learns less -- on HalfCheetah a frozen arm scored BWT ~ 0 with a peak return of exactly
    0.0 -- so backward transfer must never be read without what the arm achieved beside it.
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
    fig.suptitle("Forward and backward transfer, piecewise setup - with peak return beside them.\n"
                 "Retention metrics improve when an agent learns less, so the third panel is part "
                 "of reading the first two. Dots are individual seeds.",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _write_csv(out_dir, "piecewise_transfer",
               ["setup", "environment", "arm", "metric", "value"], rows)
    _save(fig, out_dir, "piecewise_transfer")


# ---------------------------------------------------------------------------
# Group 2b: the three setups side by side, in one image
# ---------------------------------------------------------------------------
def _pt_panel(ax, setup, envs, rows, label_gap=34):
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
    """The three setups in one image, pt only.

    READ THIS AS SHAPE, NOT AS RANKING. Every task is scored in [0,1] over exactly 1000 steps, so
    the axis is shared -- but a shared axis is not a shared difficulty. 400 on cheetah-run means
    averaging ~4 m/s against a 10 m/s target; 400 on cartpole means a product of four partially
    satisfied factors. No agent reaches 1000 anywhere, and the ATTAINABLE maximum differs per
    environment, so the vertical ordering here is mostly a difficulty ranking of the environments,
    not a statement about where pt does well. For that question see fig_pt_advantage_all_setups.
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
    fig.suptitle("pt across environments, all three setups - SHAPE, not ranking.\n"
                 "The axis is shared because every task scores in [0,1] over 1000 steps, but a "
                 "shared axis is not a shared difficulty:\nthe vertical order here mostly ranks "
                 "how hard the environments are, not where pt helps.",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _write_csv(out_dir, "pt_across_environments",
               ["setup", "environment", "env_step", "median_return"], rows)
    _save(fig, out_dir, "pt_across_environments")


def fig_pt_advantage_all_setups(out_dir):
    """What the previous figure looks like it answers, but cannot: where does pt actually help.

    Plots pt's return MINUS vanilla's, in the same environment, over training. A within-environment
    difference is comparable across environments in a way that a raw return is not: it cancels the
    environment's own difficulty, its reward scale, and its attainable maximum, leaving only what
    the method changed.
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
    fig.suptitle("Where pt actually helps: pt MINUS vanilla, same environment, over training.\n"
                 "SHAPE ONLY. No uncertainty is drawn here, and this is a difference of two noisy "
                 "medians, so most wiggles\nare sampling noise. For which gaps are actually real, "
                 "see pt_advantage_endpoints.png.",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _write_csv(out_dir, "pt_advantage_over_vanilla",
               ["setup", "environment", "env_step", "pt_minus_vanilla"], rows)
    _save(fig, out_dir, "pt_advantage_over_vanilla")


def _fmt_p(p):
    """.005 rather than 0.005 -- the leading zero is noise when 16 of these sit in one figure."""
    return "p<.001" if p < 0.001 else ("p=%.3f" % p).replace("0.", ".", 1)


def fig_advantage_endpoints(out_dir):
    """The figure that actually answers "which environments does pt help in".

    WHY THIS REPLACED A CURVE. The over-time version puts six environments in one panel with no
    uncertainty, and a reader immediately -- and reasonably -- read a positive blip on reacher as
    "pt helps here", when the endpoint is +17.3 with p = 0.58, i.e. nothing at all. A difference of
    two medians over 10 seeds is noisy enough that the curve cannot be read safely without
    intervals, and six overlapping bands in one panel are unreadable. A dot plot of the ENDPOINT
    with a confidence interval says the same thing without inviting the mistake.

    WHY IT WAS THEN REDRAWN (the version below). The first dot plot stacked three unlabelled rows
    per environment and carried TWO statistical encodings at once: a bootstrap interval, and a
    bold/faded verdict from the exact rank-sum test. Those two disagree by construction -- reacher
    piecewise is p = 0.043 with an interval spanning zero -- so the figure quietly gave two answers
    per cell and a reader had no way to know which one to believe. Three fixes:

      1. SETUP BECOMES POSITION, NOT COLOUR. One panel per setup, so no legend has to be held in
         the head while reading row two of six. Colour is freed for the thing being asked about.
      2. ONE VERDICT, AND IT IS WRITTEN DOWN. The interval is drawn in NEUTRAL GREY and is spread
         only; the coloured dot and the printed p carry the exact rank-sum verdict, which is the
         project's standard test and the one every table uses. A grey bar beside a coloured dot
         cannot be mistaken for two competing verdicts the way bold-vs-faded intervals were.
      3. THE NUMBER IS ON THE FIGURE. Effect and p are printed at every dot, so colour is never
         the only channel and the figure never has to be read against section 5's table.

    Final-20% return, pt minus vanilla, 10 seeds per arm.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from ..scripts.report_tables import perm_p

    setups = ("piecewise", "lipschitz1", "lipschitz2")
    col_title = {"piecewise": ("piecewise", "physics jump at 4 observable boundaries"),
                 "lipschitz1": ("Lipschitz1", "smooth drift, 2.5 cycles per run"),
                 "lipschitz2": ("Lipschitz2", "slow drift + fast ripple, 100 cycles")}
    # A polarity encoding, not a categorical one: two poles and a neutral grey middle. The grey
    # deliberately fails a categorical palette's chroma floor -- reading as grey IS its job here.
    BETTER, WORSE, NEITHER = PT_BLUE, VANILLA_ORANGE, INK_MUTED

    envs = list(ENV_ORDER)
    n = len(envs)
    # The exact colour of a shaded row, so a label can be given that background and punch the
    # zero line out from behind itself. Kept computed, not hard-coded, so it tracks the palette.
    band_alpha = 0.45
    band_bg = tuple(band_alpha * g + (1 - band_alpha) * su for g, su in
                    zip(mcolors.to_rgb(GRID), mcolors.to_rgb(SURFACE)))
    row_bg = {gi: (band_bg if gi % 2 == 0 else SURFACE) for gi in range(n)}
    # ENV_ORDER runs small body -> large; inverting y makes it read downwards, the way the row
    # labels are listed everywhere else in this project.
    xlo, xhi = -300.0, 420.0

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 7.0), facecolor=SURFACE,
                             sharex=True, sharey=True)
    rows = []
    rng = np.random.RandomState(0)
    for ax, setup in zip(axes, setups):
        # Alternating row bands: with three panels to scan across, they hold the eye on one
        # environment. Drawn first, below everything.
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

            # The interval is SPREAD, in neutral grey, so it cannot be read as a second verdict.
            ax.plot([max(lo, xlo), min(hi, xhi)], [y, y], color=AXIS, linewidth=2.4,
                    solid_capstyle="round", zorder=2)
            # One interval (walker-stand, piecewise) runs off the axis. Cap it rather than widen
            # the axis for a single cell and squash the other fifteen.
            if hi > xhi:
                ax.plot([xhi], [y], marker=">", markersize=5, color=AXIS, zorder=2)
            if lo < xlo:
                ax.plot([xlo], [y], marker="<", markersize=5, color=AXIS, zorder=2)
            ax.scatter([obs], [y], s=95, color=color, zorder=3,
                       edgecolor=SURFACE, linewidth=1.8)
            # The number, above its own dot: colour is then never the only channel, and the
            # figure answers without the reader going to section 5's table.
            # One decimal, not an integer: section 5 prints the levels to one decimal and its
            # differences must equal their difference exactly. Rounding both to integers breaks
            # that arithmetic (703 - 637 = 66 while the true difference is 66.5), so the figure
            # prints the same string the table does.
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
        head, sub = col_title[setup]
        ax.set_title("%s\n%s" % (head, sub), color=INK, fontsize=10.5, loc="left", pad=10,
                     fontweight="bold")
        for t in ax.get_yticklabels():
            t.set_fontsize(9.5)

    axes[1].set_xlabel("pt final-20% return minus vanilla, same environment "
                       "(return points; every task ceiling is 1000)",
                       color=INK, fontsize=10, labelpad=8)
    # Which way is good. The zero line means nothing without it.
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

    fig.suptitle("Which environments does pt help in?\n"
                 "Final-20% median return over 10 seeds per arm, pt minus vanilla. "
                 "6 of 16 cells separate from zero.",
                 color=INK, fontsize=12.5, x=0.008, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0.06, 1, 0.945))
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
    print("\nAll figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
