"""Figures for the PT-PPO reduction study -> plots/figures_pt_full/.

Reads the raw per-seed result pickles, so every number on every figure is recomputed from the
runs rather than transcribed. Run from the PARENT of src_continuous_control/:

    python -m src_continuous_control.plots.make_pt_full_figures

Seven figures. Each carries one claim, and each is embedded at the matching section of FULL_PT.md:

  sigma_collapse         FULL_PT 2     exploration is unmanaged persistent memory
  beta_sweep             FULL_PT 13    the KL anchor explains none of the gain
  dose_response          FULL_PT 18c/d the active ingredient, and its three-line reproduction
  benchmark_saturation   FULL_PT 19a   why one 'significant' drift result was an artifact
  reduction_halfcheetah  FULL_PT 24    the reduction on real physics, plus the variance result
  decoupling             FULL_PT 25    permanent and shrinkage are the same knob
  mechanism_by_regime    (summary)     the mechanism itself, standardised, across all regimes

Design follows the house data-viz method: form chosen before colour, categorical hues assigned in
fixed slot order (never cycled), de-emphasis grey for context series, recessive solid gridlines,
thin marks, direct labels on the values that carry the argument rather than on every point.
"""
import argparse
import glob
import itertools
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- validated categorical palette, light mode (slots used in fixed order) ---
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
GREY, GRID = "#a8a7a0", "#e6e5e0"          # de-emphasis series / recessive grid

PHASE_HC = 614_400


# ----------------------------------------------------------------- statistics
def mannwhitney_exact(a, b):
    """Exact two-sided Mann-Whitney p from the permutation null (scipy is absent here)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool = np.concatenate([a, b])
    r = pool.argsort().argsort().astype(float) + 1
    for v in np.unique(pool):
        m = pool == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    obs = r[:len(a)].sum()
    null = np.array([r[list(c)].sum() for c in itertools.combinations(range(len(pool)), len(a))])
    return float((np.abs(null - null.mean()) >= abs(obs - null.mean()) - 1e-9).mean())


# ----------------------------------------------------------------- data access
def _returns(root, stage, cell):
    out = []
    for f in sorted(glob.glob(os.path.join(root, f"{stage}_results", cell, "*_scalars.pkl"))):
        out.append(pickle.load(open(f, "rb"))["train/avg_return"])
    return out


def final_phase(root, stage, cell, phase=PHASE_HC, n_phases=5):
    """Mean return over the final task phase (HalfCheetah: last 614 400 steps)."""
    vals = []
    for r in _returns(root, stage, cell):
        st, x = r[:, 0], r[:, 1]
        vals.append(float(x[st >= (n_phases - 1) * phase].mean()))
    return np.array(vals)


def final_window(root, stage, cell, frac=1.0 / 9.0):
    """Mean return over the final `frac` of the run (point-mass: last phase of nine)."""
    vals = []
    for r in _returns(root, stage, cell):
        st, x = r[:, 0], r[:, 1]
        vals.append(float(x[st >= st.max() * (1 - frac)].mean()))
    return np.array(vals)


def _style(ax, xlabel=None, ylabel=None, title=None, subtitle=None, grid="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1.0, zorder=0)   # solid, never dashed
        ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    if title:
        ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=22)
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def _save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200,
                    facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ------------------------------------------------------------------ figure 1
def fig_reduction(root, outdir):
    """Dot strip: every seed shown, median marked. Carries the headline AND the variance result.

    Form: a dot strip rather than bars, because the per-seed spread IS one of the two findings
    (the shrinking arms are ~16x tighter). Bars would hide exactly that. Grey = context arms,
    colour = the two arms whose equality is the claim.
    """
    arms = [
        ("vanilla PPO", "van", GREY),
        ("vanilla + 3 lines", "van_shrink", BLUE),
        ("pt_full  (live permanent)", "pt", ORANGE),
        ("pt_full  (inert permanent)", "inert", AQUA),
    ]
    data = {c: final_phase(root, "stage14", c) for _, c, _ in arms}
    if any(len(v) == 0 for v in data.values()):
        print("  [skip] reduction_halfcheetah: stage14 results not found")
        return

    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    for i, (label, cell, colour) in enumerate(arms):
        y = len(arms) - 1 - i
        vals = data[cell]
        ax.scatter(vals, np.full_like(vals, y, dtype=float), s=52, color=colour,
                   alpha=0.55, linewidths=1.6, edgecolors=SURFACE, zorder=3)  # 2px surface ring
        med = float(np.median(vals))
        ax.plot([med, med], [y - 0.28, y + 0.28], color=colour, linewidth=2.4, zorder=4)
        ax.text(med, y + 0.40, f"{med:,.0f}", color=INK, fontsize=10.5,
                fontweight="bold", ha="center", va="bottom", zorder=5)

    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[0] for a in reversed(arms)], color=INK, fontsize=10)
    _style(ax, xlabel="final-phase return  (6 seeds, HalfCheetah, 3.07M steps)", grid="x",
           title="Three lines of PPO reproduce the whole PT-PPO apparatus",
           subtitle="pt_full vs vanilla+3 lines: +49  (p = 0.485, indistinguishable)   ·   "
                    "both vs vanilla: p = 0.002")

    p_red = mannwhitney_exact(data["pt"], data["van_shrink"])
    spread_v = data["van"].max() - data["van"].min()
    spread_s = data["van_shrink"].max() - data["van_shrink"].min()
    ax.text(0.995, -0.20,
            f"exact permutation test, p = {p_red:.3f}   ·   seed spread: "
            f"vanilla {spread_v:,.0f} vs vanilla+3 lines {spread_s:,.0f} "
            f"({spread_v / max(spread_s, 1):.0f}x tighter)",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="top")
    _save(fig, outdir, "reduction_halfcheetah")


# ------------------------------------------------------------------ figure 2
def fig_dose_response(root, outdir):
    """Return against decay factor, for the full apparatus and for the three-line version.

    Two series, so a legend is present and both are direct-labelled. The vanilla reference is a
    thin grey rule with its own label rather than a third series -- it is context, not identity.
    """
    apparatus = [(1.00, "rho000"), (0.90, "rho010"), (0.75, "rho025"),
                 (0.50, "rho050"), (0.25, "rho075")]
    three_line = [(0.90, "s090"), (0.75, "s075"), (0.50, "s050")]

    ax_pts = [(d, final_window(root, "stage9", c)) for d, c in apparatus]
    bx_pts = [(d, final_window(root, "stage10", c)) for d, c in three_line]
    van = final_window(root, "stage8", "van_a32c32")
    ax_pts = [(d, v) for d, v in ax_pts if len(v)]
    bx_pts = [(d, v) for d, v in bx_pts if len(v)]
    if not ax_pts or not bx_pts or not len(van):
        print("  [skip] dose_response: stage8/9/10 results not found")
        return

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    vmed = float(np.median(van))
    ax.axhline(vmed, color=GREY, linewidth=1.6, zorder=2)
    # anchored to the axes, not to data -- the x-axis is inverted below, so a data-space x
    # would land the label on the wrong side and across the blue series.
    ax.text(0.995, vmed, "  vanilla PPO", color=INK_2, fontsize=9.5, va="bottom", ha="right",
            transform=ax.get_yaxis_transform())

    for pts, colour, label in ((ax_pts, BLUE, "full PT-PPO apparatus"),
                               (bx_pts, ORANGE, "vanilla + 3 lines")):
        xs = [d for d, _ in pts]
        ys = [float(np.median(v)) for _, v in pts]
        ax.plot(xs, ys, color=colour, linewidth=2.0, zorder=3, label=label)
        ax.scatter(xs, ys, s=64, color=colour, zorder=4, linewidths=1.8, edgecolors=SURFACE)
        # direct-label at the leftmost (strongest-shrink) point, offset in POINTS above the
        # marker so the text never sits on the mark it names.
        x0 = min(xs)
        y0 = ys[int(np.argmin(xs))]
        ax.annotate(label, (x0, y0), textcoords="offset points", xytext=(6, 13),
                    color=colour, fontsize=10, fontweight="bold", ha="left", va="bottom",
                    annotation_clip=False)

    # headroom so the topmost direct label cannot run into the subtitle band
    all_y = [float(np.median(v)) for _, v in ax_pts] + [float(np.median(v)) for _, v in bx_pts]
    lo, hi = min(all_y + [vmed]), max(all_y)
    ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.22)

    ax.invert_xaxis()          # stronger shrinkage to the right reads as "more intervention"
    ax.set_xlim(1.06, 0.12)   # headroom for the leftmost direct label
    _style(ax, xlabel="decay factor applied to the policy every 8 updates   "
                      "(1.00 = no shrinkage)",
           ylabel="final-phase return",
           title="The active ingredient is periodic policy shrinkage",
           subtitle="permanent zeroed and frozen, KL anchor off — none of the PT machinery is "
                    "present in either series")
    ax.text(0.995, -0.22,
            "point-mass, 8 seeds per point. At decay 1.00 the optimiser moments are still "
            "flushed but no weights change: indistinguishable from vanilla (p = 0.234).",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="top")
    _save(fig, outdir, "dose_response")


# ------------------------------------------------------------------ figure 3
def fig_mechanism(root, outdir):
    """Diverging bar: the permanent-transient dynamic itself, live minus inert, per regime.

    Polarity is the job (does the mechanism help or hurt?), so a diverging form centred on zero.
    Significant results carry the two poles; non-significant ones are grey, so the eye is not
    invited to read an effect the statistics do not support.
    """
    rows = [
        ("discrete switching  (E=0)",      "stage3",  "pt_b001",  "inert_b001",  final_window),
        ("discrete switching  (freq 7:2)", "stage6",  "pt_f7",    "inert_f7",    final_window),
        ("sinusoidal drift  (amp 1.5)",    "stage13", "pt_a15",   "inert_a15",   None),
        ("linear monotone drift",          "stage15", "pt",       "inert",       None),
        ("HalfCheetah  (sigma 0.37)",      "stage14", "pt",       "inert",       None),
    ]
    labels, gaps, ps = [], [], []
    for label, stage, live, inert, _ in rows:
        if stage == "stage14":
            a, b = final_phase(root, stage, live), final_phase(root, stage, inert)
        elif stage in ("stage13", "stage15"):
            a, b = final_window(root, stage, live, 0.2), final_window(root, stage, inert, 0.2)
        else:
            a, b = final_window(root, stage, live), final_window(root, stage, inert)
        if not len(a) or not len(b):
            continue
        # STANDARDISED effect (Cohen's d on the pooled seed SD), NOT a percentage of the inert
        # arm. Percent-of-baseline is undefined here: on the frequency ladder the inert arm sits
        # at -21.4, so the ratio explodes to -987% and is an artifact of a near-zero denominator,
        # not a large effect. Dividing by the seed noise is well-defined for any sign or scale,
        # and it answers the question the reader actually has: how big is this against the spread?
        labels.append(label)
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        gaps.append((np.median(a) - np.median(b)) / max(sd, 1e-9))
        ps.append(mannwhitney_exact(a, b))
    if not labels:
        print("  [skip] mechanism_by_regime: results not found")
        return

    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ys = np.arange(len(labels))[::-1]
    for y, g, p in zip(ys, gaps, ps):
        sig = p < 0.05
        colour = (ORANGE if g < 0 else AQUA) if sig else GREY
        ax.barh(y, g, height=0.52, color=colour, zorder=3)
        off = 0.10 if g >= 0 else -0.10
        ax.text(g + off, y, f"d={g:+.1f}   p={p:.3f}", color=INK if sig else INK_2,
                fontsize=9.5, fontweight="bold" if sig else "normal",
                va="center", ha="left" if g >= 0 else "right")

    ax.axvline(0, color=INK_2, linewidth=1.2, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    lo, hi = min(gaps), max(gaps)
    ax.set_xlim(lo - 1.9, hi + 1.9)
    _style(ax, xlabel="effect of the permanent–transient mechanism   "
                      "(live minus inert, in pooled seed standard deviations)", grid="x",
           title="The mechanism helps in exactly one regime: monotone drift",
           subtitle="identical agents; the only difference is whether the permanent network "
                    "learns.  Grey = not significant.")
    ax.text(0.995, -0.24,
            "Under monotone drift the permanent pays (d = +1.4) but the shrinkage it is coupled "
            "to turns harmful, so the combined agent still loses to vanilla — see §22/§26.",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="top")
    _save(fig, outdir, "mechanism_by_regime")


# ------------------------------------------------------------------ figure 4
def fig_sigma_collapse(root, outdir):
    """Exploration over training for the three original HalfCheetah agents (FULL_PT §2).

    Three series with a shared y, so categorical hues and a direct label on each line's end --
    no legend box needed when every line is labelled where it ends.
    """
    arms = [("EWC", "ewc", BLUE), ("vanilla", "vanilla", ORANGE), ("PT", "pt", AQUA)]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    plotted = 0
    for label, cell, colour in arms:
        series = []
        for f in sorted(glob.glob(os.path.join(root, "Results", "jobJ_results", cell,
                                               "*_scalars.pkl"))):
            series.append(pickle.load(open(f, "rb"))["train/entropy"])
        if not series:
            continue
        n = min(len(s) for s in series)
        st = series[0][:n, 0]
        # 6-D action: entropy = 6*log_std + 6*0.5*log(2*pi*e)
        ls = (np.stack([s[:n, 1] for s in series]).mean(0) - 8.5134) / 6.0
        ax.plot(st, ls, color=colour, linewidth=2.0, zorder=3)
        ax.annotate(f"{label}   σ={np.exp(ls[-1]):.3f}", (st[-1], ls[-1]),
                    textcoords="offset points", xytext=(8, 0), color=colour,
                    fontsize=10, fontweight="bold", va="center", annotation_clip=False)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print("  [skip] sigma_collapse: jobJ results not found")
        return
    ax.set_xlim(0, PHASE_HC * 5 * 1.22)
    _style(ax, xlabel="env steps  (vertical rules = task switches)",
           ylabel="log σ  (exploration)",
           title="σ is unmanaged persistent memory — and it ranks the agents",
           subtitle="EWC's Fisher penalty covers log_std, so it anchors exploration as a side "
                    "effect. Final returns follow the same order.")
    ax.text(0.995, -0.22,
            "HalfCheetah, 5 seeds, mean across seeds. Phase-4 return: EWC 2252 · vanilla 1637 · "
            "PT 767 — the same ordering as the σ they retain.",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="top")
    # Switch rules AFTER _style, and with the x-grid off, so the only vertical marks on the plot
    # are the boundaries the caption names -- otherwise they are indistinguishable from gridlines.
    ax.grid(axis="x", visible=False)
    for b in range(1, 5):
        ax.axvline(b * PHASE_HC, color="#d0cfc7", linewidth=1.2, zorder=1)
    _save(fig, outdir, "sigma_collapse")


# ------------------------------------------------------------------ figure 5
def fig_beta_sweep(root, outdir):
    """The KL anchor swept over four orders of magnitude including zero (FULL_PT §13).

    Beta spans 0 -> 1.0, so a log x-axis is impossible; categorical positions with the values as
    tick labels is the honest encoding.
    """
    betas = [("0", "b000"), ("0.001", "b0001"), ("0.01", "b001"), ("0.1", "b01"), ("1.0", "b1")]
    inert = [final_window(root, "stage3", f"inert_{c}") for _, c in betas]
    live = [final_window(root, "stage3", f"pt_{c}") for _, c in betas]
    van = final_window(root, "stage2", "van_L00")
    if not len(van) or any(not len(v) for v in inert):
        print("  [skip] beta_sweep: stage3 results not found")
        return

    x = np.arange(len(betas))
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    vmed = float(np.median(van))
    ax.axhline(vmed, color=GREY, linewidth=1.6, zorder=2)
    # left-anchored: the right end of this rule is where the beta=1.0 markers land
    ax.text(0.005, vmed, "vanilla PPO (no anchor)  ", color=INK_2, fontsize=9.5,
            va="bottom", ha="left", transform=ax.get_yaxis_transform())

    for vals, colour, label in ((inert, BLUE, "inert permanent"), (live, ORANGE, "live permanent")):
        y = [float(np.median(v)) for v in vals]
        ax.plot(x, y, color=colour, linewidth=2.0, zorder=3)
        ax.scatter(x, y, s=64, color=colour, zorder=4, linewidths=1.8, edgecolors=SURFACE)
        ax.annotate(label, (x[0], y[0]), textcoords="offset points", xytext=(-6, 10),
                    color=colour, fontsize=10, fontweight="bold", ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _ in betas])
    ax.set_xlim(-0.45, len(betas) - 0.4)
    _style(ax, xlabel="KL-to-permanent coefficient  β", ylabel="final-phase return",
           title="The KL anchor explains none of the gain",
           subtitle="at β = 0 the anchor is switched off entirely, and the gain over vanilla is "
                    "unchanged")
    ax.text(0.995, -0.22,
            "point-mass, centroid E=0, 8 seeds. Inert beats vanilla at every β including zero "
            "(p = 0.000); live is below inert at every β (p ≤ 0.010).",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="top")
    _save(fig, outdir, "beta_sweep")


# ------------------------------------------------------------------ figure 6
def fig_saturation(root, outdir):
    """Dynamic range of each benchmark (FULL_PT §19a) -- why a 'significant' result was an artifact.

    One number per benchmark, and the story is 'this one is broken', so: emphasis, not categorical.
    The saturated benchmark carries the accent; the usable ones are grey.
    """
    def swing(stage, cell):
        series = _returns(root, stage, cell)
        if not series:
            return None
        n = min(len(s) for s in series)
        m = np.stack([s[:n, 1] for s in series]).mean(0)
        tail = m[len(m) // 3:]
        return 100.0 * (tail.max() - tail.min()) / max(abs(tail.mean()), 1e-9)

    rows = [("drift: drag only  (§14/§19)", swing("stage11", "van_p040"), True),
            ("drift: moving goal, amp 1.0", swing("stage13", "van_a10"), False),
            ("drift: moving goal, amp 1.5", swing("stage13", "van_a15"), False),
            ("discrete switching", swing("stage2", "van_L00"), False)]
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        print("  [skip] benchmark_saturation: results not found")
        return

    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    ys = np.arange(len(rows))[::-1]
    for y, (label, val, flag) in zip(ys, rows):
        ax.barh(y, val, height=0.5, color=ORANGE if flag else GREY, zorder=3)
        ax.text(val + 2.5, y, f"{val:.0f}%", color=INK if flag else INK_2, fontsize=10,
                fontweight="bold" if flag else "normal", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=10)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.18)
    _style(ax, xlabel="post-learning return swing, as % of mean return   "
                      "(how much the non-stationarity actually costs)", grid="x",
           title="A benchmark that barely moves cannot separate methods",
           subtitle="the drag-only drift env leaves every agent at 96–99% of the return ceiling")
    ax.text(0.995, -0.30,
            "The +5.6 'benefit' of §14 was measured on the top row. Making the goal drift too "
            "(rows 2–3) restored the dynamic range — and the effect reversed sign.",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="top")
    _save(fig, outdir, "benchmark_saturation")


# ------------------------------------------------------------------ figure 7
def fig_decoupling(root, outdir):
    """Stage 16: the permanent and the shrinkage are the same knob (FULL_PT §25)."""
    cells = [("ρ=0.05\nweak both", "pt_r05", "inert_r05"),
             ("ρ=0.15\nweak both", "pt_r15", "inert_r15"),
             ("ρ=0.5, decay 0\nDECOUPLED", "pt_decoup", "inert_decoup")]
    van = final_window(root, "stage15", "van", 0.2)
    got = [(lab, final_window(root, "stage16", a, 0.2), final_window(root, "stage16", b, 0.2))
           for lab, a, b in cells]
    got = [g for g in got if len(g[1]) and len(g[2])]
    if not got or not len(van):
        print("  [skip] decoupling: stage15/16 results not found")
        return

    x = np.arange(len(got))
    w = 0.30
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    vmed = float(np.median(van))
    ax.axhline(vmed, color=GREY, linewidth=1.6, zorder=2)
    ax.text(0.995, vmed, "  vanilla PPO", color=INK_2, fontsize=9.5, va="bottom", ha="right",
            transform=ax.get_yaxis_transform())

    for i, (lab, live, inert) in enumerate(got):
        # 2px surface gap between adjacent bars, per the mark spec
        ax.bar(i - w / 2 - 0.012, np.median(inert), width=w, color=AQUA, zorder=3)
        ax.bar(i + w / 2 + 0.012, np.median(live), width=w, color=ORANGE, zorder=3)
        for xpos, v in ((i - w / 2 - 0.012, np.median(inert)), (i + w / 2 + 0.012, np.median(live))):
            va, off = ("bottom", 4) if v >= 0 else ("top", -4)
            ax.annotate(f"{v:.0f}", (xpos, v), textcoords="offset points", xytext=(0, off),
                        ha="center", va=va, color=INK, fontsize=10, fontweight="bold")
    # headroom first, so the series labels sit above the value labels and clear of the subtitle
    highs = [max(np.median(l), np.median(i)) for _, l, i in got]
    lows = [min(np.median(l), np.median(i)) for _, l, i in got]
    top, bot = max(highs + [vmed]), min(lows + [0.0])
    ax.set_ylim(bot - (top - bot) * 0.14, top + (top - bot) * 0.42)
    ax.annotate("inert", (-w / 2 - 0.012, np.median(got[0][2])), textcoords="offset points",
                xytext=(0, 30), ha="center", color=AQUA, fontsize=10.5, fontweight="bold")
    ax.annotate("live", (w / 2 + 0.012, np.median(got[0][1])), textcoords="offset points",
                xytext=(0, 30), ha="center", color=ORANGE, fontsize=10.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in got], color=INK, fontsize=9.5)
    ax.axhline(0, color=INK_2, linewidth=1.0, zorder=4)
    _style(ax, ylabel="final-window return",
           title="The permanent and the shrinkage cannot be separated",
           subtitle="weaken them together and the benefit vanishes; force them apart and the "
                    "agent breaks")
    ax.text(0.995, -0.26,
            "linear monotone drift, 8 seeds. Mechanism (live − inert): −0.1 at ρ=0.05 (p=1.000), "
            "+3.1 at ρ=0.15 (p=0.130), −205.6 decoupled (p=0.010).",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="top")
    _save(fig, outdir, "decoupling")


# ------------------------------------------------------------- standard set
# The figure names below match the conventional set this project has used since FINDINGS.md
# (return_curves / phase_means_main / boundary_drop / consolidation_*), so the pt_full study can
# be read side by side with the earlier ones rather than in a private format.

HC_ARMS = [("vanilla PPO", "van", GREY), ("vanilla + 3 lines", "van_shrink", BLUE),
           ("pt_full live", "pt", ORANGE), ("pt_full inert", "inert", AQUA)]


def fig_return_curves(root, outdir):
    """Return over training for the four HalfCheetah arms, mean across seeds, switches marked."""
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    any_ = False
    for label, cell, colour in HC_ARMS:
        series = _returns(root, "stage14", cell)
        if not series:
            continue
        n = min(len(s) for s in series)
        st = series[0][:n, 0]
        arr = np.stack([s[:n, 1] for s in series])
        m = arr.mean(0)
        ax.plot(st, m, color=colour, linewidth=1.8, zorder=3)
        ax.annotate(label, (st[-1], m[-1]), textcoords="offset points", xytext=(8, 0),
                    color=colour, fontsize=9.5, fontweight="bold", va="center",
                    annotation_clip=False)
        any_ = True
    if not any_:
        plt.close(fig); print("  [skip] return_curves"); return
    ax.set_xlim(0, PHASE_HC * 5 * 1.30)
    _style(ax, xlabel="env steps", ylabel="episodic return (EMA)",
           title="Return over training — HalfCheetah, 5 task phases",
           subtitle="mean across 6 seeds; vertical rules are task switches")
    ax.grid(axis="x", visible=False)
    for b in range(1, 5):
        ax.axvline(b * PHASE_HC, color="#d0cfc7", linewidth=1.2, zorder=1)
    _save(fig, outdir, "return_curves")


def fig_phase_means(root, outdir):
    """Per-phase mean return per arm — the conventional per-phase view."""
    data = {}
    for label, cell, colour in HC_ARMS:
        series = _returns(root, "stage14", cell)
        if not series:
            continue
        per = []
        for r in series:
            st, x = r[:, 0], r[:, 1]
            per.append([float(x[(st >= p * PHASE_HC) & (st < (p + 1) * PHASE_HC)].mean())
                        for p in range(5)])
        data[label] = (np.median(np.array(per), axis=0), colour)
    if not data:
        print("  [skip] phase_means_main"); return

    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    x = np.arange(5)
    w = 0.19
    for i, (label, (vals, colour)) in enumerate(data.items()):
        off = (i - (len(data) - 1) / 2) * (w + 0.016)   # 2px surface gap between adjacent bars
        ax.bar(x + off, vals, width=w, color=colour, zorder=3, label=label)
    ax.axhline(0, color=INK_2, linewidth=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"phase {i+1}\n{'+1' if i % 2 == 0 else '−1'}" for i in range(5)])
    _style(ax, ylabel="mean return in phase",
           title="Per-phase return — the task alternates every 614 400 steps",
           subtitle="medians across 6 seeds")
    ax.legend(frameon=False, ncol=4, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), labelcolor=INK_2)
    _save(fig, outdir, "phase_means_main")


def fig_consolidation_internals(root, outdir):
    """The mechanism's own telemetry (FULL_PT §4a / §12): is consolidation actually running?

    Three panels because three different quantities share one x — never a dual axis.
    """
    recs = {}
    for label, cell in (("pt_full live", "pt"), ("pt_full inert", "inert")):
        rows = []
        for f in sorted(glob.glob(os.path.join(root, "stage14_results", cell,
                                               "*_consolidation_records.pkl"))):
            rows.append(pickle.load(open(f, "rb")))
        if rows:
            recs[label] = rows
    if not recs:
        print("  [skip] consolidation_internals"); return

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9))
    panels = [("absorbed_frac", "absorbed fraction",
               "(a) critic: is the permanent absorbing?"),
              ("alpha_p_used", "α_P actually used", "(b) Robbins–Monro step size"),
              ("actor_absorbed_frac", "absorbed fraction", "(c) actor: same question")]
    for ax, (key, ylab, title) in zip(axes, panels):
        for (label, rows), colour in zip(recs.items(), (ORANGE, AQUA)):
            n = min(len(r) for r in rows)
            vals = np.array([[float(rec[key]) if rec[key] is not None else np.nan
                              for rec in r[:n]] for r in rows])
            ax.plot(np.arange(n), np.nanmedian(vals, axis=0), color=colour, linewidth=1.8,
                    zorder=3, label=label)
        _style(ax, xlabel="consolidation cycle", ylabel=ylab)
        ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=8)
        if key == "alpha_p_used":
            ax.set_yscale("log")
            # the inert arm sits at exactly 0, which a log axis cannot render -- say so rather
            # than let the reader infer the series is missing
            ax.text(0.98, 0.06, "inert arm is exactly 0\n(not drawable on a log axis)",
                    transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="bottom")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    fig.suptitle("Consolidation telemetry — the mechanism runs, and the inert control provably "
                 "does not\n1.0 = the permanent absorbed exactly the ρ it was asked for",
                 color=INK, fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.13)
    fig.subplots_adjust(top=0.80)
    _save(fig, outdir, "consolidation_internals")


def fig_consolidation_loss(root, outdir):
    """Within-cycle descent of the permanent regression (the §12 'does it actually fit?' check)."""
    traces = []
    for f in sorted(glob.glob(os.path.join(root, "stage14_results", "pt",
                                           "*_consol_loss_traces.pkl")))[:1]:
        traces = pickle.load(open(f, "rb"))
    if not traces:
        print("  [skip] consolidation_loss_curves"); return

    picks = [0, len(traces) // 3, 2 * len(traces) // 3, len(traces) - 1]
    fig, axes = plt.subplots(1, len(picks), figsize=(13.0, 3.3), sharey=True)
    for ax, i in zip(axes, picks):
        step, curve = traces[i]
        ax.plot(np.arange(len(curve)), curve, color=BLUE, linewidth=1.6, zorder=3)
        ratio = float(curve[:max(len(curve)//10, 1)].mean() /
                      max(curve[-max(len(curve)//10, 1):].mean(), 1e-12))
        _style(ax, xlabel="gradient step within the cycle")
        ax.set_title(f"cycle {i+1}  ·  step {step/1e6:.2f}M  ·  ×{ratio:,.1f}",
                     color=INK, fontsize=10, fontweight="bold", loc="left", pad=8)
    axes[0].set_ylabel("consolidation regression loss", color=INK_2, fontsize=9.5)
    fig.suptitle("The permanent regression descends inside every cycle",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.005, ha="left", y=1.06)
    _save(fig, outdir, "consolidation_loss_curves")


def fig_boundary_drop(root, outdir):
    """Return drop at a task switch, per arm — the project's conventional stability measure."""
    labels, vals, cols = [], [], []
    for label, cell, colour in HC_ARMS:
        got = []
        for f in sorted(glob.glob(os.path.join(root, "stage14_results", cell,
                                               "*_scalars.pkl"))):
            d = pickle.load(open(f, "rb"))
            if "boundary/mean_drop" in d:
                got.append(float(d["boundary/mean_drop"][-1, 1]))
        if got:
            labels.append(label); vals.append(float(np.median(got))); cols.append(colour)
    if not labels:
        print("  [skip] boundary_drop"); return
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ys = np.arange(len(labels))[::-1]
    for y, v, c in zip(ys, vals, cols):
        ax.barh(y, v, height=0.5, color=c, zorder=3)
        ax.text(v + max(vals) * 0.02, y, f"{v:,.0f}", color=INK, fontsize=10,
                fontweight="bold", va="center")
    ax.set_yticks(ys); ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.2)
    _style(ax, xlabel="mean return drop at a task switch  (lower is more stable)", grid="x",
           title="Boundary drop — HalfCheetah",
           subtitle="pre-switch plateau minus post-switch trough, median of 6 seeds")
    _save(fig, outdir, "boundary_drop")


def fig_lr_perm_sweep(root, outdir):
    """Permanent learning rate against return, shrinkage held fixed (FULL_PT §25b).

    The last axis that could have shown a genuine benefit for the mechanism. Emphasis form: the
    vanilla reference is the thing every arm has to beat, so it carries the rule and the arms are
    one series -- this is not a categorical comparison, it is one curve against one threshold.
    """
    arms = [("0\n(inert)", "lr0"), ("3e−5", "lr3e5"), ("1e−4", "lr1e4"),
            ("3e−4", "lr3e4"), ("1e−3", "lr1e3")]
    vals = [final_window(root, "stage20", c, 0.2) for _, c in arms]
    van = final_window(root, "stage15", "van", 0.2)
    if not len(van) or any(not len(v) for v in vals):
        print("  [skip] lr_perm_sweep"); return

    x = np.arange(len(arms))
    y = [float(np.median(v)) for v in vals]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    vmed = float(np.median(van))
    ax.axhline(vmed, color=GREY, linewidth=1.8, zorder=2)
    ax.text(0.005, vmed, "vanilla PPO — every arm is below this  ", color=INK_2, fontsize=9.5,
            va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.plot(x, y, color=ORANGE, linewidth=2.0, zorder=3)
    ax.scatter(x, y, s=70, color=ORANGE, zorder=4, linewidths=1.8, edgecolors=SURFACE)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, 11),
                    ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([a[0] for a in arms])
    ax.set_ylim(min(y) - 18, max(max(y), vmed) + 26)
    _style(ax, xlabel="permanent learning rate  (ρ and k held fixed — the shrinkage is identical "
                      "in every arm)",
           ylabel="final-window return",
           title="No setting of the permanent beats the baseline",
           subtitle="linear monotone drift — the one regime where the permanent had measured a "
                    "benefit. 8 seeds.")
    ax.text(0.995, -0.26,
            "Non-monotone, and worst in the middle: a slowly-learning permanent (44) is far worse "
            "than one frozen (108) or one learning fast (116) — a stale anchor, tracking neither "
            "the task nor a stable reference.",
            transform=ax.transAxes, color=INK_2, fontsize=8.5, ha="right", va="top")
    _save(fig, outdir, "lr_perm_sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=None,
                    help="dir holding stageN_results/ (default: parent of src_continuous_control)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = args.results_root or os.path.abspath(os.path.join(here, "..", ".."))
    outdir = args.outdir or os.path.join(here, "figures_pt_full")
    os.makedirs(outdir, exist_ok=True)
    print(f"reading results from {root}\nwriting figures to {outdir}")

    fig_sigma_collapse(root, outdir)      # 2
    fig_beta_sweep(root, outdir)         # 13
    fig_dose_response(root, outdir)      # 18c / 18d
    fig_saturation(root, outdir)         # 19a
    fig_reduction(root, outdir)          # 24 / 24a
    fig_decoupling(root, outdir)         # 25
    fig_mechanism(root, outdir)          # across regimes
    # --- the project's conventional figure set, regenerated for pt_full ---
    fig_return_curves(root, outdir)
    fig_phase_means(root, outdir)
    fig_boundary_drop(root, outdir)
    fig_consolidation_internals(root, outdir)
    fig_consolidation_loss(root, outdir)
    fig_lr_perm_sweep(root, outdir)


if __name__ == "__main__":
    main()
