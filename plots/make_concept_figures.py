"""Five schematic figures, drawn analytically (no result files are read).

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_concept_figures

    fig2_stability_plasticity   the dilemma, as two costs that trade off   Section 2.5
    fig2_decomposition          V = V_P + V_T                              Section 2.6.1
    fig2_change_types           piecewise change against smooth drift      Section 2.4
    fig4_consolidation_timeline absorb and decay, over several cycles      Section 4.4
    fig5_design_grid            which cells of the design were run         Section 5.10
"""
import os

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
PT, VANILLA, EWC = "#2a78d6", "#eb6834", "#1baf7a"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_thesis")


def style(ax, xlabel="", ylabel="", title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=190,
                    bbox_inches="tight", facecolor=SURFACE)
    print(f"  wrote {name}.png / .pdf")
    plt.close(fig)


def box(ax, x0, y0, x1, y1, face, edge, lw=1.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.004,rounding_size=0.018",
                                linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))


# Section 2.5: stability and plasticity
def fig_stability_plasticity():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.linspace(0, 1, 400)
    forget = 2.6 * x ** 1.7                      # fast learning costs retention
    stale = 2.6 * (1 - x) ** 1.7                 # slow learning costs adaptation
    ax.plot(x, forget, color=VANILLA, linewidth=2.2, label="forgetting what was learned")
    ax.plot(x, stale, color=EWC, linewidth=2.2, label="failing to learn what is new")
    ax.plot(x, forget + stale, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)),
            label="total cost to a single set of weights")
    i = int(np.argmin(forget + stale))
    ax.scatter([x[i]], [(forget + stale)[i]], s=46, facecolor=SURFACE,
               edgecolor=MUTED, linewidth=1.6, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["slow weights", "fast weights"])
    ax.set_yticks([])
    ax.set_ylim(0, 4.6)
    style(ax, "how quickly the weights move", "cost")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper center")
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, "fig2_stability_plasticity")


# Section 2.6.1: the decomposition
def fig_decomposition():
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(SURFACE)
    box(ax, 0.02, 0.56, 0.30, 0.94, "#eaf2fd", PT)
    ax.text(0.16, 0.80, "permanent", ha="center", color=PT, fontsize=12, fontweight="bold")
    ax.text(0.16, 0.655, "slow, protected\nwhat holds across environments",
            ha="center", color=SECONDARY, fontsize=9, linespacing=1.5)
    box(ax, 0.02, 0.06, 0.30, 0.44, "#fdeee7", VANILLA)
    ax.text(0.16, 0.30, "transient", ha="center", color=VANILLA, fontsize=12, fontweight="bold")
    ax.text(0.16, 0.155, "fast, disposable\nwhat is true right now",
            ha="center", color=SECONDARY, fontsize=9, linespacing=1.5)
    ax.text(0.375, 0.50, "+", ha="center", va="center", color=INK, fontsize=22)
    for y in (0.75, 0.25):
        ax.annotate("", xy=(0.43, 0.50), xytext=(0.31, y),
                    arrowprops=dict(arrowstyle="-|>", color=AXIS, linewidth=1.4,
                                    shrinkA=0, shrinkB=2))
    box(ax, 0.44, 0.30, 0.72, 0.70, SURFACE, INK, lw=1.8)
    ax.text(0.58, 0.545, "what the agent acts on", ha="center", color=INK, fontsize=11)
    ax.text(0.58, 0.40, "the sum, and the only thing\nbehavior ever sees",
            ha="center", color=SECONDARY, fontsize=9, linespacing=1.5)
    ax.annotate("", xy=(0.80, 0.50), xytext=(0.73, 0.50),
                arrowprops=dict(arrowstyle="-|>", color=AXIS, linewidth=1.4))
    ax.text(0.90, 0.545, "action", ha="center", color=INK, fontsize=11)
    ax.text(0.90, 0.42, "Section 4.2 is about\nwhether this arrow exists",
            ha="center", color=MUTED, fontsize=8.5, style="italic", linespacing=1.5)
    save(fig, "fig2_decomposition")


# Section 2.4: two kinds of change
def fig_change_types():
    t = np.linspace(0, 3.072e6, 3000)
    piece = np.where(t < 0.6144e6, 1.0,
             np.where(t < 1.2288e6, 1.6,
              np.where(t < 1.8432e6, 0.6,
               np.where(t < 2.4576e6, 1.6, 0.6))))
    smooth = 1.0 + 0.5 * np.sin(2 * np.pi * t / 1.2288e6)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4), sharey=True)
    for ax, y, title, marks in ((axes[0], piece, "piecewise change at observable boundaries", True),
                                (axes[1], smooth, "smooth drift, no boundary anywhere", False)):
        ax.plot(t / 1e6, y, color=PT, linewidth=2.0)
        if marks:
            for b in range(1, 5):
                ax.axvline(b * 0.6144, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_ylim(0.35, 1.75)
        style(ax, "million env steps", "physics multiplier" if marks else "", title)
    save(fig, "fig2_change_types")


# Section 4.4: the consolidation operator
def fig_consolidation_timeline():
    k, rho, cycles = 10, 0.5, 6
    steps = np.arange(0, k * cycles + 1)
    perm, trans = [], []
    p, tr = 0.0, 0.0
    for s in steps:
        if s and s % k == 0:
            p = p + rho * tr
            tr = (1 - rho) * tr
        tr += 0.22                                  # the transient learns every update
        perm.append(p); trans.append(tr)
    perm, trans = np.array(perm), np.array(trans)
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.plot(steps, perm, color=PT, linewidth=2.2, label="permanent")
    ax.plot(steps, trans, color=VANILLA, linewidth=2.2, label="transient")
    ax.plot(steps, perm + trans, color=INK, linewidth=1.3, linestyle=(0, (4, 3)),
            label="the sum, which behavior sees")
    for c in range(1, cycles + 1):
        ax.axvline(c * k, color=AXIS, linewidth=0.8, linestyle=(0, (2, 3)))
    style(ax, "PPO updates", "contribution to the estimate")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, "fig4_consolidation_timeline")


# Section 5.10: the design as a grid
def fig_design_grid():
    envs = ["HalfCheetah", "cartpole-swingup", "reacher-easy", "ball_in_cup-catch",
            "cheetah-run", "walker-stand", "walker-walk"]
    cols = ["piecewise", "slow drift", "slow + fast", "switching rate", "stationary"]
    # From Tables 5.8 and 6.18. 1 = run, 0 = not run.
    run = {
        "HalfCheetah":       [1, 1, 1, 1, 0],
        "cartpole-swingup":  [1, 1, 1, 1, 1],
        "reacher-easy":      [1, 1, 1, 0, 0],
        "ball_in_cup-catch": [1, 0, 0, 0, 0],
        "cheetah-run":       [1, 1, 1, 0, 1],
        "walker-stand":      [1, 1, 1, 0, 0],
        "walker-walk":       [1, 1, 1, 0, 0],
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(-0.5, len(envs) - 0.5)
    ax.set_facecolor(SURFACE)
    for r, e in enumerate(envs):
        for c in range(len(cols)):
            on = run[e][c]
            ax.add_patch(Rectangle((c - 0.44, r - 0.40), 0.88, 0.80,
                                   facecolor="#eaf2fd" if on else SURFACE,
                                   edgecolor=PT if on else AXIS,
                                   linewidth=1.4 if on else 0.8, zorder=2))
            if not on:
                ax.text(c, r, "not run", ha="center", va="center",
                        color=MUTED, fontsize=8, style="italic", zorder=3)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(envs)))
    ax.set_xticklabels(cols, fontsize=9.5, color=SECONDARY)
    ax.set_yticklabels(envs, fontsize=9.5, color=SECONDARY)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig5_design_grid")


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_stability_plasticity, fig_decomposition, fig_change_types,
               fig_consolidation_timeline, fig_design_grid):
        print(f"- {fn.__name__}")
        fn()


if __name__ == "__main__":
    main()
