"""Figure 4.1: the PT actor-critic architecture, four networks on two timescales.

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_architecture_figure

Reads no run data. What it draws, and where it is implemented:

    actor      SplitGaussianActor   mu = mu_P(s).detach() + mu_T(s)     models/actor.py
    critic     SplitCritic          V  = V_P(s).detach()  + V_T(s)      models/critic.py
    detach     evaluate_actions(detach_perm=True), critic_loss()        agents/ppo_pt.py
    log_std    owned by the permanent set, frozen                       models/actor.py
    consolid.  every k updates: regress perm onto perm + rho*trans,
               then scale the transient's output layer by (1 - rho)     agents/ppo_pt.py:_consolidate

Box widths are the HalfCheetah pair from configs/ppo_pt.yaml; the dm_control environments use
their own widths. The PPO gradient reaches only the transient halves (dashed arrows are detached);
the permanent changes only at consolidation.
"""
import os

from .make_phase2_figures import (
    AXIS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PT_BLUE,
    PT_BLUE_LIGHT,
    SURFACE,
)

# configs/ppo_pt.yaml, HalfCheetah; actor and critic carry the same pair
PERM_HIDDEN = "[51, 51]"
TRANS_HIDDEN = "[32, 32]"
K = 10
STEPS_PER_UPDATE = 2_048

PERM_Y = (0.715, 0.830)
TRANS_Y = (0.445, 0.560)
MID_Y = 0.6375                      # the sum sits midway between the two boxes
CONS_Y = (0.200, 0.300)             # the consolidation box
COLUMNS = ((0.095, 0.535), (0.560, 1.000))


def _out_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_thesis")
    os.makedirs(d, exist_ok=True)
    return d


def _box(ax, x0, y0, x1, y1, face, edge, lw=1.4, z=2, radius=0.012):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z))


def _arrow(ax, xy_from, xy_to, color=INK_SECONDARY, lw=1.3, connection="arc3,rad=0",
           dashed=False, z=4):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, zorder=z, connectionstyle=connection,
        linestyle=(0, (4, 2.5)) if dashed else "solid", shrinkA=0, shrinkB=0))


def _column(ax, x0, x1, title, perm_label, trans_label, out_label, out_note):
    """Draw one half of the agent (permanent, transient, sum, output).

    Returns (bx0, cons_x), the x positions the consolidation arrows attach to.
    """
    s_x = x0 + 0.055
    bx0, bx1 = x0 + 0.105, x0 + 0.295
    sum_x = x0 + 0.350
    cons_x = x0 + 0.022                      # consolidation riser, outside the boxes

    _box(ax, x0, 0.395, x1, 0.885, "none", AXIS, lw=0.9, z=1, radius=0.016)
    ax.text((x0 + x1) / 2, 0.900, title, ha="center", va="bottom",
            color=INK, fontsize=11.5, fontweight="bold")

    # the two networks
    _box(ax, bx0, PERM_Y[0], bx1, PERM_Y[1], "#eaf2fd", PT_BLUE)
    ax.text((bx0 + bx1) / 2, PERM_Y[1] - 0.030, perm_label, ha="center", va="center",
            color=PT_BLUE, fontsize=11)
    ax.text((bx0 + bx1) / 2, PERM_Y[0] + 0.030,
            "permanent (Adam)", ha="center", va="center",
            color=INK_SECONDARY, fontsize=8.5, linespacing=1.35)

    _box(ax, bx0, TRANS_Y[0], bx1, TRANS_Y[1], "#f2f7fe", PT_BLUE_LIGHT)
    ax.text((bx0 + bx1) / 2, TRANS_Y[1] - 0.030, trans_label, ha="center", va="center",
            color=PT_BLUE, fontsize=11)
    ax.text((bx0 + bx1) / 2, TRANS_Y[0] + 0.030,
            "transient (Adam)", ha="center", va="center",
            color=INK_SECONDARY, fontsize=8.5, linespacing=1.35)

    # observation in
    ax.text(s_x, MID_Y, "$s$", ha="center", va="center", color=INK, fontsize=12)
    _arrow(ax, (s_x + 0.016, MID_Y + 0.012), (bx0, PERM_Y[0] + 0.040),
           connection="angle3,angleA=0,angleB=90", color=AXIS, lw=1.1)
    _arrow(ax, (s_x + 0.016, MID_Y - 0.012), (bx0, TRANS_Y[1] - 0.040),
           connection="angle3,angleA=0,angleB=-90", color=AXIS, lw=1.1)

    # into the sum; the permanent's branch is dashed because it is detached
    _arrow(ax, (bx1, PERM_Y[0] + 0.040), (sum_x - 0.019, MID_Y + 0.019),
           connection="angle3,angleA=0,angleB=-135", dashed=True, color=INK_MUTED)
    ax.text((bx1 + sum_x) / 2 + 0.004, PERM_Y[0] + 0.012, "detach", ha="center",
            va="center", color=INK_MUTED, fontsize=8.2, style="italic")
    _arrow(ax, (bx1, TRANS_Y[1] - 0.040), (sum_x - 0.019, MID_Y - 0.019),
           connection="angle3,angleA=0,angleB=135", color=PT_BLUE)

    from matplotlib.patches import Circle
    ax.add_patch(Circle((sum_x, MID_Y), 0.019, facecolor=SURFACE,
                        edgecolor=INK_SECONDARY, linewidth=1.2, zorder=5))
    ax.text(sum_x, MID_Y + 0.001, "+", ha="center", va="center", color=INK,
            fontsize=13, zorder=6)

    _arrow(ax, (sum_x + 0.019, MID_Y), (x1 - 0.008, MID_Y), color=INK_SECONDARY)
    ax.text((sum_x + x1) / 2, MID_Y + 0.024, out_label, ha="center", va="bottom",
            color=INK, fontsize=10.5)

    # consolidation coupling: down the inside of the column, up the outside, crossing no box
    _arrow(ax, (bx0 + 0.060, TRANS_Y[0]), (bx0 + 0.060, CONS_Y[1]),
           color=PT_BLUE_LIGHT, lw=1.3)
    _arrow(ax, (cons_x, CONS_Y[1]), (bx0, PERM_Y[0] + 0.040),
           color=PT_BLUE, lw=1.3,
           connection="angle,angleA=90,angleB=180,rad=10")
    return bx0, cons_x


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.17, 1)
    ax.set_axis_off()

    # timescale bands, drawn first so everything sits on top
    for y0, y1 in ((0.700, 0.845), (0.430, 0.575)):
        ax.axhspan(y0, y1, xmin=0.085, xmax=1.0, facecolor=GRID, alpha=0.45, zorder=0)
    ax.text(0.078, 0.7725, "slow", ha="right",
            va="center", color=PT_BLUE, fontsize=9, linespacing=1.45, fontweight="bold")
    ax.text(0.078, 0.5025, "fast", ha="right", va="center",
            color=PT_BLUE_LIGHT, fontsize=9, linespacing=1.45, fontweight="bold")

    # the two halves
    actor = _column(ax, *COLUMNS[0], "ACTOR",
                    r"$\mu_P(s)$", r"$\mu_T(s)$", r"$\mu = \mu_P + \mu_T$",
                    r"$a \sim \mathcal{N}(\mu,\, \sigma)$" "\n"
                    r"$\sigma$ frozen, NOT split")
    _column(ax, *COLUMNS[1], "CRITIC",
            r"$V_P(s)$", r"$V_T(s)$", r"$V = V_P + V_T$",
            "into the advantage,\nand so into the\nPPO surrogate")

    # consolidation
    _box(ax, 0.095, CONS_Y[0], 1.000, CONS_Y[1], "#eaf2fd", PT_BLUE, lw=1.2, z=1,
         radius=0.014)
    ax.text(0.5475, (CONS_Y[0] + CONS_Y[1]) / 2, "consolidation, every k updates",
            ha="center", va="center", color=PT_BLUE, fontsize=11, fontweight="bold")

    fig.tight_layout(pad=0.4)
    out = _out_dir()
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"fig4_1_architecture.{ext}")
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
