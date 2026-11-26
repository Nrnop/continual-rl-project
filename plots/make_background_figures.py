"""Schematic figures for Chapters 2 and 3 (no run data is read).

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_background_figures

    fig2_1_mdp_loop        the agent-environment loop with a time-indexed transition kernel P_t
    fig2_2_ppo_clip        PPO's clipped objective against the probability ratio
    fig3_1_research_tree   the four families of response to non-stationarity, and the gap
"""
import os

import numpy as np

from .make_phase2_figures import (
    AXIS,
    EWC_AQUA,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PT_BLUE,
    PT_BLUE_LIGHT,
    SURFACE,
    VANILLA_ORANGE,
    _style,
)


def _out_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_thesis")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, name):
    import matplotlib.pyplot as plt
    out = _out_dir()
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


def _box(ax, x0, y0, x1, y1, face, edge, lw=1.5, radius=0.02, z=2):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z))


def _arrow(ax, a, b, color=INK_SECONDARY, lw=1.6, conn="arc3,rad=0", z=4, dashed=False):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color,
        zorder=z, connectionstyle=conn, shrinkA=0, shrinkB=0,
        linestyle=(0, (4, 2.5)) if dashed else "solid"))


# Figure 2.1
def mdp_loop():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 1); ax.set_ylim(0.38, 0.88); ax.set_axis_off()

    _box(ax, 0.08, 0.56, 0.38, 0.82, "#eaf2fd", PT_BLUE)
    ax.text(0.23, 0.69, "AGENT", ha="center", va="center", color=PT_BLUE,
            fontsize=12, fontweight="bold")
    ax.text(0.23, 0.615, r"policy $\pi(a \mid s)$", ha="center", va="center",
            color=INK_SECONDARY, fontsize=9)

    _box(ax, 0.62, 0.56, 0.92, 0.82, "#fdf0ea", VANILLA_ORANGE)
    ax.text(0.77, 0.69, "ENVIRONMENT", ha="center", va="center", color=VANILLA_ORANGE,
            fontsize=12, fontweight="bold")
    ax.text(0.77, 0.615, r"transition kernel $\mathcal{P}_t$", ha="center", va="center",
            color=INK_SECONDARY, fontsize=9)

    # action out, state and reward back
    _arrow(ax, (0.385, 0.755), (0.615, 0.755), color=PT_BLUE)
    ax.text(0.5, 0.785, r"action $a_t$", ha="center", va="bottom", color=INK, fontsize=10)

    _arrow(ax, (0.615, 0.625), (0.385, 0.625), color=VANILLA_ORANGE)
    ax.text(0.5, 0.595, r"state $s_{t+1}$,  reward $r_t$", ha="center", va="top",
            color=INK, fontsize=10)

    ax.text(0.5, 0.45, r"$\mathcal{M}_t = (\mathcal{S},\, \mathcal{A},\, \mathcal{P}_t,\,"
                       r"\mathcal{R},\, \gamma)$",
            ha="center", va="center", color=INK, fontsize=13)


    fig.tight_layout(pad=0.3)
    _save(fig, "fig2_1_mdp_loop")


# Figure 2.2
def ppo_clip():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = 0.2
    r = np.linspace(0.0, 2.0, 2000)
    fig, (axp, axn) = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, A, title in ((axp, +1.0, "positive advantage  (A > 0)"),
                         (axn, -1.0, "negative advantage  (A < 0)")):
        unclipped = r * A
        clipped = np.clip(r, 1 - eps, 1 + eps) * A
        obj = np.minimum(unclipped, clipped)

        ax.plot(r, unclipped, color=AXIS, linewidth=1.3, linestyle=(0, (4, 3)),
                label="unclipped  $r_t A$")
        ax.plot(r, obj, color=PT_BLUE, linewidth=2.4, label="PPO objective")
        ax.axvline(1 - eps, color=GRID, linewidth=1.0)
        ax.axvline(1 + eps, color=GRID, linewidth=1.0)
        ax.axvline(1.0, color=AXIS, linewidth=0.9, linestyle=":")

        ax.set_xlim(0, 2)
        _style(ax, r"probability ratio $r_t(\theta)$", "objective", title)
        leg = ax.legend(loc="upper left" if A > 0 else "lower left",
                        frameon=False, fontsize=8.6)
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)


    fig.tight_layout(pad=0.6)
    _save(fig, "fig2_2_ppo_clip")


# Figure 3.1
def research_tree():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()

    _box(ax, 0.29, 0.845, 0.71, 0.945, "#eef1f4", INK_SECONDARY, lw=1.4)
    ax.text(0.5, 0.895, "Reinforcement learning under non-stationary dynamics",
            ha="center", va="center", color=INK, fontsize=11, fontweight="bold")

    # Author names rather than citation numbers, which change when the bibliography is
    # renumbered. Sections 3.4 to 3.7 carry the citations.
    fams = [
        (0.025, 0.225, "Memory-based", "replay the past\nRolnick, Goyal", EWC_AQUA, False),
        (0.260, 0.460, "Regularization", "protect the weights\nEWC, Kirkpatrick", EWC_AQUA, False),
        (0.495, 0.695, "Meta-learning", "learn to adapt fast\nAl-Shedivat, Clavera, Gupta",
         EWC_AQUA, False),
        (0.730, 0.975, "Architectural\ndecomposition", "separate the timescales\nAnand and Precup",
         PT_BLUE, True),
    ]
    for x0, x1, name, sub, colour, ours in fams:
        face = "#eaf2fd" if ours else SURFACE
        _box(ax, x0, 0.545, x1, 0.735, face, colour, lw=1.8 if ours else 1.2)
        ax.text((x0 + x1) / 2, 0.685, name, ha="center", va="center", color=colour,
                fontsize=10.5, fontweight="bold", linespacing=1.25)
        ax.text((x0 + x1) / 2, 0.595, sub, ha="center", va="center",
                color=INK_SECONDARY, fontsize=8.6, linespacing=1.35)
        # tree connector: down to a shared bus, then down into the box. Drawn as segments
        # because angle3 with angleA=-90/angleB=90 is rejected by matplotlib.
        cx = (x0 + x1) / 2
        ax.plot([0.5, 0.5], [0.845, 0.795], color=AXIS, lw=1.1, zorder=1)
        ax.plot([min(0.5, cx), max(0.5, cx)], [0.795, 0.795], color=AXIS, lw=1.1, zorder=1)
        _arrow(ax, (cx, 0.795), (cx, 0.735), color=AXIS, lw=1.1)
        # what each family needs that smooth drift does not give
        need = {"Memory-based": "past data must\nstill be valid",
                "Regularization": "a boundary at which\nto measure importance",
                "Meta-learning": "a task distribution\nto train on",
                "Architectural\ndecomposition": "nothing: the timescales\nbelong to the learner"}[name]
        ax.text((x0 + x1) / 2, 0.455, need, ha="center", va="center",
                color=PT_BLUE if ours else INK_MUTED, fontsize=8.4, linespacing=1.35,
                style="italic")
        _arrow(ax, ((x0 + x1) / 2, 0.545), ((x0 + x1) / 2, 0.505), color=AXIS, lw=0.9)

    ax.text(0.012, 0.455, "requires", ha="left", va="center", color=INK_MUTED,
            fontsize=8.4, rotation=90)

    _box(ax, 0.60, 0.085, 0.995, 0.335, "#eaf2fd", PT_BLUE, lw=1.6)
    ax.text(0.797, 0.295, "THE GAP THIS THESIS OCCUPIES", ha="center", va="center",
            color=PT_BLUE, fontsize=9.6, fontweight="bold")
    ax.text(0.797, 0.185,
            "The decomposition needs no boundary, which is what\n"
            "suits it to smooth drift, and it has been evaluated only\n"
            "on value-based agents, under discrete task changes,\n"
            "and on one family of environments at a time.",
            ha="center", va="center", color=INK_SECONDARY, fontsize=8.8, linespacing=1.5)
    _arrow(ax, (0.85, 0.42), (0.85, 0.34), color=PT_BLUE, lw=1.4)

    ax.text(0.30, 0.21,
            "Each family answers the same question,\n"
            "how to stay stable and remain able to learn,\n"
            "and each needs something the boundary-free\n"
            "setting does not provide.",
            ha="center", va="center", color=INK_MUTED, fontsize=8.8, linespacing=1.5)

    ax.text(0.5, 0.995, "The four families, and what each one needs",
            ha="center", va="top", color=INK, fontsize=11.5, fontweight="bold")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig3_1_research_tree")


def main():
    mdp_loop()
    ppo_clip()
    research_tree()


if __name__ == "__main__":
    main()
