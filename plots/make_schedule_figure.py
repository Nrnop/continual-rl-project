"""Figure 5.1: the three non-stationarity schedules against the consolidation grid.

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_schedule_figure

Reads no run data; every curve is the analytic schedule. Quantities drawn:

    1 PPO update   = n_steps 256 x num_envs 8      = 2,048 env steps
    1 run          = 3,072,000 env steps           = 1,500 updates
    piecewise      = 5 tasks of 614,400 steps      = 300 updates per task, 4 boundaries
    Lipschitz1     = 1 + 0.5*sin(2*pi*t/1,228,800) = 600 updates per cycle, 2.5 cycles
    Lipschitz2     = 1 + 0.4*sin(2*pi*t/1,228,800) + 0.2*sin(2*pi*t/30,720), 15 updates per
                     ripple, 100 ripples
    consolidation  = every k=10 updates            = every 20,480 env steps, 150 per run

The lower panel shows that consolidation (about 1.5 per ripple) is coarser than the fast ripple.
"""
import os

import numpy as np

from .make_phase2_figures import (
    AXIS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PT_BLUE,
    SURFACE,
    VANILLA_ORANGE,
    EWC_AQUA,
    _style,
)

TOTAL_STEPS = 3_072_000
STEPS_PER_UPDATE = 2_048
SWITCH = 614_400
MULTIPLIERS = [1.0, 1.6, 0.6, 1.6, 0.6]
SLOW_PERIOD = 1_228_800
FAST_PERIOD = 30_720
K = 10
CONSOLIDATION_EVERY = K * STEPS_PER_UPDATE          # 20,480 env steps


def piecewise(t):
    idx = np.minimum((t // SWITCH).astype(int), len(MULTIPLIERS) - 1)
    return np.array(MULTIPLIERS)[idx]


def lipschitz1(t):
    return 1.0 + 0.5 * np.sin(2 * np.pi * t / SLOW_PERIOD)


def lipschitz2(t):
    return (1.0 + 0.4 * np.sin(2 * np.pi * t / SLOW_PERIOD)
            + 0.2 * np.sin(2 * np.pi * t / FAST_PERIOD))


def _out_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_thesis")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.0, 6.2), gridspec_kw={"height_ratios": [2.0, 1.0]})
    fig.patch.set_facecolor(SURFACE)

    # top: the whole run
    t = np.linspace(0, TOTAL_STEPS, 12_000)
    ax_top.step(t / 1e6, piecewise(t), where="post", color=VANILLA_ORANGE,
                linewidth=2.0, label="piecewise (5 tasks, 4 boundaries)")
    ax_top.plot(t / 1e6, lipschitz1(t), color="#7a3fa0", linewidth=1.8,
                label="slow drift")
    ax_top.plot(t / 1e6, lipschitz2(t), color=PT_BLUE, linewidth=0.9, alpha=0.95,
                label="slow + fast drift")

    for b in range(1, len(MULTIPLIERS)):
        ax_top.axvline(b * SWITCH / 1e6, color=AXIS, linewidth=0.8, linestyle=":")

    _style(ax_top, "environment steps (millions)", "physics multiplier", None)
    ax_top.set_xlim(0, TOTAL_STEPS / 1e6)
    ax_top.set_ylim(0.3, 2.05)
    # legend above the curves; the piecewise line reaches 1.6
    leg = ax_top.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3,
                        frameon=False, fontsize=9, handlelength=1.6,
                        columnspacing=1.6)
    for txt in leg.get_texts():
        txt.set_color(INK_SECONDARY)


    # bottom: a 120k-step window, fast ripple against the consolidation grid
    window = 122_880                      # 60 updates = 4 fast ripples = 6 consolidations
    t2 = np.linspace(0, window, 4_000)
    ax_bot.plot(t2 / 1e3, lipschitz2(t2), color=PT_BLUE, linewidth=1.6,
                label="slow + fast drift")

    ticks = np.arange(0, window + 1, CONSOLIDATION_EVERY)
    span = lipschitz2(t2)
    lo = float(span.min()) - 0.12 * float(np.ptp(span))
    hi = float(span.max()) + 0.26 * float(np.ptp(span))
    # faint full-height guides, plus solid ticks in a band at the foot
    ax_bot.vlines(ticks / 1e3, lo, hi, color=AXIS, linewidth=0.8, alpha=0.6,
                  linestyle=":")
    band = lo + 0.045 * (hi - lo)
    ax_bot.vlines(ticks / 1e3, lo, band, color=INK_MUTED, linewidth=2.2)
    ax_bot.plot([], [], color=INK_MUTED, linewidth=2.2,
                label="consolidation (every 10 updates)")

    _style(ax_bot, "environment steps (thousands)", "multiplier", None)
    ax_bot.set_xlim(0, window / 1e3)
    ax_bot.set_ylim(lo, hi)
    leg2 = ax_bot.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2,
                         frameon=False, fontsize=9, handlelength=1.6,
                         columnspacing=1.6)
    for txt in leg2.get_texts():
        txt.set_color(INK_SECONDARY)


    fig.tight_layout()
    out = _out_dir()
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"fig5_1_schedules.{ext}")
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
