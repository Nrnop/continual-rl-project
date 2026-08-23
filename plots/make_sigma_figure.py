"""The sigma sweep as one figure: where PT's advantage over vanilla lives.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.plots.make_sigma_figure

Left panel: run-average return against exploration width, one line per agent, IQR shaded.
Right panel: PT minus vanilla, which is the actual claim -- positive only in a middle band.

CAVEAT DRAWN INTO THE FIGURE: the sigma = 1.0 point comes from the stage12 study, which has NO
boundary reset, while 0.20 / 0.37 / 0.55 all do. The reset was measured to be worth less than noise
(PT +93, p = 1.00), so the point is comparable enough to plot -- but it is not the identical setup,
so it is drawn hollow with a dashed connector rather than silently blended into the line.
"""
import glob
import os
import pickle

import numpy as np

VAN, PT, EWC = "#eb6834", "#2a78d6", "#1baf7a"
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#14161a", "#79828d", "#e3e6e9", "#c8ced4"

P2 = "src_continuous_control/results"

# sigma -> {agent: glob}. 1.0 is the no-reset variant; flagged in the figure.
POINTS = [
    (0.20, False, {"Vanilla PPO": f"{P2}/sigma_sweep/s020_van/*_returns.pkl",
                   "PT": f"{P2}/sigma_sweep/s020_pt/*_returns.pkl",
                   "Online EWC": f"{P2}/sigma_sweep/s020_ewc/*_returns.pkl"}),
    (0.37, False, {"Vanilla PPO": f"{P2}/s14reset/van/*_returns.pkl",
                   "PT": f"{P2}/s14reset/pt/*_returns.pkl",
                   "Online EWC": f"{P2}/s14reset/ewc/*_returns.pkl"}),
    (0.55, False, {"Vanilla PPO": f"{P2}/sigma_sweep/s055_van/*_returns.pkl",
                   "PT": f"{P2}/sigma_sweep/s055_pt/*_returns.pkl",
                   "Online EWC": f"{P2}/sigma_sweep/s055_ewc/*_returns.pkl"}),
    (1.00, True,  {"Vanilla PPO": "stage12_results/van/*_returns.pkl",
                   "PT": "stage12_results/pt/*_returns.pkl",
                   "Online EWC": f"{P2}/ewc_flip_s10/*_returns.pkl"}),
]
COLOUR = {"Vanilla PPO": VAN, "PT": PT, "Online EWC": EWC}


def seed_means(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        if "ep_returns" in path or "eval_returns" in path:
            continue
        a = np.asarray(pickle.load(open(path, "rb")), dtype=float)
        if a.ndim == 2 and a[-1, 0] >= 3_000_000:
            out.append(a[:, 1].mean())
    return np.array(out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = ["Vanilla PPO", "PT", "Online EWC"]
    data = {a: [] for a in agents}          # (sigma, median, q25, q75, is_noreset)
    for sig, noreset, arms in POINTS:
        for a in agents:
            v = seed_means(arms[a])
            if len(v):
                data[a].append((sig, np.median(v), np.percentile(v, 25),
                                np.percentile(v, 75), noreset))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4))

    for a in agents:
        pts = data[a]
        x = [p[0] for p in pts]; y = [p[1] for p in pts]
        lo = [p[2] for p in pts]; hi = [p[3] for p in pts]
        ax.plot(x[:3], y[:3], color=COLOUR[a], lw=2, marker="o", ms=6, label=a, zorder=3)
        ax.plot(x[2:], y[2:], color=COLOUR[a], lw=2, ls=(0, (4, 3)), zorder=3)
        ax.plot(x[3:], y[3:], color=COLOUR[a], marker="o", ms=6, mfc=SURFACE, mew=2, zorder=3)
        ax.fill_between(x, lo, hi, color=COLOUR[a], alpha=0.13, lw=0)

    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.20, 0.37, 0.55, 1.00])
    ax.set_xticklabels(["0.20", "0.37", "0.55", "1.00"])
    ax.set_xlabel("Exploration width σ  (log scale)", color=INK, fontsize=10)
    ax.set_ylabel("Return (median across seeds, IQR)", color=INK, fontsize=10)
    ax.set_title("Every agent degrades with noise — but not in the same shape",
                 color=INK, fontsize=11, loc="left", pad=8)

    # PT minus vanilla: the actual claim.
    xs = [p[0] for p in data["PT"]]
    diff = [p[1] - q[1] for p, q in zip(data["PT"], data["Vanilla PPO"])]
    bx.axhspan(0, max(diff) * 1.35, color=PT, alpha=0.05, lw=0)
    bx.plot(xs[:3], diff[:3], color=PT, lw=2, marker="o", ms=6, zorder=3)
    bx.plot(xs[2:], diff[2:], color=PT, lw=2, ls=(0, (4, 3)), zorder=3)
    bx.plot(xs[3:], diff[3:], color=PT, marker="o", ms=6, mfc=SURFACE, mew=2, zorder=3)
    bx.axhline(0, color=INK, lw=1.1)
    # Offsets are per-point: the two negative values sit at opposite ends of the axis, so a single
    # rule for "below the marker" pushes the leftmost label into the tick labels.
    for x, d, sig, (ox, oy), ha in zip(
            xs, diff, ["p=0.008", "p=0.065", "p=0.421", "p=0.026"],
            [(10, 8), (0, 14), (0, 14), (-10, 6)], ["left", "center", "center", "right"]):
        bx.annotate(f"{d:+.0f}  {sig}", (x, d), textcoords="offset points",
                    xytext=(ox, oy), ha=ha, fontsize=8.5, color=MUTED)
    bx.set_xscale("log")
    bx.set_xticks([0.20, 0.37, 0.55, 1.00])
    bx.set_xticklabels(["0.20", "0.37", "0.55", "1.00"])
    bx.set_xlabel("Exploration width σ  (log scale)", color=INK, fontsize=10)
    bx.set_ylabel("PT − vanilla", color=INK, fontsize=10)
    bx.set_title("PT beats vanilla only in a band — and loses at both ends",
                 color=INK, fontsize=11, loc="left", pad=8)
    bx.text(0.37, max(diff) * 1.18, "PT ahead", color=PT, fontsize=9, ha="center")

    for c in (ax, bx):
        # A log axis draws its own minor labels (3x10^-1, 4x10^-1 ...) straight through the four
        # sigma labels we actually want. Silence them.
        c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        c.set_xlim(0.17, 1.18)
        c.set_facecolor(SURFACE)
        c.grid(True, color=GRID, lw=0.6)
        c.set_axisbelow(True)
        for s in ("top", "right"):
            c.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            c.spines[s].set_color(AXIS)
        c.tick_params(colors=MUTED, labelsize=9)

    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.5, -0.04, "Reward flips + reset at every boundary. Hollow marker and dashed segment: "
             "σ = 1.0 comes from the no-reset study (the reset was worth less than noise, p = 1.00).",
             ha="center", fontsize=8.5, color=MUTED)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()

    out_dir = "src_continuous_control/plots/figures_phase2"
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"sigma_curve.{ext}"), dpi=190,
                    bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] {out_dir}/sigma_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
