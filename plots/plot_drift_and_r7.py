"""Two remaining thesis figures, both from the runs' own seed curves.

  drift_comparison — PT vs vanilla across the three drift regimes (slow / two-timescale / fast).
      The point of the figure is the TREND: tied under slow drift, then a widening PT deficit as
      real fast-timescale content is added — the reverse of what the decomposition predicts. EWC is
      annotated rather than plotted, because under drift it is bit-identical to vanilla (verified:
      5/5 seeds, max |diff| = 0), so a fourth line would sit exactly on vanilla's.

  r7_grid — the round-7 2x2 (transient decay x resetting the transient's optimiser state). Both
      candidate mechanisms failed here; the reset helps at decay=0.5 and HURTS at decay=0.0, the
      reverse of the stale-momentum prediction.

Run from the PARENT of src_continuous_control/:
    python -m src_continuous_control.plots.plot_drift_and_r7
"""
import argparse
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_VANILLA, C_PT = "#2a78d6", "#1baf7a"
C_A, C_B, C_C, C_D = "#2a78d6", "#eb6834", "#4a3aa7", "#e34948"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEG, NSEG = 614400, 5
_RE = re.compile(r"_seed_\d+_returns\.pkl$")


def seg_means(pattern):
    ps = [p for p in sorted(glob.glob(pattern)) if _RE.search(os.path.basename(p))]
    rows = []
    for p in ps:
        a = np.asarray(pickle.load(open(p, "rb")), dtype=float)
        s, r = a[:, 0], a[:, 1]
        rows.append([r[(s > i * SEG) & (s <= (i + 1) * SEG)].mean() for i in range(NSEG)])
    if not rows:
        return None, None, 0
    a = np.array(rows)
    n = len(rows)
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(NSEG), n


def _style(ax, title, ylabel=None):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=MUT, fontsize=10)
    ax.axhline(0, color=MUT, lw=1.0, zorder=1)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9, length=0)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  wrote {p}")
    plt.close(fig)


def drift_figure(abl, out_dir):
    regimes = [("Slow drift\n(period 1.23M)", "drift_vanilla", "drift_pt"),
               ("Two-timescale drift\n(slow trend + fast wobble)", "drift_twoscale_vanilla", "drift_twoscale_pt"),
               ("Fast drift\n(period 123k)", "drift_fast_vanilla", "drift_fast_pt")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    x = np.arange(NSEG)
    for ax, (title, vd, pd) in zip(axes, regimes):
        vm, vs, vn = seg_means(os.path.join(abl, vd, "*_returns.pkl"))
        pm, ps_, pn = seg_means(os.path.join(abl, pd, "*_returns.pkl"))
        if vm is None or pm is None:
            print(f"  [skip] {title}")
            continue
        w = 0.38
        ax.bar(x - w / 2, vm, w * 0.92, yerr=vs, color=C_VANILLA, label=f"vanilla (n={vn})",
               error_kw=dict(ecolor=MUT, lw=1.0, capsize=2), zorder=3)
        ax.bar(x + w / 2, pm, w * 0.92, yerr=ps_, color=C_PT, label=f"PT (n={pn})",
               error_kw=dict(ecolor=MUT, lw=1.0, capsize=2), zorder=3)
        # mark segments where vanilla beats PT beyond the combined SEM
        for i in range(NSEG):
            gap = pm[i] - vm[i]
            if abs(gap) > np.hypot(vs[i], ps_[i]):
                ax.annotate(f"{gap:+.0f}", (x[i], min(vm[i], pm[i])), textcoords="offset points",
                            xytext=(0, -16), ha="center", fontsize=7.5, color=INK, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i+1}" for i in range(NSEG)])
        _style(ax, title, "Mean return within segment" if ax is axes[0] else None)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=MUT, loc="upper left")
    axes[0].annotate("EWC omitted: under drift it is bit-identical to vanilla\n"
                     "(no task boundary → Fisher never computed; verified 5/5 seeds)",
                     xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5, color=MUT)
    fig.suptitle("Smooth dynamics drift: bold numbers are PT − vanilla where the gap exceeds the "
                 "combined SEM", color=INK, fontsize=12, x=0.008, ha="left", y=1.02)
    _save(fig, out_dir, "drift_comparison")


def r7_figure(abl, out_dir):
    cells = [("decay 0.0, no reset", C_A, "r7_decay00_noreset"),
             ("decay 0.0, reset optim", C_B, "r7_decay00_reset"),
             ("decay 0.5, no reset", C_C, "r7_decay05_noreset"),
             ("decay 0.5, reset optim", C_D, "r7_decay05_reset")]
    series = []
    for lab, col, d in cells:
        m, s, n = seg_means(os.path.join(abl, d, "*_returns.pkl"))
        if m is None:
            print(f"  [skip] {lab}")
            continue
        series.append((f"{lab} (n={n})", col, m, s))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(NSEG)
    w = 0.8 / len(series)
    for j, (lab, col, m, s) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * w
        ax.bar(x + off, m, w * 0.92, yerr=s, color=col, label=lab,
               error_kw=dict(ecolor=MUT, lw=1.0, capsize=2), zorder=3)
        for xi, v in zip(x + off, m):
            ax.annotate(f"{v:,.0f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -11), ha="center", fontsize=6.5, color=MUT)
    # vanilla reference — the thing none of the cells beats
    van, _, _ = seg_means("src_continuous_control/results/vanilla_ppo_seed_*_returns.pkl")
    if van is not None:
        ax.step(np.concatenate([x - 0.5, [x[-1] + 0.5]]), np.concatenate([van, [van[-1]]]),
                where="post", color=MUT, lw=1.6, ls="--", zorder=5, label="Vanilla PPO (reference)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(NSEG)])
    _style(ax, "Round 7: transient decay × resetting the transient's optimiser state\n"
               "the reset helps at decay 0.5 and hurts at decay 0.0 — the reverse of the "
               "stale-momentum prediction", "Mean return within phase")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUT, ncol=2, loc="lower left")
    _save(fig, out_dir, "r7_grid")


def main():
    ap = argparse.ArgumentParser()
    _abl = next((d for d in ("abl_results", "src_continuous_control/abl_results")
                 if os.path.isdir(d)), "abl_results")
    ap.add_argument("--abl-dir", default=_abl)
    ap.add_argument("--out-dir", default="src_continuous_control/plots/figures")
    a = ap.parse_args()
    print(f"  reading {a.abl_dir!r}")
    drift_figure(a.abl_dir, a.out_dir)
    r7_figure(a.abl_dir, a.out_dir)


if __name__ == "__main__":
    main()
