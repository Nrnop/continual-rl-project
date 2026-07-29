"""Thesis figures: per-phase comparison of the main agents and of the PT ablation variants.

Complements plots/plot_compare.py (which draws the time-series curves for the three main agents).
This script draws the *per-phase mean* summaries — the primary metric used in FINDINGS.md, because
per-phase means are far more stable than single end-of-phase points at n=5 seeds.

Run from the PARENT of src_continuous_control/:

    python -m src_continuous_control.plots.plot_thesis_figures

Reads:
    results/{vanilla,ewc,pt}_ppo_seed_*_returns.pkl        (main sweep)
    abl_results/<variant>/pt_ppo_seed_*_returns.pkl        (ablation variants)
Writes:
    plots/figures/phase_means_main.{png,pdf}
    plots/figures/phase_means_ablation.{png,pdf}

Colour policy: one fixed entity -> colour map, used in every figure so a series never changes
colour between plots. Palette validated for CVD (light surface, all-pairs); the aqua/red pair sits
in the 6-8 dEta band, which is legal only with secondary encoding -> every bar carries a direct
value label, which also satisfies the sub-3:1 contrast note for aqua.
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

# --- fixed entity -> colour map (never reassign; a colour follows the entity, not its rank) ---
C_VANILLA = "#2a78d6"   # blue
C_EWC     = "#eb6834"   # orange
C_PT      = "#1baf7a"   # aqua    - PT as implemented (broken consolidation)
C_NOCON   = "#4a3aa7"   # violet  - PT, consolidation disabled
C_SHARED  = "#e34948"   # red     - PT, shared trunk + linear heads (exact consolidation)
C_TRAINED = "#eda100"   # yellow  - PT, separate trunks + a properly trained consolidation regression
INK       = "#0b0b0b"
INK_MUTED = "#52514e"
GRIDC     = "#d9d8d4"

SWITCH = 614400
N_PHASES = 5
PHASE_LABELS = ["P1\n(fwd)", "P2\n(bwd)", "P3\n(fwd)", "P4\n(bwd)", "P5\n(fwd)"]


def _load_curve(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:                       # legacy 1-D format
        return np.arange(1, len(arr) + 1) * 2048.0, arr
    return arr[:, 0], arr[:, 1]


# A seed's training curve is EXACTLY '<agent>_ppo_seed_<N>_returns.pkl'. The sibling files
# '..._ep_returns.pkl', '..._eval_returns.pkl' and '..._velocities.pkl' must NOT be picked up:
# a bare 'seed_*_returns.pkl' glob also matches them (the '*' swallows '0_ep' / '0_eval'), which
# silently averages three different curve types together and triples the apparent seed count.
_SEED_CURVE_RE = re.compile(r"_seed_\d+_returns\.pkl$")


def _seed_curves(pattern):
    return [p for p in sorted(glob.glob(pattern)) if _SEED_CURVE_RE.search(os.path.basename(p))]


def phase_means(pattern):
    """-> (means[5], sems[5], n_seeds). Mean return within each phase, averaged over seeds."""
    per_seed = []
    for path in _seed_curves(pattern):
        steps, rets = _load_curve(path)
        row = []
        for i in range(N_PHASES):
            lo, hi = i * SWITCH, (i + 1) * SWITCH
            m = (steps > lo) & (steps <= hi)
            row.append(rets[m].mean() if m.any() else np.nan)
        per_seed.append(row)
    if not per_seed:
        return None, None, 0
    a = np.asarray(per_seed, dtype=np.float64)
    n = a.shape[0]
    return np.nanmean(a, axis=0), np.nanstd(a, axis=0) / max(np.sqrt(n), 1), n


def _style(ax, ylabel, title):
    ax.set_title(title, color=INK, fontsize=13, pad=12, loc="left")
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)          # zero = "not moving" reference
    ax.yaxis.grid(True, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRIDC)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def _grouped_bars(ax, series, ylabel, title):
    """series: list of (label, colour, means, sems). Direct value labels on every bar."""
    x = np.arange(N_PHASES)
    n = len(series)
    w = 0.8 / n
    for j, (label, colour, means, sems) in enumerate(series):
        off = (j - (n - 1) / 2) * w
        # 2px surface gap between adjacent fills -> width*0.92
        ax.bar(x + off, means, w * 0.92, yerr=sems, label=label, color=colour,
               error_kw=dict(ecolor=INK_MUTED, lw=1.0, capsize=2), zorder=3)
        for xi, v in zip(x + off, means):
            if np.isnan(v):
                continue
            ax.annotate(f"{v:,.0f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -11), ha="center",
                        fontsize=7, color=INK_MUTED, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_LABELS)
    _style(ax, ylabel, title)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=2, loc="upper right")


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="src_continuous_control/results")
    ap.add_argument("--abl-dir", default="abl_results")
    ap.add_argument("--out-dir", default="src_continuous_control/plots/figures")
    ap.add_argument("--expect-seeds", type=int, default=5,
                    help="warn loudly if a variant does not match exactly this many seed curves")
    a = ap.parse_args()

    def check(label, n):
        if n != a.expect_seeds:
            print(f"  [WARN] {label}: matched {n} seed curves, expected {a.expect_seeds} "
                  f"-- check the glob; do NOT use this figure until resolved")

    # ---------- Figure 1: the three main agents ----------
    main_spec = [("Vanilla PPO", C_VANILLA, "vanilla"), ("Online EWC", C_EWC, "ewc"),
                 ("PT-PPO", C_PT, "pt")]
    series = []
    for label, colour, agent in main_spec:
        m, s, n = phase_means(os.path.join(a.results_dir, f"{agent}_ppo_seed_*_returns.pkl"))
        if m is None:
            print(f"  [skip] no data for {agent}")
            continue
        check(label, n)
        print(f"  {label:26} n={n}  phase means: " + " ".join(f"{v:8.1f}" for v in m))
        series.append((f"{label} (n={n})", colour, m, s))
    if series:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        _grouped_bars(ax, series, "Mean return within phase",
                      "Continual HalfCheetah: per-phase mean return (error bars = SEM over seeds)")
        _save(fig, a.out_dir, "phase_means_main")

    # ---------- Figure 2: PT ablation variants, against a vanilla reference ----------
    # Ordered by HOW MUCH CONSOLIDATION REGRESSION ACTUALLY RUNS, because that ordering is the
    # result: none -> exact-but-no-regression -> barely trained -> well trained, and performance
    # falls monotonically along it. Colours follow the entity and are fixed across all figures.
    abl_spec = [
        ("PT, consolidation off", C_NOCON,
         os.path.join(a.abl_dir, "pt_noconsol", "pt_ppo_seed_*_returns.pkl")),
        ("PT, shared trunk (exact, no regression)", C_SHARED,
         os.path.join(a.abl_dir, "pt_sharedtrunk", "pt_ppo_seed_*_returns.pkl")),
        ("PT as implemented (regression barely runs)", C_PT,
         os.path.join(a.results_dir, "pt_ppo_seed_*_returns.pkl")),
        ("PT, trained consolidation (regression fits)", C_TRAINED,
         os.path.join(a.abl_dir, "pt_trained_consol", "pt_ppo_seed_*_returns.pkl")),
    ]
    series = []
    for label, colour, pat in abl_spec:
        m, s, n = phase_means(pat)
        if m is None:
            print(f"  [skip] no data: {pat}")
            continue
        check(label, n)
        print(f"  {label:26} n={n}  phase means: " + " ".join(f"{v:8.1f}" for v in m))
        series.append((f"{label} (n={n})", colour, m, s))
    van_m, _, _ = phase_means(os.path.join(a.results_dir, "vanilla_ppo_seed_*_returns.pkl"))
    if series:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        _grouped_bars(ax, series, "Mean return within phase",
                      "PT ablations: broken -> repaired, against the vanilla baseline")
        if van_m is not None:
            # vanilla is a REFERENCE here, not a peer series -> neutral dashed step line
            xs = np.arange(N_PHASES)
            ax.step(np.concatenate([xs - 0.5, [xs[-1] + 0.5]]),
                    np.concatenate([van_m, [van_m[-1]]]), where="post",
                    color=INK_MUTED, lw=1.6, ls="--", zorder=5, label="Vanilla PPO (reference)")
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=2, loc="upper right")
        _save(fig, a.out_dir, "phase_means_ablation")

    print("done.")


if __name__ == "__main__":
    main()
