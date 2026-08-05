"""Figures for REINVESTIGATION.md — built from workspace/ raw per-seed results.

Four figures, one per claim:
  1. per-phase medians, the four main arms          -> the headline result
  2. mechanism contrast (absorbed_frac)             -> the mechanism demonstrably ran
  3. the diagnostic ladder                          -> where the deficit is, and is not
  4. task-sign asymmetry                            -> what the mechanism actually costs

Run from the PARENT of src_continuous_control/:
    python -m src_continuous_control.plots.make_reinvestigation_figures
"""
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WS = os.path.join(os.path.dirname(__file__), "..", "workspace")
OUT = os.path.join(os.path.dirname(__file__), "figures_reinvestigation")

# Categorical slots 1-4 of the validated default palette (blue, orange, aqua, yellow).
# Assigned by ENTITY in fixed order and never cycled; yellow carries direct labels
# throughout, which is the relief rule for its light-surface contrast.
C = {"vanilla": "#2a78d6", "ewc": "#eb6834", "pt": "#1baf7a", "pt_inert": "#eda100"}
INK, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
SWITCH, PHASES = 614400, 5


def _phase_means(curve):
    """Mean return within each of the 5 phases, from a (step, return) curve."""
    steps, rets = curve[:, 0], curve[:, 1]
    return [float(np.mean(rets[(steps > p * SWITCH) & (steps <= (p + 1) * SWITCH)]))
            for p in range(PHASES)]


def load_arm(results_dir, prefix):
    """-> (phase_means [n_seeds, 5], whole_run [n_seeds])"""
    pat = os.path.join(WS, results_dir, prefix, "*_returns.pkl")
    per_seed = []
    for f in sorted(glob.glob(pat)):
        if "ep_returns" in f:
            continue
        with open(f, "rb") as fh:
            per_seed.append(_phase_means(np.asarray(pickle.load(fh), dtype=float)))
    a = np.asarray(per_seed)
    return a, a.mean(axis=1)


def load_absorbed(results_dir, prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(WS, results_dir, prefix, "*_scalars.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if "train/consol/absorbed_frac" in d:
            out.append(float(np.median(d["train/consol/absorbed_frac"][:, 1])))
    return np.asarray(out)


def _style(ax, ygrid=True):
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9, length=0)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------- fig 1
def fig_phase_medians():
    """Median return per phase, four arms. Lines: the x-axis is ordered time."""
    arms = {k: load_arm("final2_results", k)[0] for k in C}
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(1, PHASES + 1)
    for name, a in arms.items():
        med = np.median(a, axis=0)
        ax.plot(x, med, color=C[name], lw=2, marker="o", ms=7,
                markeredgecolor=SURF, markeredgewidth=1.5, label=name, zorder=3)
        ax.annotate(f"{med[-1]:.0f}", (x[-1], med[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=MUT, fontweight="bold")
    ax.axvspan(0.6, 1.45, color=GRID, alpha=0.45, zorder=0, lw=0)
    ax.annotate("no switch yet\n(all arms at parity)", (1.0, ax.get_ylim()[1]),
                xytext=(0, -14), textcoords="offset points",
                ha="center", va="top", fontsize=8.5, color=MUT)
    ax.set_xticks(x)
    ax.set_xlabel("phase  (task alternates +1 / −1 at each boundary)", color=MUT, fontsize=10)
    ax.set_ylabel("median return  (n=10)", color=MUT, fontsize=10)
    ax.set_title("PT's deficit opens at the first task switch, never before it",
                 fontsize=11.5, color=INK, loc="left")
    # Legend below the axes: at upper-right it collided with EWC's phase-4 peak.
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_xlim(0.6, 5.55)
    _style(ax)
    _save(fig, "phase_medians")


# ---------------------------------------------------------------- fig 2
def fig_mechanism():
    """absorbed_frac per seed. Log scale: the two arms differ by >an order of magnitude."""
    pt, inert = load_absorbed("final2_results", "pt"), load_absorbed("final2_results", "pt_inert")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.axhspan(0.02, 0.15, color=C["pt"], alpha=0.10, zorder=0, lw=0)
    for i, (vals, name) in enumerate(((inert, "pt_inert"), (pt, "pt"))):
        jit = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.16
        ax.scatter(np.full(len(vals), i) + jit, vals, s=52, color=C[name],
                   edgecolor=SURF, linewidth=1.4, zorder=3, label=name)
        med = float(np.median(vals))
        # Label beside the cluster, not on it.
        ax.annotate(f"median\n{med:.4f}", (i + 0.18, med), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=9, color=INK, fontweight="bold")
    ax.annotate("target band for a permanent\nthat averages rather than tracks",
                (-0.42, 0.045), fontsize=8.5, color=MUT, ha="left")
    ax.set_yscale("log")
    ax.set_ylim(0.0025, 0.42)          # headroom so no label is clipped
    ax.set_xticks([0, 1]); ax.set_xticklabels(["permanent OFF", "permanent ON"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("fraction of the transient absorbed\nper consolidation", color=MUT, fontsize=10)
    ax.set_title("The mechanism demonstrably ran — and it changed nothing",
                 fontsize=11.5, color=INK, loc="left")
    _style(ax)
    _save(fig, "mechanism_contrast")


# ---------------------------------------------------------------- fig 3
def fig_ladder():
    """Deficit vs vanilla for each diagnostic arm, phase 1 against phases 2-5."""
    van, _ = load_arm("final2_results", "vanilla")
    arms = [("pt\n(full mechanism)", "final2_results", "pt"),
            ("mechanism OFF", "jobC_results", "pt_theorem1"),
            ("+ capacity matched", "jobD_results", "pt_theorem1_wide"),
            ("+ V_P initialised to 0", "jobF_results", "theorem1_zeroperm")]
    labels, d1, drest = [], [], []
    for lab, d, p in arms:
        a, _ = load_arm(d, p)
        labels.append(lab)
        d1.append(np.median(a[:, 0]) - np.median(van[:, 0]))
        drest.append(np.median(a[:, 1:]) - np.median(van[:, 1:]))

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    y, h = np.arange(len(labels)), 0.34
    ax.barh(y + h / 2, d1, h * 0.92, color=C["vanilla"], zorder=3, label="phase 1 (no switch)")
    ax.barh(y - h / 2, drest, h * 0.92, color=C["ewc"], zorder=3, label="phases 2–5 (post-switch)")
    for yy, v in zip(y + h / 2, d1):
        ax.annotate(f"{v:+.0f}", (v, yy), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=MUT)
    for yy, v in zip(y - h / 2, drest):
        ax.annotate(f"{v:+.0f}", (v, yy), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=INK,
                    fontweight="bold")
    ax.axvline(0, color=MUT, lw=1.2, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("median return  −  vanilla", color=MUT, fontsize=10)
    ax.set_title("Removing each suspect in turn: phase 1 stays at parity, the post-switch gap stays",
                 fontsize=11.5, color=INK, loc="left")
    # Below the axes: inside, it sat on top of the bottom row's y-tick label.
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    _save(fig, "diagnostic_ladder")


# ---------------------------------------------------------------- fig 4
def fig_task_sign():
    """Return pooled by task sign. Vanilla earns most on -1; PT loses most there."""
    rows = [("vanilla", "final2_results", "vanilla"),
            ("pt_zeroperm\n(mechanism live)", "jobF_results", "pt_zeroperm"),
            ("mechanism OFF", "jobF_results", "theorem1_zeroperm")]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x, w = np.arange(len(rows)), 0.34
    for off, phases, lab, col in ((-w / 2, [0, 2, 4], "+1 phases (1, 3, 5)", C["pt"]),
                                  (+w / 2, [1, 3], "−1 phases (2, 4)", C["pt_inert"])):
        vals = []
        for _, d, p in rows:
            a, _ = load_arm(d, p)
            vals.append(np.median(a[:, phases]))
        ax.bar(x + off, vals, w * 0.92, color=col, zorder=3, label=lab)
        for xx, v in zip(x + off, vals):
            ax.annotate(f"{v:.0f}", (xx, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylabel("pooled median return", color=MUT, fontsize=10)
    ax.set_title("Vanilla earns most running backward; the mechanism is exactly what costs PT there",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, loc="upper right")
    _style(ax)
    _save(fig, "task_sign_asymmetry")


if __name__ == "__main__":
    fig_phase_medians()
    fig_mechanism()
    fig_ladder()
    fig_task_sign()
    print("done")
