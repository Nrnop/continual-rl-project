"""Regenerate the STANDARD figure set against the corrected re-run data in workspace/.

Same figures as plots/figures/ (produced by plot_compare.py for the original study), rebuilt for
the August 2026 re-investigation. plot_compare.py is left untouched — it expects a FLAT results dir
with `{agent}_ppo_seed_N_*.pkl` and is hardcoded to three agents, while workspace/ is nested per arm
and we have four (plus six diagnostic arms). This script reads the nested layout directly.

Produces, into plots/figures_reinvestigation/:
    return_curves          mean +/- 95% CI over training, task boundaries marked
    phase_means_main       per-phase mean return, four arms
    boundary_drop          relative return drop at each switch
    recovery_time          rollouts to regain 90% of the pre-switch plateau
    velocity_curves        mean x-velocity (the physical read on direction reversal)
    td_error_curves        critic loss
    asymptotic_bar         final-phase vs whole-run mean
    phase_means_ablation   the diagnostic ladder (mechanism off / capacity / init)
    consolidation_internals absorbed_frac, transient magnitude, permanent drift

NOT reproducible from workspace/ (see REINVESTIGATION.md):
    offline_curves      needs *_eval_returns.pkl; every re-run used --no-eval, because the
                        offline eval was corrupting the training RNG stream (defect #14).
                        _isolated_rng() has since fixed that, so a future run can restore it.
    consolidation_insitu panel (b)   needs consolidation_holdout_frac > 0, never set on a re-run.
    drift_comparison    the drift regimes were not re-run under the corrected code.

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
SWITCH, TOTAL, PHASES, BATCH = 614400, 3072000, 5, 2048

# Validated categorical palette, assigned by ENTITY in fixed order and never cycled.
# Yellow (slot 4) carries direct labels throughout, which is the relief rule for its
# light-surface contrast.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
INK, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"

MAIN = [("vanilla", "final2_results", "vanilla"), ("ewc", "final2_results", "ewc"),
        ("pt", "final2_results", "pt"), ("pt_inert", "final2_results", "pt_inert")]
ABLATION = [("vanilla", "final2_results", "vanilla"),
            ("pt (full mechanism)", "final2_results", "pt"),
            ("mechanism OFF", "jobC_results", "pt_theorem1"),
            ("+ capacity matched", "jobD_results", "pt_theorem1_wide"),
            ("+ V_P init = 0", "jobF_results", "theorem1_zeroperm"),
            ("mechanism ON, V_P init = 0", "jobF_results", "pt_zeroperm")]


# ------------------------------------------------------------------ loading
def load_curves(d, sub):
    """Per-seed (step, return) curves for one arm."""
    out = []
    for f in sorted(glob.glob(os.path.join(WS, d, sub, "*_returns.pkl"))):
        if "ep_returns" in f:
            continue
        with open(f, "rb") as fh:
            out.append(np.asarray(pickle.load(fh), dtype=float))
    return out


def load_aux(d, sub, suffix):
    out = []
    for f in sorted(glob.glob(os.path.join(WS, d, sub, f"*_{suffix}.pkl"))):
        with open(f, "rb") as fh:
            out.append(np.asarray(pickle.load(fh), dtype=float))
    return out


def load_scalar(d, sub, key):
    out = []
    for f in sorted(glob.glob(os.path.join(WS, d, sub, "*_scalars.pkl"))):
        with open(f, "rb") as fh:
            sc = pickle.load(fh)
        if key in sc:
            out.append(np.asarray(sc[key], dtype=float))
    return out


def phase_means(curve):
    s, r = curve[:, 0], curve[:, 1]
    return [float(np.mean(r[(s > p * SWITCH) & (s <= (p + 1) * SWITCH)])) for p in range(PHASES)]


def mean_ci(curves):
    n = min(len(c) for c in curves)
    m = np.stack([c[:n, 1] for c in curves])
    mean = m.mean(0)
    ci = 1.96 * m.std(0, ddof=1) / np.sqrt(m.shape[0]) if m.shape[0] > 1 else np.zeros_like(mean)
    return curves[0][:n, 0], mean, ci


def smooth(a, w=20):
    """Centred rolling mean with SHRINKING windows at the edges.

    A plain np.convolve(..., mode='same') divides the edge sums by the full window even though
    fewer terms contribute, so every curve appears to dive at t=0 and t=end. That artifact is
    present in the original plot_compare figures; it is not reproduced here.
    """
    if len(a) < 3:
        return a
    w = min(w, len(a))
    c = np.convolve(a, np.ones(w), mode="same")
    n = np.convolve(np.ones_like(a), np.ones(w), mode="same")   # actual terms per position
    return c / n


# ------------------------------------------------------------------ styling
def _style(ax, ygrid=True, xgrid=False):
    # Passing line properties with False still ENABLES the grid in matplotlib, so branch.
    ax.yaxis.grid(True, color=GRID, lw=0.8) if ygrid else ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8) if xgrid else ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9, length=0)


def _boundaries(ax):
    for b in range(1, PHASES):
        ax.axvline(b * SWITCH, color=MUT, ls=(0, (4, 4)), lw=1, alpha=0.55, zorder=1)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


# ------------------------------------------------------------------ figures
def fig_return_curves():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, (lab, d, s) in enumerate(MAIN):
        x, m, ci = mean_ci(load_curves(d, s))
        m, ci = smooth(m), smooth(ci)
        ax.plot(x, m, color=PAL[i], lw=2, label=lab, zorder=3)
        ax.fill_between(x, m - ci, m + ci, color=PAL[i], alpha=0.16, lw=0, zorder=2)
    _boundaries(ax)
    ax.set_xlabel("environment step", color=MUT, fontsize=10)
    ax.set_ylabel("episodic return  (mean ± 95% CI, n=10)", color=MUT, fontsize=10)
    ax.set_title("Return over training — dashed lines are task switches",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.17))
    _style(ax)
    _save(fig, "return_curves")


def fig_phase_means(arms, name, title, figsize=(8.4, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    x, w = np.arange(1, PHASES + 1), 0.8 / len(arms)
    for i, (lab, d, s) in enumerate(arms):
        pm = np.array([phase_means(c) for c in load_curves(d, s)])
        mean, sem = pm.mean(0), pm.std(0, ddof=1) / np.sqrt(len(pm))
        off = (i - (len(arms) - 1) / 2) * w
        ax.bar(x + off, mean, w * 0.88, yerr=sem, color=PAL[i], label=lab, zorder=3,
               error_kw=dict(ecolor=MUT, lw=1, capsize=2))
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xticks(x)
    ax.set_xlabel("phase   (task alternates +1 / −1)", color=MUT, fontsize=10)
    ax.set_ylabel("mean return  (± SEM)", color=MUT, fontsize=10)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUT,
              ncol=min(len(arms), 3), loc="upper center", bbox_to_anchor=(0.5, -0.17))
    _style(ax)
    _save(fig, name)


def fig_boundary_drop():
    """Relative drop at each switch: (pre − post) / |pre|, from the return curve."""
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x, w = np.arange(1, PHASES), 0.8 / len(MAIN)
    for i, (lab, d, s) in enumerate(MAIN):
        per_seed = []
        for c in load_curves(d, s):
            st, r = c[:, 0], c[:, 1]
            drops = []
            for b in range(1, PHASES):
                pre = r[(st > b * SWITCH - 5 * BATCH) & (st <= b * SWITCH)]
                post = r[(st > b * SWITCH) & (st <= b * SWITCH + 10 * BATCH)]
                if len(pre) and len(post):
                    drops.append((pre.mean() - post.min()) / (abs(pre.mean()) + 1e-8) * 100)
            per_seed.append(drops)
        a = np.array(per_seed)
        off = (i - (len(MAIN) - 1) / 2) * w
        ax.bar(x + off, a.mean(0), w * 0.88, yerr=a.std(0, ddof=1) / np.sqrt(len(a)),
               color=PAL[i], label=lab, zorder=3, error_kw=dict(ecolor=MUT, lw=1, capsize=2))
    ax.set_xticks(x); ax.set_xticklabels([f"switch {b}" for b in x])
    ax.set_ylabel("relative return drop (%)", color=MUT, fontsize=10)
    ax.set_title("Return drop at each task switch  (larger = worse adaptation)",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    _style(ax)
    _save(fig, "boundary_drop")


def fig_recovery_time():
    """Rollouts needed to regain 90% of the pre-switch plateau after each boundary."""
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x, w = np.arange(1, PHASES), 0.8 / len(MAIN)
    for i, (lab, d, s) in enumerate(MAIN):
        per_seed = []
        for c in load_curves(d, s):
            st, r = c[:, 0], c[:, 1]
            rec = []
            for b in range(1, PHASES):
                pre = r[(st > b * SWITCH - 5 * BATCH) & (st <= b * SWITCH)]
                seg = r[(st > b * SWITCH) & (st <= (b + 1) * SWITCH)]
                if not len(pre) or not len(seg):
                    continue
                tgt, hit = 0.9 * pre.mean(), np.where(seg >= 0.9 * pre.mean())[0]
                rec.append(float(hit[0]) if len(hit) else float(len(seg)))
            per_seed.append(rec)
        a = np.array(per_seed)
        off = (i - (len(MAIN) - 1) / 2) * w
        ax.bar(x + off, a.mean(0), w * 0.88, yerr=a.std(0, ddof=1) / np.sqrt(len(a)),
               color=PAL[i], label=lab, zorder=3, error_kw=dict(ecolor=MUT, lw=1, capsize=2))
    ax.set_xticks(x); ax.set_xticklabels([f"switch {b}" for b in x])
    ax.set_ylabel("PPO updates to regain 90% of plateau", color=MUT, fontsize=10)
    ax.set_title("Recovery time after each switch  (capped at the phase length)",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    _style(ax)
    _save(fig, "recovery_time")


def fig_velocity():
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, (lab, d, s) in enumerate(MAIN):
        cur = load_aux(d, s, "velocities")
        if not cur:
            continue
        x, m, _ = mean_ci(cur)
        ax.plot(x, smooth(m, 30), color=PAL[i], lw=2, label=lab, zorder=3)
    _boundaries(ax)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xlabel("environment step", color=MUT, fontsize=10)
    ax.set_ylabel("mean x-velocity", color=MUT, fontsize=10)
    ax.set_title("Physical direction reversal — the cheetah's actual velocity",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.17))
    _style(ax)
    _save(fig, "velocity_curves")


def fig_critic_loss():
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, (lab, d, s) in enumerate(MAIN):
        cur = load_scalar(d, s, "train/critic_loss")
        if not cur:
            continue
        x, m, _ = mean_ci(cur)
        ax.plot(x, smooth(m, 30), color=PAL[i], lw=2, label=lab, zorder=3)
    _boundaries(ax)
    ax.set_xlabel("environment step", color=MUT, fontsize=10)
    ax.set_ylabel("critic loss", color=MUT, fontsize=10)
    ax.set_title("Critic loss stays small for every arm — no divergence anywhere",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.17))
    _style(ax)
    _save(fig, "td_error_curves")


def fig_asymptotic():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x, w = np.arange(len(MAIN)), 0.34
    for off, sel, lab, ci in ((-w / 2, slice(4, 5), "final phase", 0), (w / 2, slice(0, 5), "whole run", 2)):
        vals, errs = [], []
        for _, d, s in MAIN:
            pm = np.array([phase_means(c) for c in load_curves(d, s)])
            v = pm[:, sel].mean(1)
            vals.append(v.mean()); errs.append(v.std(ddof=1) / np.sqrt(len(v)))
        ax.bar(x + off, vals, w * 0.9, yerr=errs, color=PAL[ci], label=lab, zorder=3,
               error_kw=dict(ecolor=MUT, lw=1, capsize=2))
        for xx, v in zip(x + off, vals):
            ax.annotate(f"{v:.0f}", (xx, v), xytext=(0, 4 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MAIN])
    ax.set_ylabel("mean return", color=MUT, fontsize=10)
    ax.set_title("Asymptotic vs whole-run performance", fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, loc="upper right")
    _style(ax)
    _save(fig, "asymptotic_bar")


def fig_consolidation_internals():
    """The mechanism's own telemetry — the diagnostics the original study lacked."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))
    pairs = [("pt", "final2_results", "pt", PAL[2]), ("pt_inert", "final2_results", "pt_inert", PAL[3])]

    ax = axes[0]
    for lab, d, s, col in pairs:
        cur = load_scalar(d, s, "train/consol/absorbed_frac")
        x, m, _ = mean_ci(cur)
        ax.plot(x, smooth(m, 40), color=col, lw=2, label=lab, zorder=3)
    ax.axhspan(0.02, 0.15, color=PAL[2], alpha=0.10, lw=0, zorder=0)
    ax.set_yscale("log"); ax.set_ylabel("absorbed fraction", color=MUT, fontsize=10)
    ax.set_title("(a) how much of the transient the\npermanent absorbs per consolidation",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT)

    ax = axes[1]
    for lab, d, s, col in pairs:
        for key, ls in (("train/consol/trans_l2_before", "-"), ("train/consol/trans_l2_after", "--")):
            cur = load_scalar(d, s, key)
            if cur:
                x, m, _ = mean_ci(cur)
                ax.plot(x, smooth(m, 40), color=col, lw=1.8, ls=ls, zorder=3,
                        label=f"{lab} {'before' if ls == '-' else 'after'} decay")
    ax.set_ylabel("‖V_trans‖₂", color=MUT, fontsize=10)
    ax.set_title("(b) transient magnitude either side\nof the decay (λ = 0.95, exact)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=MUT)

    ax = axes[2]
    for lab, d, s, col in pairs:
        cur = load_scalar(d, s, "perm/drift_from_init")
        if cur:
            x, m, _ = mean_ci(cur)
            ax.plot(x, smooth(m, 8), color=col, lw=2, label=lab, zorder=3)
    ax.set_ylabel("‖V_perm − V_perm(t=0)‖", color=MUT, fontsize=10)
    ax.set_title("(c) how far the permanent has moved\nfrom its initialisation",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT)

    for ax in axes:
        _boundaries(ax)
        ax.set_xlabel("environment step", color=MUT, fontsize=9.5)
        _style(ax)
    _save(fig, "consolidation_internals")


if __name__ == "__main__":
    fig_return_curves()
    fig_phase_means(MAIN, "phase_means_main",
                    "Per-phase mean return — the four main arms")
    fig_boundary_drop()
    fig_recovery_time()
    fig_velocity()
    fig_critic_loss()
    fig_asymptotic()
    fig_phase_means(ABLATION, "phase_means_ablation",
                    "Diagnostic ladder — each suspect removed in turn", figsize=(9.6, 4.6))
    fig_consolidation_internals()
    print("done")
