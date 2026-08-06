"""Regenerate the STANDARD figure set against the corrected re-run data in workspace/.

Same figures as plots/figures/ (produced by plot_compare.py for the original study), rebuilt for
the August 2026 re-investigation. plot_compare.py is left untouched — it expects a FLAT results dir
with `{agent}_ppo_seed_N_*.pkl` and is hardcoded to three agents, while workspace/ is nested per arm
and we have four (plus six diagnostic arms). This script reads the nested layout directly.

Produces, into plots/figures/ (which this script now OWNS — every file there is post-fix, and
the pre-fix originals are archived out of the repo; see plots/figures/PROVENANCE.md):
    return_curves          mean +/- 95% CI over training, task boundaries marked
    phase_means_main       per-phase mean return, four arms
    boundary_drop          relative return drop at each switch
    recovery_time          rollouts to regain 90% of the pre-switch plateau
    velocity_curves        mean x-velocity (the physical read on direction reversal)
    td_error_curves        critic loss
    asymptotic_bar         final-phase vs whole-run mean
    phase_means_ablation   the diagnostic ladder (mechanism off / capacity / init)
    consolidation_internals absorbed_frac, transient magnitude, permanent drift
    consolidation_prepost  regression loss + transient magnitude across the decay. Same three
                        panels as the Jul-30 consolidation_internals_{trained,shipped}, which
                        predated the alpha_P, decay_mode and theta_P-init fixes and so showed
                        the mechanism while the permanent was inert.
    consolidation_insitu   absorbed_frac on FITTED vs HELD-OUT states (Job G) — rules out the
                        reading that the transfer is memorising the ConsolidationBuffer.
    consolidation_loss_curves  one panel per consolidation cycle, x = gradient step within that
                        cycle's regression (Job I).
    offline_curves      zero-momentum evaluation from standstill (Job J) — the only return figure
                        here without the carried-momentum confound at a direction reversal.

NOT reproducible from workspace/ (see PROVENANCE.md):
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
OUT = os.path.join(os.path.dirname(__file__), "figures")
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
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    bbox_inches="tight", facecolor=SURF, **kw)
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


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


def fig_consolidation_loss_curves(d="jobI_results", sub="pt_zeroperm", seed=0, switch=SWITCH):
    """One panel per consolidation cycle; x = gradient step WITHIN that cycle's regression.

    The scalar log only keeps first/last/mean, which cannot distinguish "descends smoothly" from
    "flat" from "diverges then recovers". These are the full traces.

    Laid out rows = task phase, columns = consolidation index within that phase, so the grid reads
    as the whole run. The per-minibatch loss is noisy (each gradient step sees a different
    minibatch), so a smoothed trend is drawn over the raw trace.
    """
    path = glob.glob(os.path.join(WS, d, sub, f"*seed_{seed}_consol_loss_traces.pkl"))
    if not path:
        print(f"  SKIP consolidation_loss_curves — no traces under {d}/{sub} "
              f"(needs a run with the trace-logging code)")
        return
    with open(path[0], "rb") as fh:
        traces = pickle.load(fh)

    # Derive the grid from the data rather than assuming the production schedule, so a shortened
    # diagnostic run doesn't render four empty rows.
    # Clamp: the final consolidation fires at exactly total_steps, and total_steps // switch is
    # PHASES, not PHASES-1 — which would render a spurious extra row for the last cycle of the
    # last phase. There are five phases; the boundary belongs to the phase that just ended.
    phase_of = [min(int(st // switch), PHASES - 1) for st, _ in traces]
    phases = sorted(set(phase_of))
    per_phase = {p: [i for i, q in enumerate(phase_of) if q == p] for p in phases}
    nrow, ncol = len(phases), max(len(v) for v in per_phase.values())
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.0 * nrow),
                             sharex=True, squeeze=False)
    lo = min(float(c.min()) for _, c in traces)
    hi = max(float(np.percentile(c, 99)) for _, c in traces)

    for r, p in enumerate(phases):
        for j in range(ncol):
            ax = axes[r][j]
            if j >= len(per_phase[p]):
                ax.axis("off")
                continue
            step, curve = traces[per_phase[p][j]]
            x = np.arange(1, len(curve) + 1)
            ax.plot(x, curve, color=PAL[0], lw=0.7, alpha=0.35, zorder=2)
            ax.plot(x, smooth(curve, max(5, len(curve) // 25)), color=PAL[0], lw=1.8, zorder=3)
            drop = curve[:len(curve) // 10].mean() / max(curve[-len(curve) // 10:].mean(), 1e-9)
            ax.annotate(f"×{drop:.2f}", (0.96, 0.9), xycoords="axes fraction", ha="right",
                        fontsize=8, color=INK, fontweight="bold")
            ax.set_ylim(lo, hi)
            if j == 0:
                ax.set_ylabel(f"phase {p + 1}\nregression loss", color=MUT, fontsize=8.5)
            if p == PHASES - 1:
                ax.set_xlabel("gradient step", color=MUT, fontsize=8.5)
            _style(ax)
            ax.tick_params(labelsize=7)
    fig.suptitle("Consolidation regression, one panel per cycle — ×N is first-decile ÷ last-decile "
                 "loss (>1 = descending)", fontsize=11.5, color=INK, x=0.01, ha="left", y=1.005)
    _save(fig, "consolidation_loss_curves")


def fig_offline_curves(d="jobJ_results"):
    """Zero-momentum offline evaluation — return from a STANDSTILL, sampled through training.

    Every other return figure here is read off the training rollout, where the cheetah carries its
    momentum across a task boundary: the reward sign flips while the body is already moving at
    speed, so part of the post-switch drop is the physics of reversing a moving body rather than
    the policy being wrong. This probe restarts the episode from rest, so it isolates policy
    quality from that confound. It is the measurement the supervisors' framing asks for.

    Job J. These runs reproduce final2_results BITWISE on all 20 seeds, so these curves belong to
    the same training as every other figure in this folder — the evaluation is genuinely read-only
    (defect #14, fixed by _isolated_rng()).
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    drawn = 0
    for i, (lab, _, sub) in enumerate(MAIN):
        cur = load_aux(d, sub, "eval_returns")
        if not cur:
            continue
        x, m, ci = mean_ci(cur)
        ax.plot(x, m, color=PAL[i], lw=2, label=lab, zorder=3, marker="o", ms=3)
        ax.fill_between(x, m - ci, m + ci, color=PAL[i], alpha=0.16, lw=0, zorder=2)
        drawn += 1
    if not drawn:
        print(f"  SKIP offline_curves — no *_eval_returns.pkl under {d}")
        plt.close(fig)
        return
    _boundaries(ax)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xlabel("environment step", color=MUT, fontsize=10)
    ax.set_ylabel("offline return from standstill  (mean ± 95% CI, n=5)", color=MUT, fontsize=10)
    ax.set_title("Zero-momentum offline evaluation — the momentum confound removed",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.17))
    _style(ax)
    _save(fig, "offline_curves")


def fig_consolidation_insitu(d="jobG_results", sub="pt_holdout"):
    """Does the consolidation transfer GENERALISE, or is it memorising the buffer?

    Job G ran the production agent with consolidation_holdout_frac > 0: the regression fits on one
    part of the ConsolidationBuffer and `*_holdout` is measured on states it never trained on.

    This is the figure that rules out the obvious deflationary reading of a working absorbed_frac
    — that theta_P is only reproducing old_V_perm + V_trans on the exact states it was shown. If
    that were true the holdout curve would sit well below the fitted one. It does not.
    """
    fitted = load_scalar(d, sub, "train/consol/absorbed_frac")
    held = load_scalar(d, sub, "train/consol/absorbed_frac_holdout")
    if not fitted or not held:
        print(f"  SKIP consolidation_insitu — no holdout tags under {d}/{sub}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))

    ax = axes[0]
    for cur, lab, col, ls in ((fitted, "fitted states", PAL[2], "-"),
                              (held, "held-out states", PAL[1], "--")):
        x, m, ci = mean_ci(cur)
        m, ci = smooth(m, 8), smooth(ci, 8)
        ax.plot(x, m, color=col, lw=1.6, ls=ls, label=lab, zorder=3)
        ax.fill_between(x, m - ci, m + ci, color=col, alpha=0.16, lw=0, zorder=2)
    ax.set_ylabel("absorbed fraction", color=MUT, fontsize=9.5)
    ax.set_title("(a) how much of the transient the permanent takes up\n"
                 "on states it fitted vs states it never saw", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUT)

    # The ratio is the actual claim: 1.0 means the transfer is a function, not a lookup.
    ax = axes[1]
    n = min(min(len(a) for a in fitted), min(len(a) for a in held))
    ratio = np.stack([h[:n, 1] / np.where(np.abs(f[:n, 1]) < 1e-9, np.nan, f[:n, 1])
                      for f, h in zip(fitted, held)])
    x = fitted[0][:n, 0]
    m = np.nanmean(ratio, 0)
    ci = 1.96 * np.nanstd(ratio, 0, ddof=1) / np.sqrt(ratio.shape[0])
    ax.plot(x, smooth(m, 8), color=PAL[0], lw=1.6, zorder=3)
    ax.fill_between(x, smooth(m - ci, 8), smooth(m + ci, 8),
                    color=PAL[0], alpha=0.16, lw=0, zorder=2)
    ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.text(x[0], 1.06, "perfect generalisation", color=INK, fontsize=8.5, va="bottom")
    ax.set_ylim(0, 2)
    ax.set_ylabel("held-out ÷ fitted", color=MUT, fontsize=9.5)
    ax.set_title(f"(b) ratio of the two — mean {np.nanmean(ratio):.3f}\n"
                 "memorisation would sit well below 1", fontsize=10.5, color=INK, loc="left")

    for ax in axes:
        _boundaries(ax)
        ax.set_xlabel("environment steps", color=MUT, fontsize=9.5)
        _style(ax)
    _save(fig, "consolidation_insitu")


def fig_consolidation_prepost():
    """The July-30 three-panel diagnostic, regenerated from the POST-FIX sweep.

    The version in plots/figures/ (consolidation_internals_{trained,shipped}) predates the alpha_P
    tuning, the decay_mode fix and the theta_P init fix, so it shows the mechanism as it behaved
    while the permanent was effectively inert. Same three panels, same tags, final2_results/pt.

    The staircase is not an artifact of the mechanism: these tags are written once per
    consolidation but logged every update, so each value is held until the next cycle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))

    ax = axes[0]
    for key, lab, col, lw in (("loss_first", "first gradient step of the cycle", PAL[6], 1.0),
                              ("loss_mean", "mean over the cycle", MUT, 1.0),
                              ("loss_last", "last gradient step of the cycle", PAL[0], 1.0)):
        cur = load_scalar("final2_results", "pt", f"train/consol/{key}")
        if not cur:
            continue
        x, m, _ = mean_ci(cur)
        ax.plot(x, m, color=col, lw=lw, label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("MSE loss", color=MUT, fontsize=9.5)
    ax.set_title("(a) consolidation regression loss\nper consolidation, averaged over seeds",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUT, loc="lower left")

    for ax, stem, ylab, title, logy in (
            (axes[1], "trans_mean", "mean value",
             "(b) mean transient value over the batch\nbefore vs after decay", False),
            (axes[2], "trans_l2", "L2 norm",
             "(c) L2 norm of the transient values\nover the batch", True)):
        for when, col, ls in (("before", PAL[2], "-"), ("after", PAL[1], "--")):
            cur = load_scalar("final2_results", "pt", f"train/consol/{stem}_{when}")
            if not cur:
                continue
            x, m, _ = mean_ci(cur)
            ax.plot(x, m, color=col, lw=1.1, ls=ls, label=f"{when} decay")
        if logy:
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color=INK, lw=0.8, zorder=1)
        ax.set_ylabel(ylab, color=MUT, fontsize=9.5)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=MUT)

    for ax in axes:
        _boundaries(ax)
        ax.set_xlabel("environment steps", color=MUT, fontsize=9.5)
        _style(ax)
    _save(fig, "consolidation_prepost")


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
    fig_consolidation_prepost()
    fig_consolidation_insitu()
    fig_offline_curves()
    fig_consolidation_loss_curves()
    print("done")
