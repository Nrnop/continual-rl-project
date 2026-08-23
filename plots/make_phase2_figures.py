"""The three Phase 2 figures, built straight from the per-seed result pickles.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.plots.make_phase2_figures --seeds 0 1 2 3 4

    (a) return_vs_timestep -- median return across seeds, boundaries marked with the physics
                              multiplier that starts there.
    (b) transfer           -- forward and backward transfer per agent (Lopez-Paz & Ranzato).
    (c) ablation           -- three bars: vanilla | pt with the permanent FROZEN | pt.
                              bar 1 -> bar 2 = what having a split at all buys.
                              bar 2 -> bar 3 = what the permanent actually LEARNING buys.
                              That is "the effect of having 2 components", readable without
                              explanation, which is why it is the one figure in the write-up.

Every number is read from the raw pickles and also written to `phase2_figures_data.csv`, so
nothing is transcribed by hand and every figure has a table view.

STATISTICS. Medians across seeds, with the inter-quartile range as the band — 5 seeds is thin and
a mean is not robust at that n (CLAUDE.md). Per-seed values are drawn as dots on the bar charts so
the spread is visible rather than implied.

COLOR. One fixed assignment, used identically in all three figures: pt = blue, vanilla = orange,
ewc = aqua. The frozen-permanent arm is a LIGHTER STEP OF PT'S OWN BLUE, because it is a variant
of pt rather than a fourth method. The three hues were validated for colorblind separation on the
all-pairs gate (worst CVD dE 9.2, worst normal-vision dE 24.0 on the light surface); aqua and the
light blue sit below 3:1 against the surface, so every bar carries a printed value and every
figure has its CSV — identity and magnitude are never color-alone.
"""
import argparse
import csv
import glob
import os
import pickle

import numpy as np

# --- palette (light surface; validated, see the module docstring) ---
PT_BLUE = "#2a78d6"
PT_BLUE_LIGHT = "#86b6ef"      # same hue, lighter step: pt with a component switched off
VANILLA_ORANGE = "#eb6834"
EWC_AQUA = "#1baf7a"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

AGENTS = [("pt", "PT", PT_BLUE), ("vanilla", "Vanilla PPO", VANILLA_ORANGE),
          ("ewc", "Online EWC", EWC_AQUA)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _seed_files(results_dir, agent, seeds, suffix):
    out = []
    for s in seeds:
        path = os.path.join(results_dir, f"{agent}_ppo_seed_{s}_{suffix}.pkl")
        if os.path.exists(path):
            out.append((s, path))
    return out


def load_return_curves(results_dir, agent, seeds):
    """(steps, matrix[seed, t]) truncated to the shortest seed, or (None, None)."""
    curves, steps = [], None
    for _, path in _seed_files(results_dir, agent, seeds, "returns"):
        arr = np.asarray(_load(path), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        curves.append(arr[:, 1])
        steps = arr[:, 0] if steps is None or len(arr) < len(steps) else steps
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    return steps[:n], np.stack([c[:n] for c in curves])


def load_transfer(results_dir, agent, seeds):
    """Per-seed {'bwt':…, 'fwt':…, 'transfer_matrix':…} summaries."""
    return [_load(p) for _, p in _seed_files(results_dir, agent, seeds, "transfer_matrix")]


def absorbed_fracs(results_dir, agent, seeds):
    """Every consolidation's actor absorbed_frac — the check on whether the arm did anything."""
    vals = []
    for _, path in _seed_files(results_dir, agent, seeds, "consolidation_records"):
        for rec in _load(path):
            if rec.get("actor_absorbed_frac") is not None:
                vals.append(float(rec["actor_absorbed_frac"]))
    return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def _style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def _save(fig, out_dir, name):
    fig.patch.set_facecolor(SURFACE)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        print(f"[fig] {path}")


def _median_iqr(mat):
    return (np.median(mat, axis=0), np.percentile(mat, 25, axis=0),
            np.percentile(mat, 75, axis=0))


def _seed_dots(ax, pos, values):
    """Per-seed values as dots, jittered so they never stack on each other or on the label."""
    n = len(values)
    offsets = np.linspace(-0.22, 0.22, n) if n > 1 else np.zeros(1)
    ax.scatter(pos + offsets, values, s=20, facecolor=SURFACE, edgecolor=INK_MUTED,
               linewidth=0.9, zorder=3)


def _bar_label(ax, pos, median, values, fmt="{:.0f}"):
    """Print the median just outside the BAR END.

    Anchored to the bar, never to the extreme seed dot: one outlier seed would otherwise drag the
    label to the bottom of the axes, far from the thing it labels. The dots are jittered wide
    enough that a dot sitting at the bar end does not land under the centred label.
    """
    span = max(np.ptp(ax.get_ylim()), 1e-9)
    pad = 0.03 * span
    y, va = (median + pad, "bottom") if median >= 0 else (median - pad, "top")
    ax.text(pos, y, fmt.format(median), ha="center", va=va, color=INK, fontsize=9)


def _smooth(y, window):
    if window < 2 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(window - pad - 1, y[-1])])
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# (a) return vs timestep
# ---------------------------------------------------------------------------
def figure_return_curves(args, rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    max_step = 0.0
    for agent, label, color in AGENTS:
        steps, mat = load_return_curves(args.results_dir, agent, args.seeds)
        if mat is None:
            print(f"[fig] no return curves for {agent} — skipping it in figure (a)")
            continue
        med, lo, hi = _median_iqr(mat)
        med, lo, hi = (_smooth(v, args.smooth) for v in (med, lo, hi))
        ax.plot(steps, med, color=color, linewidth=2.0, label=f"{label} (n={mat.shape[0]})",
                solid_capstyle="round")
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)
        max_step = max(max_step, float(steps[-1]))
        rows.append(["return_curve_final", agent, f"{med[-1]:.2f}", f"n_seeds={mat.shape[0]}"])

    # Boundaries, annotated with the physics that starts there — a bare dashed line does not say
    # what changed, and "what changed" is the whole experiment.
    mults = list(args.task_multipliers)
    ymin, ymax = ax.get_ylim()
    for i, bx in enumerate(range(args.switch, int(max_step), args.switch), start=1):
        ax.axvline(bx, color=AXIS, linestyle=(0, (4, 3)), linewidth=0.9)
        ax.text(bx, ymax, f" x{mults[i % len(mults)]:g}", color=INK_MUTED, fontsize=8,
                va="top", ha="left")
    ax.text(0, ymax, f" x{mults[0]:g}", color=INK_MUTED, fontsize=8, va="top", ha="left")

    # The known ceiling, when the benchmark has one. On cartpole-swingup the reward is in [0,1]
    # over exactly 1000 steps with no early termination, so 1000 is the maximum achievable return
    # BY CONSTRUCTION — and drawing it is the whole reason that environment was chosen. A curve at
    # 400 means "40% of optimal" rather than "400, is that good?". HalfCheetah has no such number,
    # so the line is opt-in rather than a default that would be a guess.
    if args.ceiling:
        ax.axhline(args.ceiling, color=INK_MUTED, linestyle=(0, (1, 2)), linewidth=1.0)
        ax.text(0, args.ceiling, f" ceiling = {args.ceiling:g} (max possible return)",
                color=INK_MUTED, fontsize=8, va="bottom", ha="left")

    _style(ax, "Environment steps", "Episodic return (median of seeds, IQR band)",
           f"{args.env_label}: return through the task sequence")
    # Below the axes, horizontally: a legend inside the plot collides with whichever corner the
    # curves happen to occupy, and which corner that is changes with the data.
    leg = ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=3, handlelength=1.6, columnspacing=2.0)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    _save(fig, args.out_dir, "a_return_vs_timestep")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) forward / backward transfer
# ---------------------------------------------------------------------------
def figure_transfer(args, rows):
    import matplotlib.pyplot as plt
    data = {}
    for agent, label, color in AGENTS:
        summaries = load_transfer(args.results_dir, agent, args.seeds)
        fwt = [s["fwt"] for s in summaries if s.get("fwt") is not None]
        bwt = [s["bwt"] for s in summaries if s.get("bwt") is not None]
        if fwt or bwt:
            data[agent] = (label, color, np.asarray(fwt), np.asarray(bwt))
    if not data:
        print("[fig] no transfer matrices found — skipping figure (b). "
              "Was transfer_eval_episodes > 0?")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    # Two panels, not two y-axes on one plot: FWT and BWT are different quantities and a dual
    # axis would invite a comparison that means nothing.
    for ax, key, idx, title in ((axes[0], "fwt", 2, "Forward transfer (FWT)"),
                                (axes[1], "bwt", 3, "Backward transfer (BWT)")):
        labels, positions = [], []
        for pos, (agent, (label, color, fwt, bwt)) in enumerate(data.items()):
            vals = (fwt, bwt)[0 if key == "fwt" else 1]
            if vals.size == 0:
                continue
            med = float(np.median(vals))
            ax.bar(pos, med, width=0.6, color=color, linewidth=0)
            _seed_dots(ax, pos, vals)
            _bar_label(ax, pos, med, vals)
            labels.append(label)
            positions.append(pos)
            rows.append([key, agent, f"{med:.2f}",
                         "per-seed " + " ".join(f"{v:.1f}" for v in vals)])
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        _style(ax, "", "Return", title)
        ax.grid(axis="x", visible=False)
    axes[0].set_xlabel("higher = better zero-shot competence on unseen physics", fontsize=8,
                       color=INK_MUTED)
    axes[1].set_xlabel("higher = less forgetting of physics already seen", fontsize=8,
                       color=INK_MUTED)
    fig.tight_layout()
    _save(fig, args.out_dir, "b_transfer")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (c) the two-component ablation
# ---------------------------------------------------------------------------
def _run_score(results_dir, agent, seeds):
    """Per-seed whole-run mean return: the area under the learning curve, not the endpoint.

    A continual agent is judged on how it does THROUGHOUT a changing sequence, and the endpoint
    only reports the last task.
    """
    _, mat = load_return_curves(results_dir, agent, seeds)
    if mat is None:
        return None
    return mat.mean(axis=1)


def figure_ablation(args, rows):
    import matplotlib.pyplot as plt
    arms = [
        ("vanilla", "Vanilla PPO\n(no split)", VANILLA_ORANGE, args.results_dir, "vanilla"),
        ("pt_frozen", "PT, permanent FROZEN\n(split, but it never learns)", PT_BLUE_LIGHT,
         args.frozen_dir, "pt"),
        ("pt", "PT\n(both components live)", PT_BLUE, args.results_dir, "pt"),
    ]
    scores, labels, colors = [], [], []
    for key, label, color, rdir, agent in arms:
        vals = _run_score(rdir, agent, args.seeds) if rdir else None
        if vals is None:
            print(f"[fig] no runs for the '{key}' arm in {rdir} — figure (c) needs all three")
            return
        scores.append(vals)
        labels.append(label)
        colors.append(color)
        rows.append(["ablation", key, f"{np.median(vals):.2f}",
                     "per-seed " + " ".join(f"{v:.1f}" for v in vals)])

    # DID THE MANIPULATION ACTUALLY FIRE? `lr_perm = 0` stops the permanent LEARNING; it does not
    # stop the decay, and a control that was not actually off has bitten this project before. The
    # frozen arm must report ~0 absorption and the live arm must not.
    frozen_abs = absorbed_fracs(args.frozen_dir, "pt", args.seeds)
    live_abs = absorbed_fracs(args.results_dir, "pt", args.seeds)
    for name, vals, want_inert in (("frozen", frozen_abs, True), ("live", live_abs, False)):
        if vals.size == 0:
            print(f"[fig] WARNING: no consolidation records for the {name} PT arm — cannot verify "
                  "the manipulation fired")
            continue
        mean_abs = float(vals.mean())
        rows.append(["actor_absorbed_frac", f"pt_{name}", f"{mean_abs:.4f}",
                     f"min={vals.min():.4f} max={vals.max():.4f}"])
        if want_inert and mean_abs > 0.01:
            print(f"[fig] WARNING: the FROZEN arm absorbed {mean_abs:.3f} of the transient — its "
                  "permanent is still learning, so bar 2 is not the control it claims to be.")
        if not want_inert and mean_abs < 0.01:
            print(f"[fig] WARNING: the LIVE arm absorbed only {mean_abs:.4f} — its permanent is "
                  "INERT, so bars 2 and 3 are the same agent and the figure is meaningless.")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    positions = np.arange(len(scores))
    medians = [float(np.median(v)) for v in scores]
    ax.bar(positions, medians, width=0.62, color=colors, linewidth=0)
    for pos, vals, med in zip(positions, scores, medians):
        _seed_dots(ax, pos, vals)
        _bar_label(ax, pos, med, vals)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    _style(ax, "", "Mean return over the whole task sequence",
           "What each half of the decomposition buys")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, args.out_dir, "c_ablation")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")

    p = argparse.ArgumentParser(description="Phase 2 figures (a), (b) and (c)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--results-dir", type=str, default="src_continuous_control/results")
    p.add_argument("--frozen-dir", type=str,
                   default="src_continuous_control/results/ablation_frozen",
                   help="results dir of the pt runs with lr_perm = 0 (figure c, bar 2)")
    p.add_argument("--switch", type=int, default=614400)
    p.add_argument("--task-multipliers", nargs="+", type=float,
                   default=[1.0, 1.6, 0.6, 1.6, 0.6])
    p.add_argument("--smooth", type=int, default=15)
    p.add_argument("--out-dir", type=str,
                   default="src_continuous_control/plots/figures_phase2")
    p.add_argument("--env-label", type=str, default="HalfCheetah with changing physics",
                   help="figure title prefix; e.g. 'cartpole-swingup with changing pole'")
    p.add_argument("--ceiling", type=float, default=None,
                   help="draw the maximum achievable return. 1000 for cartpole-swingup (reward in "
                        "[0,1] x 1000 steps, no early termination). Omit for HalfCheetah, whose "
                        "ceiling is not known a priori.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = [["quantity", "arm", "median", "detail"]]
    figure_return_curves(args, rows)
    figure_transfer(args, rows)
    figure_ablation(args, rows)

    csv_path = os.path.join(args.out_dir, "phase2_figures_data.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[fig] {csv_path}  (the table view behind every figure)")


if __name__ == "__main__":
    main()
