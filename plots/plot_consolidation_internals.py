"""Internals of the permanent-consolidation step, read from a run's own TensorBoard events.

Four panels. Together (a)/(c) show what the TRANSFER fails to move and (d) shows the permanent
network barely moving as a result, while (b)/(c) show what the DECAY removes — the two defects of
FINDINGS.md 6.1, measured during training rather than on probes:

  (a) THE CONSOLIDATION REGRESSION LOSS. Every consolidation runs a supervised regression fitting
      V_perm -> old_V_perm + (1-decay)*V_trans over the buffered states. Logged per consolidation as
      the loss at the FIRST gradient step, the LAST, and the mean over the cycle. The first-vs-last
      gap shows how much each individual consolidation actually converges; the trend of the last
      value shows whether the regression gets easier as training proceeds.

  (d) THE PERMANENT'S MAGNITUDE ACROSS THE REGRESSION. Mean of V_perm(s) over the same batch,
      immediately before and immediately after the consolidation regression (the decay that follows
      touches only the transient, so it cannot affect this pair). This is the quantity consolidation
      is supposed to MOVE: if the transfer worked, the permanent should visibly absorb the
      transient's value. The script also prints the relative change in ||V_perm||, which is the
      transfer expressed as a single number.

  (b) THE TRANSIENT'S MAGNITUDE ACROSS THE DECAY. Mean and L2 norm of V_trans(s) over the
      consolidation batch, measured immediately before and immediately after theta_T is decayed.
      This is the quantity the decay is supposed to shrink: with `decay = d` and a LINEAR head the
      output scales exactly by d, whereas scaling the PARAMETERS of an MLP by d does not scale its
      output by d (see FINDINGS.md 6.1), so the before/after ratio is itself a diagnostic.

Requires the metrics added for this purpose (`train/consol/*`), so it needs a run made after that
instrumentation — earlier runs do not contain these tags and will be reported as missing.

Run from the PARENT of src_continuous_control/:
    python -m src_continuous_control.plots.plot_consolidation_internals --runs-dir <dir>
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

C_FIRST, C_LAST, C_MEAN = "#e34948", "#2a78d6", "#8a8a86"
C_BEFORE, C_AFTER = "#1baf7a", "#eb6834"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SWITCH = 614400


def _scalars(run_dir, tag):
    """-> list of (steps, values), one per seed directory, for `tag`."""
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "**", "events*"), recursive=True)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            continue
        if tag not in ea.Tags()["scalars"]:
            continue
        ev = ea.Scalars(tag)
        out.append((np.array([e.step for e in ev]), np.array([e.value for e in ev])))
    return out


def _mean_curve(series):
    """Average across seeds on the shortest common prefix."""
    if not series:
        return None, None
    n = min(len(v) for _, v in series)
    return series[0][0][:n], np.mean([v[:n] for _, v in series], axis=0)


def _style(ax, title, xlabel, ylabel, xmax=None):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUT, fontsize=10)
    ax.set_ylabel(ylabel, color=MUT, fontsize=10)
    if xmax:
        for b in range(SWITCH, int(xmax) + 1, SWITCH):
            ax.axvline(b, color=GRID, ls="--", lw=0.9, zorder=1)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9, length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True,
                    help="a directory of TB event files (one subdirectory per seed)")
    ap.add_argument("--out-dir", default="src_continuous_control/plots/figures")
    ap.add_argument("--name", default="consolidation_internals")
    a = ap.parse_args()

    tags = {k: _scalars(a.runs_dir, f"train/consol/{k}") for k in
            ("loss_first", "loss_last", "loss_mean",
             "perm_mean_before", "perm_mean_after", "perm_l2_before", "perm_l2_after",
             "trans_mean_before", "trans_mean_after", "trans_l2_before", "trans_l2_after")}
    have = {k: len(v) for k, v in tags.items()}
    print("  seeds found per tag:", have)
    if not any(have.values()):
        print(f"  no train/consol/* tags in {a.runs_dir!r} — this run predates the instrumentation.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 4.3))

    # ---- (a) regression loss ----
    ax = axes[0]
    xmax = 0
    for key, col, lab in (("loss_first", C_FIRST, "first gradient step of the cycle"),
                          ("loss_mean", C_MEAN, "mean over the cycle"),
                          ("loss_last", C_LAST, "last gradient step of the cycle")):
        x, y = _mean_curve(tags[key])
        if x is None:
            continue
        ax.plot(x, y, color=col, lw=1.8, label=lab)
        xmax = max(xmax, x.max())
    ax.set_yscale("log")
    _style(ax, "(a) Consolidation regression loss\nper consolidation, averaged over seeds",
           "environment steps", "MSE loss", xmax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUT)

    # ---- (b) transient mean before/after decay ----
    ax = axes[1]
    for key, col, ls, lab in (("trans_mean_before", C_BEFORE, "-", "before decay"),
                              ("trans_mean_after", C_AFTER, "--", "after decay")):
        x, y = _mean_curve(tags[key])
        if x is None:
            continue
        ax.plot(x, y, color=col, lw=1.8, ls=ls, label=lab)
    ax.axhline(0, color=MUT, lw=1.0)
    _style(ax, "(b) Mean transient value over the batch\n" r"$\bar{V}_{trans}(s)$, before vs after decay",
           "environment steps", "mean value", xmax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUT)

    # ---- (c) transient L2 norm before/after decay ----
    ax = axes[2]
    for key, col, ls, lab in (("trans_l2_before", C_BEFORE, "-", "before decay"),
                              ("trans_l2_after", C_AFTER, "--", "after decay")):
        x, y = _mean_curve(tags[key])
        if x is None:
            continue
        ax.plot(x, y, color=col, lw=1.8, ls=ls, label=lab)
    ax.set_yscale("log")
    _style(ax, "(c) L2 norm of the transient values\n" r"$\|V_{trans}\|_2$ over the batch",
           "environment steps", "L2 norm", xmax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUT)

    # ---- (d) permanent mean before/after the consolidation regression ----
    ax = axes[3]
    for key, col, ls, lab in (("perm_mean_before", C_BEFORE, "-", "before consolidation"),
                              ("perm_mean_after", C_AFTER, "--", "after consolidation")):
        x, y = _mean_curve(tags[key])
        if x is None:
            continue
        ax.plot(x, y, color=col, lw=1.8, ls=ls, label=lab)
    ax.axhline(0, color=MUT, lw=1.0)
    _style(ax, "(d) Mean permanent value over the batch\n"
               r"$\bar{V}_{perm}(s)$, before vs after the regression",
           "environment steps", "mean value", xmax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUT)

    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(a.out_dir, f"{a.name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  wrote {p}")

    # numbers for the write-up
    for key in ("loss_first", "loss_last", "perm_mean_before", "perm_mean_after",
                "perm_l2_before", "perm_l2_after", "trans_mean_before", "trans_mean_after",
                "trans_l2_before", "trans_l2_after"):
        x, y = _mean_curve(tags[key])
        if x is None:
            continue
        print(f"  {key:20} first {y[0]:12.5g}   final {y[-1]:12.5g}   overall mean {y.mean():12.5g}")
    pb, pyb = _mean_curve(tags["perm_l2_before"])
    pa, pya = _mean_curve(tags["perm_l2_after"])
    if pb is not None and pa is not None:
        n = min(len(pyb), len(pya))
        d = np.abs(pya[:n] - pyb[:n]) / np.maximum(pyb[:n], 1e-12)
        print(f"  permanent L2 relative CHANGE across the regression: mean {d.mean()*100:.3f}%  "
              f"(how much the transfer actually moved V_perm)")
    xb, yb = _mean_curve(tags["trans_l2_before"])
    xa, ya = _mean_curve(tags["trans_l2_after"])
    if xb is not None and xa is not None:
        n = min(len(yb), len(ya))
        ratio = ya[:n] / np.maximum(yb[:n], 1e-12)
        print(f"  L2 ratio after/before: mean {ratio.mean():.4f}  "
              f"(a linear head would give exactly the configured decay)")


if __name__ == "__main__":
    main()
