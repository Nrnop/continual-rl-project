"""In-situ consolidation quality, measured during real training (rounds 5 and 6).

Reads the TensorBoard event files written by the runs themselves — no re-simulation, no offline
proxy. Two panels:

  (a) round 5 (`pt_trained_consol`): value drift per consolidation on the states the regression
      fitted, across training, one line per seed. Shows the regression converging to a near-exact
      transfer while PT still collapses.
  (b) round 6 (`pt_consol_holdout`): the same metric alongside its HELD-OUT counterpart — 20 % of
      the buffer excluded from the regression. The two track each other, which is what falsified the
      memorisation hypothesis (FINDINGS.md §5.6).

Run from the PARENT of src_continuous_control/:
    python -m src_continuous_control.plots.plot_consolidation_insitu
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

C_FIT, C_HOLD = "#1baf7a", "#e34948"      # same entity colours as the other thesis figures
INK, MUT, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SWITCH = 614400
FIT_TAG = "train/consolidation_error_pct"
HOLD_TAG = "train/consolidation_error_holdout_pct"


def _series(run_dir, tag):
    """-> list of (steps, values), one entry per seed, for `tag`."""
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


def _style(ax, title, ylabel, xmax):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("environment steps", color=MUT, fontsize=10)
    ax.set_ylabel(ylabel, color=MUT, fontsize=10)
    for b in range(SWITCH, int(xmax) + 1, SWITCH):        # task boundaries, for orientation
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
    _abl = next((d for d in ("abl_runs", "src_continuous_control/abl_runs") if os.path.isdir(d)),
                "abl_runs")
    ap.add_argument("--abl-runs-dir", default=_abl)
    ap.add_argument("--out-dir", default="src_continuous_control/plots/figures")
    a = ap.parse_args()

    r5 = _series(os.path.join(a.abl_runs_dir, "pt_trained_consol"), FIT_TAG)
    r6f = _series(os.path.join(a.abl_runs_dir, "pt_consol_holdout"), FIT_TAG)
    r6h = _series(os.path.join(a.abl_runs_dir, "pt_consol_holdout"), HOLD_TAG)
    print(f"  round 5 fitted: {len(r5)} seeds | round 6 fitted: {len(r6f)} | held-out: {len(r6h)}")
    if not r5 and not r6f:
        print("  no consolidation metrics found — nothing to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax = axes[0]
    for i, (x, y) in enumerate(r5):
        ax.plot(x, y, color=C_FIT, lw=1.2, alpha=0.75, label="per seed" if i == 0 else None)
    if r5:
        n = min(len(y) for _, y in r5)
        ax.plot(r5[0][0][:n], np.mean([y[:n] for _, y in r5], axis=0),
                color=C_FIT, lw=2.4, label=f"mean of {len(r5)} seeds")
        _style(ax, "(a) Round 5 — the regression converges\n"
                   "value drift per consolidation, fitted states",
               "value drift (% of |V|)", max(x.max() for x, _ in r5))
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=9, labelcolor=MUT)

    ax = axes[1]
    for i, (x, y) in enumerate(r6f):
        ax.plot(x, y, color=C_FIT, lw=1.2, alpha=0.7,
                label="fitted states (in the regression)" if i == 0 else None)
    for i, (x, y) in enumerate(r6h):
        ax.plot(x, y, color=C_HOLD, lw=1.2, alpha=0.7, ls="--",
                label="held-out states (excluded from it)" if i == 0 else None)
    if r6f:
        _style(ax, "(b) Round 6 — held-out tracks fitted\n"
                   "the memorisation hypothesis is falsified",
               "value drift (% of |V|)", max(x.max() for x, _ in r6f))
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=9, labelcolor=MUT)

    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(a.out_dir, f"consolidation_insitu.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  wrote {p}")

    # numbers for the report, straight from the same logs
    for name, ser in (("round 5 fitted", r5), ("round 6 fitted", r6f), ("round 6 held-out", r6h)):
        if ser:
            allv = np.concatenate([y for _, y in ser])
            last = np.mean([y[-10:].mean() for _, y in ser])
            print(f"  {name:18} overall mean {allv.mean():6.3f}%   final-10-point mean {last:6.3f}%")


if __name__ == "__main__":
    main()
