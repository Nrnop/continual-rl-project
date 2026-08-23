"""What the transient's decay costs, on its own, on both environments.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.plots.plot_decay_gain

THE ONE CLEAN INSTRUMENT IN THIS PROJECT. At each boundary the training loop evaluates the policy,
scales the transient actor's output layer by (1 - rho), and evaluates again -- with **no gradient
step in between**. The difference is what the decay ALONE does, with learning held still.

Why it is the clean one: for a SPLIT CRITIC the decay is provably zero-effect, so whatever this
measures is purely what the split ACTOR's decay costs. Nothing else logged in a run isolates that.

PLOTTED AS A PERCENTAGE of the pre-decay return, not in raw return units. HalfCheetah returns run
~1500 and cartpole ~600, so a raw gain of +110 and +9 are the same thing in different currencies.
`probe/decay_before` is the denominator.

WHAT WOULD BE CONCLUSIVE. If the gain is strongly negative on HalfCheetah and ~0 on cartpole, the
decay SCHEDULE is what breaks the method there -- a fixable knob (rho), not a dead mechanism. If it
is ~0 on both, the decay is exonerated and the problem lies upstream in what the permanent stores.

CAVEAT, as everywhere in this comparison: the HalfCheetah runs use LEARNED sigma and the cartpole
runs freeze it at 0.37, so a difference between the panels is not purely an environment difference.
"""
import glob
import os
import pickle

import numpy as np

from .make_phase2_figures import (AXIS, GRID, INK, INK_MUTED, PT_BLUE, SURFACE, _style)

SETS = [
    ("cartpole", "src_continuous_control/results/cartpole/pt_ppo_seed_*_scalars.pkl"),
    ("HalfCheetah", "src_continuous_control/results/clean/pt/*_scalars.pkl"),
]


def _relative_gain(pattern):
    """(after - before) / |before| at each boundary, per seed. Shape (n_seeds, 4)."""
    rows = []
    for path in sorted(glob.glob(pattern)):
        try:
            hist = pickle.load(open(path, "rb"))
        except Exception:
            continue
        gain, before = hist.get("probe/decay_gain"), hist.get("probe/decay_before")
        if gain is None or before is None:
            continue
        g = np.asarray(gain, dtype=float)
        b = np.asarray(before, dtype=float)
        if g.shape[0] != 4 or b.shape[0] != 4:
            continue
        denom = np.abs(b[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            rows.append(np.where(denom > 1e-9, 100.0 * g[:, 1] / denom, np.nan))
    return np.array(rows) if rows else np.zeros((0, 4))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {label: _relative_gain(pat) for label, pat in SETS}
    # A handful of HalfCheetah seeds swing +-650%, which on a shared axis flattens cartpole into a
    # line and hides the very thing the figure is for. Clip to a robust range and SAY how many
    # points fall outside, rather than letting outliers dictate the view or silently dropping them.
    pooled = np.concatenate([v[np.isfinite(v)].ravel() for v in data.values() if v.size])
    lim = float(np.percentile(np.abs(pooled), 90)) * 1.6

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.RandomState(0)

    for ax, (label, _) in zip(axes, SETS):
        vals = data[label]
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.2, zorder=1)
        for b in range(4):
            col = vals[:, b]
            col = col[np.isfinite(col)]
            if not col.size:
                continue
            x = b + 1 + rng.uniform(-0.13, 0.13, col.size)
            # Per-seed points, then the median as a wide tick. One series, so no legend: the
            # panel title names it.
            ax.scatter(x, col, s=26, facecolor="white", edgecolor=AXIS, linewidth=1.0, zorder=2)
            ax.plot([b + 1 - 0.26, b + 1 + 0.26], [np.median(col)] * 2,
                    color=PT_BLUE, linewidth=2.6, solid_capstyle="round", zorder=3)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(-lim, lim)
        n = int(np.isfinite(vals).all(axis=1).sum())
        finite = vals[np.isfinite(vals)]
        clipped = int((np.abs(finite) > lim).sum())
        if clipped:
            ax.text(0.98, 0.03, f"{clipped} of {finite.size} points beyond ±{lim:.0f}%",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color=INK_MUTED, fontsize=8)
        _style(ax, "Task boundary", "Return change from the decay (% of pre-decay)",
               f"{label}  (n={n})")

    fig.suptitle("What the transient's decay costs on its own — no gradient step in between",
                 color=INK, fontsize=12, x=0.055, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_dir = "src_continuous_control/plots/figures_cartpole"
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"e_decay_gain.{ext}")
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        print(f"[fig] {p}")
    plt.close(fig)

    print("\n  median decay effect, % of pre-decay return")
    for label, _ in SETS:
        v = data[label]
        per = np.nanmedian(v, axis=0)
        print(f"  {label:<12} per-boundary {np.round(per, 2)}   "
              f"overall {np.nanmedian(v):+.2f}%   spread {np.nanmin(v):+.0f}..{np.nanmax(v):+.0f}")


if __name__ == "__main__":
    main()
