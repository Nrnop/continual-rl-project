"""Figures for Section 6.9.6: fixed decay (1 - rho) against the absorption actually achieved.

matched_decay_absorption: absorbed fraction c = <dmu_P, rho*mu_T> / ||rho*mu_T||^2 per consolidation
    (c = 1 is exact) for the `pt` arm of every environment, median over the first and last ten
    consolidations, then over seeds.
matched_decay_pilot: final-fifth return when decaying by (1 - c*rho) instead of (1 - rho), five
    seeds per arm, one panel per environment.

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_matched_decay_figure
"""
import argparse
import csv
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from .make_phase2_figures import (INK, INK_MUTED, PT_BLUE, PT_BLUE_LIGHT, SURFACE, AXIS,
                                  _save, _seed_dots, _style)

R = "src_continuous_control/results"
# (label, directory of the pt arm). HalfCheetah is the learned-sigma study; the rest are the
# piecewise dm_control family.
ENVIRONMENTS = [
    ("HalfCheetah", f"{R}/clean/pt"),
    ("cheetah-run", f"{R}/multienv/cheetah_run/pt"),
    ("cartpole-swingup", f"{R}/multienv/cartpole_swingup/pt"),
    ("reacher-easy", f"{R}/multienv/reacher_easy/pt"),
    ("ball_in_cup-catch", f"{R}/multienv/ball_in_cup_catch/pt"),
    ("walker-stand", f"{R}/multienv/walker_stand/pt"),
    ("walker-walk", f"{R}/multienv/walker_walk/pt"),
]
PILOTS = [("cheetah-run", f"{R}/matched_decay_pilot"),
          ("HalfCheetah", f"{R}/matched_decay_pilot_halfcheetah")]
ARMS = [("pt_fixed", "shipped decay\n(1 − ρ)", PT_BLUE),
        ("pt_realised", "matched decay\n(1 − cρ)", PT_BLUE_LIGHT)]


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def early_late_absorption(directory, window=10):
    """Per seed: median actor absorption over the first and the last `window` consolidations."""
    early, late = [], []
    for p in sorted(glob.glob(f"{directory}/*_consolidation_records.pkl")):
        c = [r["actor_absorbed_align"] for r in _load(p)]
        early.append(float(np.median(c[:window])))
        late.append(float(np.median(c[-window:])))
    return np.array(early), np.array(late)


def final_fifth(directory):
    out = []
    for p in sorted(glob.glob(f"{directory}/*_returns.pkl")):
        if p.endswith(("_ep_returns.pkl", "_eval_returns.pkl")):
            continue
        a = _load(p)[:, 1]
        out.append(float(a[int(0.8 * len(a)):].mean()))
    return np.array(out)


def absorption_figure(out_dir):
    data = [(label, *early_late_absorption(d)) for label, d in ENVIRONMENTS]
    data.sort(key=lambda row: np.median(row[2]))  # lowest late absorption first
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(data))
    w = 0.36
    rows = []
    for i, (label, early, late) in enumerate(data):
        # Early is grey because light blue means the matched-decay arm in the pilot figure.
        for dx, vals, color, when in ((-w / 2, early, AXIS, "first 10"),
                                      (w / 2, late, PT_BLUE, "last 10")):
            med = float(np.median(vals))
            ax.bar(i + dx, med, width=w - 0.04, color=color, linewidth=0, zorder=2,
                   label=f"{when} consolidations" if i == 0 else None)
            ax.text(i + dx, med + 0.015, f"{med:.2f}", ha="center", va="bottom", color=INK,
                    fontsize=8.5)
            rows.append({"environment": label, "window": when, "median_c": med,
                         "seeds": len(vals)})
    # Dotted line: c = 1 (exact absorption).
    ax.axhline(1.0, color=INK_MUTED, linestyle=(0, (1, 2)), linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in data], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, len(data) - 0.4)
    _style(ax, "", "absorbed fraction c")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    _save(fig, out_dir, "matched_decay_absorption")
    return rows


def pilot_figure(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    rows = []
    for ax, (label, directory) in zip(axes, PILOTS):
        values = [final_fifth(f"{directory}/{arm}") for arm, _, _ in ARMS]
        top = max(v.max() for v in values)
        ax.set_ylim(0, top * 1.18)
        for pos, ((arm, _, color), v) in enumerate(zip(ARMS, values)):
            med = float(np.median(v))
            ax.bar(pos, med, width=0.6, color=color, linewidth=0, zorder=2)
            _seed_dots(ax, pos, v)
            # Median label sits above all seed dots so it does not overlap one.
            ax.text(pos, top * 1.08, f"{med:.0f}", ha="center", va="bottom", color=INK,
                    fontsize=9)
            rows += [{"environment": label, "arm": arm, "seed": s, "final_fifth_return": float(x)}
                     for s, x in enumerate(v)]
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([a[1] for a in ARMS], fontsize=9)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        # Environment name goes in the x-axis label.
        _style(ax, label, "final-fifth return")
    fig.subplots_adjust(wspace=0.35)
    _save(fig, out_dir, "matched_decay_pilot")
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="src_continuous_control/plots/figures_matched_decay")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _write_csv(os.path.join(args.out_dir, "matched_decay_absorption.csv"),
               absorption_figure(args.out_dir))
    _write_csv(os.path.join(args.out_dir, "matched_decay_pilot.csv"), pilot_figure(args.out_dir))


if __name__ == "__main__":
    main()
