"""Recompute every number in HALFCHEETAH_RESULTS.md from the result files, as markdown.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.scripts.report_tables

Written because every figure in the report had been produced by a separate throwaway snippet. That
works until two snippets disagree, and then there is no way to tell which number is stale -- the
report is the only record and it cannot be re-derived. This script is the single definition of how
a reported number is computed, so the tables can be regenerated from scratch at any time.

Metric definitions, fixed here once:

  avg    -- median across seeds of each seed's OWN mean return over the whole run. Validated
            against the previously reported entries: it reproduces 668, 877, -97, 589, 784, 42 and
            753 exactly.
  final  -- the same, over the last 20% of the run.
  p      -- exact permutation test on the difference of medians, two-sided. At 5v5 the floor is
            0.0079 and at 6v6 it is 0.0022, so a comparison at 5 seeds CANNOT report significance
            below 0.0079 however large the effect.

Directory -> experiment mapping is documented in results/MANIFEST.md; the SETUPS table below must
stay in step with it.
"""
import glob
import itertools
import pickle
import sys

import numpy as np

# The tables use en-dashes and sigma; a Windows console defaults to cp1252 and dies on them.
# Force UTF-8 here rather than requiring the caller to set PYTHONIOENCODING.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

P2 = "src_continuous_control/results"

# label -> {arm: glob}. Arm names are fixed so columns line up across rows.
SETUPS = [
    ("Reward flips · exploration held at 1.0", {
        "vanilla": "stage12_results/van/*_returns.pkl",
        "PT": "stage12_results/pt/*_returns.pkl",
        "EWC": f"{P2}/ewc_flip_s10/*_returns.pkl",
        "PT frozen": "stage12_results/inert/*_returns.pkl"}),
    ("Reward flips · exploration held at 0.37", {
        "vanilla": "stage14_results/van/*_returns.pkl",
        "PT": "stage14_results/pt/*_returns.pkl",
        "EWC": f"{P2}/ewc_flip_s037/*_returns.pkl",
        "PT frozen": "stage14_results/inert/*_returns.pkl"}),
    ("Reward flips · 0.37 · big PT networks", {
        "vanilla": "stage18_results/van/*_returns.pkl",
        "PT": "stage18_results/pt_hisexact/*_returns.pkl",
        "EWC": f"{P2}/ewc_flip_s037/*_returns.pkl"}),
    ("Reward flips · 0.37 · reset at boundary", {
        "vanilla": f"{P2}/s14reset/van/*_returns.pkl",
        "PT": f"{P2}/s14reset/pt/*_returns.pkl",
        "EWC": f"{P2}/s14reset/ewc/*_returns.pkl",
        "PT frozen": f"{P2}/s14reset/inert/*_returns.pkl"}),
    ("Reward flips · 0.37 · reset · TASK LABEL", {
        "vanilla": f"{P2}/taskid/van/*_returns.pkl",
        "PT": f"{P2}/taskid/pt/*_returns.pkl",
        "EWC": f"{P2}/taskid/ewc/*_returns.pkl"}),
    ("Reward flips · σ learned · reset", {
        "vanilla": f"{P2}/rewardflip/vanilla/*_returns.pkl",
        "PT": f"{P2}/rewardflip/pt/*_returns.pkl",
        "EWC": f"{P2}/rewardflip/ewc/*_returns.pkl",
        "PT frozen": f"{P2}/rewardflip/pt_frozen/*_returns.pkl"}),
    ("Physics · σ learned", {
        "vanilla": f"{P2}/clean/vanilla/*_returns.pkl",
        "PT": f"{P2}/clean/pt/*_returns.pkl",
        "EWC": f"{P2}/clean/ewc_fixed/*_returns.pkl",
        "PT frozen": f"{P2}/clean/pt_frozen/*_returns.pkl"}),
    ("Physics · exploration held at 0.37", {
        "vanilla": f"{P2}/van_physics_s037/*_returns.pkl",
        "PT": f"{P2}/pt_physics_s037/*_returns.pkl",
        "EWC": f"{P2}/ewc_physics_s037/*_returns.pkl"}),
    ("Physics · σ learned · big PT networks", {
        "vanilla": f"{P2}/clean/vanilla/*_returns.pkl",
        "PT": f"{P2}/clean/pt_sup/*_returns.pkl",
        "EWC": f"{P2}/clean/ewc_fixed/*_returns.pkl",
        "PT frozen": f"{P2}/clean/pt_sup_frozen/*_returns.pkl"}),
    ("Flips · σ 0.55 · reset", {
        "vanilla": f"{P2}/sigma_sweep/s055_van/*_returns.pkl",
        "PT": f"{P2}/sigma_sweep/s055_pt/*_returns.pkl",
        "EWC": f"{P2}/sigma_sweep/s055_ewc/*_returns.pkl"}),
    ("Flips · σ 0.20 · reset", {
        "vanilla": f"{P2}/sigma_sweep/s020_van/*_returns.pkl",
        "PT": f"{P2}/sigma_sweep/s020_pt/*_returns.pkl",
        "EWC": f"{P2}/sigma_sweep/s020_ewc/*_returns.pkl"}),
]


def load(pattern):
    """Per-seed (whole-run mean, final-20% mean). Only full-length runs count."""
    out = []
    for path in sorted(glob.glob(pattern)):
        if "ep_returns" in path or "eval_returns" in path:
            continue
        try:
            a = np.asarray(pickle.load(open(path, "rb")), dtype=float)
        except Exception:
            continue
        if a.ndim == 2 and a[-1, 0] >= 3_000_000:
            out.append((a[:, 1].mean(), a[int(0.8 * len(a)):, 1].mean()))
    return np.array(out)


def perm_p(x, y):
    """Exact two-sided Mann-Whitney (rank-sum), by enumerating every split.

    THE TEST IS MANN-WHITNEY, NOT A PERMUTATION TEST ON MEDIANS. Both are exact and both are
    defensible, but they give materially different answers on this data -- the flip / sigma-learned
    row is p = 0.008 by rank-sum and p = 0.048 by median-difference -- so the choice has to be made
    once and applied everywhere. Mixing them within one column, which is what happened while these
    tables were maintained by hand, makes the column meaningless.

    Rank-sum is the one used because it is what every historical number in the report was computed
    with, and it is the more powerful of the two here: the median-difference statistic ties heavily
    at these sample sizes and throws away the ordering information that rank-sum keeps.

    Floors: 2/C(10,5) = 0.0079 at 5v5, 2/C(12,6) = 0.0022 at 6v6. A comparison at 5 seeds cannot
    report anything below 0.0079 however large the effect -- which is the entire reason the physics
    arms were taken to 10 seeds.
    """
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    pool = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(pool)) + 1.0
    n = len(x)
    obs = abs(ranks[:n].sum() - n * (len(pool) + 1) / 2.0)
    hits = total = 0
    for idx in itertools.combinations(range(len(pool)), n):
        total += 1
        hits += abs(ranks[list(idx)].sum() - n * (len(pool) + 1) / 2.0) >= obs - 1e-9
    return hits / total


def cell(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}"


def main():
    data = {label: {arm: load(pat) for arm, pat in arms.items()} for label, arms in SETUPS}

    print("## Average return over the whole run\n")
    print("| setup | seeds | vanilla | PT | EWC | PT frozen | PT − vanilla | p | PT − EWC | p |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, arms in SETUPS:
        d = data[label]
        med = {a: (np.median(v[:, 0]) if len(v) else None) for a, v in d.items()}
        n = max((len(v) for v in d.values()), default=0)
        if n == 0:
            print(f"| {label} | — | *not run yet* | | | | | | | |")
            continue
        row = [cell(med.get(a)) for a in ("vanilla", "PT", "EWC", "PT frozen")]
        # A row whose arms hold different numbers of seeds is MID-FLIGHT, not a result: comparing a
        # finished 10-seed arm against a 7-seed one reads as a finding and is an artefact of when
        # the snapshot was taken. Flag it loudly rather than printing a number that looks final.
        counts = {a: len(v) for a, v in d.items() if len(v)}
        provisional = len(set(counts.values())) > 1
        label = label + ("  ⚠️ PARTIAL " + " ".join(f"{a}={c}" for a, c in counts.items())
                         if provisional else "")
        pv = pe = dv = de = None
        if len(d.get("PT", [])) and len(d.get("vanilla", [])):
            dv = med["PT"] - med["vanilla"]; pv = perm_p(d["PT"][:, 0], d["vanilla"][:, 0])
        if len(d.get("PT", [])) and len(d.get("EWC", [])):
            de = med["PT"] - med["EWC"]; pe = perm_p(d["PT"][:, 0], d["EWC"][:, 0])
        print(f"| {label} | {n} | " + " | ".join(row) +
              f" | {('%+.0f' % dv) if dv is not None else '—'} |"
              f" {('%.3f' % pv) if pv is not None else '—'} |"
              f" {('%+.0f' % de) if de is not None else '—'} |"
              f" {('%.3f' % pe) if pe is not None else '—'} |")

    print("\n## Final-phase return (last 20%)\n")
    print("| setup | vanilla | PT | EWC | PT frozen |")
    print("|---|---:|---:|---:|---:|")
    for label, arms in SETUPS:
        d = data[label]
        med = {a: (np.median(v[:, 1]) if len(v) else None) for a, v in d.items()}
        if all(v is None for v in med.values()):
            print(f"| {label} | *not run yet* | | | |")
            continue
        print(f"| {label} | " + " | ".join(cell(med.get(a))
                                           for a in ("vanilla", "PT", "EWC", "PT frozen")) + " |")

    print("\n## Seed counts (a partial arm is a warning, not a result)\n")
    for label, arms in SETUPS:
        counts = ", ".join(f"{a}={len(data[label][a])}" for a in arms)
        print(f"- {label}: {counts}")


if __name__ == "__main__":
    main()
