"""Print the HalfCheetah summary tables as markdown.

    python -m src_continuous_control.scripts.report_tables

  avg    median across seeds of each seed's mean return over the whole run
  final  the same over the last 20% of the run
  p      exact two-sided Mann-Whitney (see `perm_p`); floor 0.0079 at 5 vs 5, 0.0022 at 6 vs 6

Only runs that reach 3M steps are counted.
"""
import glob
import itertools
import pickle
import sys

import numpy as np

# The tables print non-ASCII characters (sigma, minus); a Windows console defaults to cp1252.
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
        "PT": f"{P2}/clean/pt_double_actor/*_returns.pkl",
        "EWC": f"{P2}/clean/ewc_fixed/*_returns.pkl",
        "PT frozen": f"{P2}/clean/pt_double_actor_frozen/*_returns.pkl"}),
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
    """Return per-seed (whole-run mean, final-20% mean) for full-length runs, shape (n_seeds, 2)."""
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
    """Exact two-sided Mann-Whitney (rank-sum) p-value, by enumerating every split.

    Rank-sum rather than a permutation test on medians: the median-difference statistic ties
    heavily at these sample sizes, and the two can differ a lot (0.008 vs 0.048 on one row). Use
    this one everywhere. Floors: 2/C(10,5) = 0.0079 at 5 vs 5, 2/C(12,6) = 0.0022 at 6 vs 6.
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
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}"


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
            print(f"| {label} | - | *not run yet* | | | | | | | |")
            continue
        row = [cell(med.get(a)) for a in ("vanilla", "PT", "EWC", "PT frozen")]
        # Flag rows whose arms have different seed counts: some runs are still in progress.
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
              f" | {('%+.0f' % dv) if dv is not None else '-'} |"
              f" {('%.3f' % pv) if pv is not None else '-'} |"
              f" {('%+.0f' % de) if de is not None else '-'} |"
              f" {('%.3f' % pe) if pe is not None else '-'} |")

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
