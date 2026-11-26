"""Print the switching-frequency 2x2: does pt's advantage grow when phases are shorter?

    python -m src_continuous_control.scripts.report_switch2x2

Cells are {cartpole, HalfCheetah} x {5 phases of 614,400 steps, 20 phases of 153,600}, same total
budget, identical except `--switch`, all run on one machine. The predicted direction is that pt's
edge over vanilla grows as phases shorten. The test statistic is

    interaction = [median(pt) - median(vanilla)]_fast - [median(pt) - median(vanilla)]_slow

Exact enumeration is infeasible (C(20,10)^2 ~ 3.4e10), so it uses a seeded random permutation of the
fast/slow labels within each arm. Per-cell comparisons use the exact `report_tables.perm_p`.
"""
import sys

import numpy as np

from .report_tables import load, perm_p

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

R = "src_continuous_control/results"
N_PERM = 20000
CEILING = {"cartpole": 1000.0, "halfcheetah": None}

CELLS = [
    ("cartpole", "slow (5 x 614,400)", f"{R}/switch5_cartpole"),
    ("cartpole", "fast (20 x 153,600)", f"{R}/switch20_cartpole"),
    ("halfcheetah", "slow (5 x 614,400)", f"{R}/switch5_halfcheetah"),
    ("halfcheetah", "fast (20 x 153,600)", f"{R}/switch20_halfcheetah"),
]
ARMS = [("vanilla", "vanilla"), ("ewc", "EWC"), ("pt", "PT")]


def _cell(path):
    out = {}
    for key, _ in ARMS:
        out[key] = load(f"{path}/{key}_ppo_seed_*_returns.pkl")
    return out


def _interaction_p(pt_fast, pt_slow, van_fast, van_slow, col, rng):
    """Return the interaction statistic and its permutation p-value."""
    obs = ((np.median(pt_fast[:, col]) - np.median(van_fast[:, col]))
           - (np.median(pt_slow[:, col]) - np.median(van_slow[:, col])))
    pt_pool = np.concatenate([pt_fast[:, col], pt_slow[:, col]])
    van_pool = np.concatenate([van_fast[:, col], van_slow[:, col]])
    n_pt, n_van = len(pt_fast), len(van_fast)
    hits = 0
    for _ in range(N_PERM):
        p = rng.permutation(pt_pool)
        v = rng.permutation(van_pool)
        stat = ((np.median(p[:n_pt]) - np.median(v[:n_van]))
                - (np.median(p[n_pt:]) - np.median(v[n_van:])))
        hits += abs(stat) >= abs(obs) - 1e-9
    return obs, (hits + 1) / (N_PERM + 1)


def main():
    data = {(env, lab): _cell(path) for env, lab, path in CELLS}

    print("## Return by cell\n")
    print("| environment | phases | arm | seeds | whole-run | final 20% |")
    print("|---|---|---|---:|---:|---:|")
    for env, lab, _ in CELLS:
        for key, name in ARMS:
            d = data[(env, lab)][key]
            if not len(d):
                print(f"| {env} | {lab} | {name} | 0 | - | - |")
                continue
            pct = ""
            if CEILING[env]:
                pct = f" ({100 * np.median(d[:, 1]) / CEILING[env]:.0f}%)"
            print(f"| {env} | {lab} | {name} | {len(d)} | {np.median(d[:, 0]):.0f} "
                  f"| {np.median(d[:, 1]):.0f}{pct} |")

    print("\n## PT's advantage within each cell (exact Mann-Whitney)\n")
    print("| environment | phases | PT − vanilla (whole) | p | PT − vanilla (final) | p |")
    print("|---|---|---:|---:|---:|---:|")
    for env, lab, _ in CELLS:
        pt, van = data[(env, lab)]["pt"], data[(env, lab)]["vanilla"]
        if not len(pt) or not len(van):
            continue
        print(f"| {env} | {lab} "
              f"| {np.median(pt[:, 0]) - np.median(van[:, 0]):+.0f} | {perm_p(pt[:, 0], van[:, 0]):.4f} "
              f"| {np.median(pt[:, 1]) - np.median(van[:, 1]):+.0f} | {perm_p(pt[:, 1], van[:, 1]):.4f} |")

    print(f"\n## THE TEST - does shortening the phase widen PT's edge?"
          f"  ({N_PERM:,} random permutations, seed 0)\n")
    print("| environment | metric | gap when slow | gap when fast | change | p |")
    print("|---|---|---:|---:|---:|---:|")
    for env in ("cartpole", "halfcheetah"):
        slow, fast = data.get((env, "slow (5 x 614,400)")), data.get((env, "fast (20 x 153,600)"))
        if not slow or not fast or not len(slow["pt"]) or not len(fast["pt"]):
            continue
        for col, metric in ((0, "whole-run"), (1, "final 20%")):
            rng = np.random.RandomState(0)
            obs, p = _interaction_p(fast["pt"], slow["pt"], fast["vanilla"], slow["vanilla"],
                                    col, rng)
            g_slow = np.median(slow["pt"][:, col]) - np.median(slow["vanilla"][:, col])
            g_fast = np.median(fast["pt"][:, col]) - np.median(fast["vanilla"][:, col])
            print(f"| {env} | {metric} | {g_slow:+.0f} | {g_fast:+.0f} | {obs:+.0f} | {p:.4f} |")

    print("\nPositive `change` = PT's edge GREW when phases were shortened, i.e. the prediction held.")
    print("Negative or ~0 = evidence against the mechanism. Report it either way.")


if __name__ == "__main__":
    sys.exit(main())
