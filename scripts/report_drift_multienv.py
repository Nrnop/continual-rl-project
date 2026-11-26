"""Print returns for the dm_control family under boundary-free drift, and the L1 vs L2 contrast.

    python -m src_continuous_control.scripts.report_drift_multienv
    python -m src_continuous_control.scripts.report_drift_multienv --settings lipschitz1

With no task boundaries there is no transfer matrix, so only returns are reported. Lipschitz2 adds
a fast ripple (~100 cycles per run) to Lipschitz1's slow drift; the contrast is
change = (pt - baseline) under L2 - (pt - baseline) under L1, printed beside both gaps because a
positive change with two negative gaps means pt lost by less. Lipschitz2's multiplier range is
1.2 against Lipschitz1's 1.0, so it also carries 20% more total drift.
"""
import argparse
import glob
import os

import numpy as np

from .report_multienv import CEILING, load_pickle, percent_of_ceiling, run_returns
from .report_tables import perm_p

SETTINGS = ("lipschitz1", "lipschitz2")
ENV_DIRS = ("cartpole_swingup", "reacher_easy", "walker_stand", "walker_walk", "cheetah_run")
ARMS = ("vanilla", "ewc", "pt")


def arm_runs(root, setting, env, arm):
    """Per-seed (whole-run mean, final-20% mean) for one cell."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, setting, env, arm,
                                              f"{arm}_ppo_seed_*_returns.pkl"))):
        got = run_returns(path)
        if got and got[2] >= 3_000_000:
            out.append(got[:2])
    return np.asarray(out, dtype=float)


def _median(v, col):
    return float(np.median(v[:, col])) if len(v) else float("nan")


def returns_table(root, settings, envs):
    """Medians per arm per environment per setting, as a percentage of the 1000 ceiling."""
    data = {}
    for setting in settings:
        print(f"\n=== {setting.upper()}: whole-run and (final-20%) median return, ceiling 1000 ===\n")
        print(f"{'environment':20s} {'n':>3s} {'vanilla':>16s} {'ewc':>16s} {'pt':>16s} "
              f"{'pt-van':>9s} {'p':>8s}")
        for env in envs:
            cell = {a: arm_runs(root, setting, env, a) for a in ARMS}
            if any(len(v) == 0 for v in cell.values()):
                print(f"{env:20s} (incomplete)")
                continue
            data[(setting, env)] = cell
            n = min(len(v) for v in cell.values())
            txt = []
            for a in ARMS:
                txt.append("%7.1f (%6.1f)" % (_median(cell[a], 0), _median(cell[a], 1)))
            gap = _median(cell["pt"], 0) - _median(cell["vanilla"], 0)
            p = perm_p(cell["pt"][:, 0], cell["vanilla"][:, 0])
            print(f"{env:20s} {n:>3d} " + " ".join(f"{t:>16s}" for t in txt)
                  + f" {gap:>9.1f} {p:>8.4f}")
        print("\n  Percent of ceiling, whole-run: "
              + ", ".join(
                  f"{env}={percent_of_ceiling(_median(data[(setting, env)]['pt'], 0)):.0f}%"
                  for env in envs if (setting, env) in data))
    return data


def contrast_table(data, envs):
    """Print the change in pt's gap over each baseline from Lipschitz1 to Lipschitz2.

    p is a permutation test on the setting label (see `_contrast_p`), which tests the interaction
    rather than either gap on its own.
    """
    print("\n=== THE CONTRAST: does pt's lead grow from Lipschitz1 to Lipschitz2? ===")
    print("Both gaps are shown; the change alone is meaningless without them.")
    print("A positive change with two NEGATIVE gaps means pt lost by less, not that pt won.\n")
    print(f"{'environment':20s} {'baseline':9s} {'metric':11s} {'gap L1':>8s} {'gap L2':>8s} "
          f"{'change':>8s} {'p':>8s}")
    rows = []
    for env in envs:
        if ("lipschitz1", env) not in data or ("lipschitz2", env) not in data:
            continue
        for baseline in ("vanilla", "ewc"):
            for col, metric in ((0, "whole-run"), (1, "final 20%")):
                g1 = (_median(data[("lipschitz1", env)]["pt"], col)
                      - _median(data[("lipschitz1", env)][baseline], col))
                g2 = (_median(data[("lipschitz2", env)]["pt"], col)
                      - _median(data[("lipschitz2", env)][baseline], col))
                p = _contrast_p(data, env, baseline, col)
                rows.append((env, baseline, metric, g1, g2, g2 - g1, p))
                print(f"{env:20s} {baseline:9s} {metric:11s} {g1:>8.1f} {g2:>8.1f} "
                      f"{g2 - g1:>8.1f} {p:>8.3f}")
    print("\n  ⚠️  Lipschitz2 carries 20% more total drift (multiplier range 1.2 vs 1.0), so part")
    print("      of any change is more variation rather than a fast component. Report it with")
    print("      every number in the change column.")
    print("  ⚠️  These rows are NOT independent: two metrics x two baselines come from the same")
    print("      runs, so five environments give five independent cases, not twenty.")
    return rows


def _contrast_p(data, env, baseline, col, n_perm=20000, seed=0):
    """Permutation test on the setting label, for the difference of differences."""
    rng = np.random.RandomState(seed)
    pt1, pt2 = data[("lipschitz1", env)]["pt"][:, col], data[("lipschitz2", env)]["pt"][:, col]
    b1, b2 = (data[("lipschitz1", env)][baseline][:, col],
              data[("lipschitz2", env)][baseline][:, col])
    obs = (np.median(pt2) - np.median(b2)) - (np.median(pt1) - np.median(b1))
    pt_pool, b_pool = np.concatenate([pt1, pt2]), np.concatenate([b1, b2])
    n_pt, n_b = len(pt1), len(b1)
    hits = 0
    for _ in range(n_perm):
        p_perm, b_perm = rng.permutation(pt_pool), rng.permutation(b_pool)
        stat = ((np.median(p_perm[n_pt:]) - np.median(b_perm[n_b:]))
                - (np.median(p_perm[:n_pt]) - np.median(b_perm[:n_b])))
        hits += abs(stat) >= abs(obs) - 1e-9
    return (hits + 1) / (n_perm + 1)


def main():
    p = argparse.ArgumentParser(description="the family's drift settings")
    p.add_argument("--results-dir", default="src_continuous_control/results/multienv_drift")
    p.add_argument("--settings", nargs="+", default=list(SETTINGS))
    p.add_argument("--envs", nargs="+", default=list(ENV_DIRS))
    args = p.parse_args()

    data = returns_table(args.results_dir, args.settings, args.envs)
    if len(args.settings) == 2:
        contrast_table(data, args.envs)
    else:
        print("\n(one setting only - the Lipschitz1 vs Lipschitz2 contrast needs both)")
    print("\nNo forward or backward transfer in this setting: there are no task indices to "
          "define them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
