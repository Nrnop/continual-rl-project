"""Print carry-over, disruption and per-arm returns for the dm_control family.

    python -m src_continuous_control.scripts.report_multienv --gates <dir>    # the gate table
    python -m src_continuous_control.scripts.report_multienv                  # the full study

    carry_over = mean over j != 0 of (R[0, j] - b_j) / (R[0, 0] - b_0)

Row 0 only (the policy trained on task 0 alone), so every cell has the same training budget.
b is the random-init policy's return, the convention `TransferMatrix` and `fwt()` use, not a
uniform-random policy (they differ a lot, e.g. 0.8 vs 32.9 on nominal cartpole). 1 means the task-0
policy is as competent on the other physics as on its own, 0 means back to the floor; values a
little above 1 are possible. Carry-over runs high for weakly trained policies, so compare it only
between runs of the same length.
"""
import argparse
import glob
import os
import pickle

import numpy as np

from .report_tables import perm_p

# The dm_control family's ceiling: reward in [0,1] over exactly 1000 steps, no early termination.
CEILING = 1000.0

ENV_DIRS = ("cartpole_swingup", "reacher_easy", "ball_in_cup_catch",
            "walker_stand", "walker_walk", "cheetah_run")
ARMS = ("vanilla", "ewc", "pt")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# Metrics
def carry_over(matrix, baselines):
    """Fraction of its above-baseline return a task-0 policy keeps on the other tasks.

    Returns nan when the task-0 policy is at or below its baseline (denominator ~ 0).
    """
    m = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(baselines, dtype=np.float64).reshape(-1)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"transfer matrix must be square, got {m.shape}")
    if b.shape != (m.shape[0],):
        raise ValueError(f"expected {m.shape[0]} baselines, got {b.shape}")
    above = m[0] - b
    if not np.isfinite(above[0]) or above[0] <= 1e-9:
        return float("nan")
    return float(np.mean(above[1:]) / above[0])


def disruption(matrix):
    """Spread of one frozen policy's return across the physics settings, as a fraction of its mean.

    Used as the dynamic-range gate and as a per-environment covariate. Uses row 0, as carry_over.
    """
    row = np.asarray(matrix, dtype=np.float64)[0]
    return float((row.max() - row.min()) / max(abs(row.mean()), 1e-9))


def percent_of_ceiling(value):
    return 100.0 * value / CEILING


def run_returns(path):
    """Return (whole-run mean, final-20% mean, last step) for one run, as `report_tables.load`."""
    if any(path.endswith(s) for s in ("_ep_returns.pkl", "_eval_returns.pkl")):
        return None
    a = np.asarray(load_pickle(path), dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        return None
    return float(a[:, 1].mean()), float(a[int(0.8 * len(a)):, 1].mean()), float(a[-1, 0])


# Gate table (short pre-sweep runs)
def gate_table(root, env_dirs=ENV_DIRS):
    """Per-environment dynamic range, ceiling headroom and carry-over, from one vanilla run each."""
    print("=== GATE 2: DYNAMIC RANGE, and the disruption covariate ===")
    print("One frozen policy across every physics setting. >= 20% or the environment is dropped.\n")
    print(f"{'environment':22s} {'disruption':>11s} {'carry-over':>11s} "
          f"{'final':>8s} {'% ceiling':>10s}  {'verdict':>8s}")
    rows = []
    for env in env_dirs:
        hits = glob.glob(os.path.join(root, env, "*_transfer_matrix.pkl"))
        if not hits:
            print(f"{env:22s} (no transfer matrix found)")
            continue
        t = load_pickle(hits[0])
        matrix, baselines = t["transfer_matrix"], t["baselines"]
        d, c = disruption(matrix), carry_over(matrix, baselines)
        final = np.nan
        for p in glob.glob(os.path.join(root, env, "*_returns.pkl")):
            got = run_returns(p)
            if got:
                final = got[1]
        ok = d >= 0.20
        rows.append((env, d, c, final))
        print(f"{env:22s} {d * 100:10.1f}% {c:11.2f} {final:8.1f} "
              f"{percent_of_ceiling(final):9.1f}%  {'PASS' if ok else '**FAIL**':>8s}")
    print("\nOrdered by carry-over - this is the x-axis of the study's headline plot:")
    for env, _, c, _ in sorted(rows, key=lambda r: r[2]):
        print(f"  {c:6.2f}   {env}")
    return rows


# Study tables: medians and exact tests per environment
def arm_returns(root, env, arm):
    """Every seed's (whole-run, final-20%) return for one arm of one environment."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, env, arm, "*_returns.pkl"))):
        got = run_returns(path)
        if got:
            out.append(got[:2])
    return np.asarray(out, dtype=float)


def study_table(root, env_dirs=ENV_DIRS):
    """Print median final-20% return per arm with exact rank-sum tests; return pt's advantage."""
    print("\n=== THE FAMILY: final-20% return per arm, as a percentage of the 1000 ceiling ===\n")
    print(f"{'environment':22s} {'n':>3s} {'vanilla':>9s} {'ewc':>9s} {'pt':>9s} "
          f"{'pt-van':>8s} {'p':>8s} {'pt-ewc':>8s} {'p':>8s}")
    advantages = {}
    for env in env_dirs:
        data = {arm: arm_returns(root, env, arm) for arm in ARMS}
        if any(len(v) == 0 for v in data.values()):
            print(f"{env:22s} (incomplete)")
            continue
        med = {a: float(np.median(v[:, 1])) for a, v in data.items()}
        n = min(len(v) for v in data.values())
        p_van = perm_p(data["pt"][:, 1], data["vanilla"][:, 1])
        p_ewc = perm_p(data["pt"][:, 1], data["ewc"][:, 1])
        adv = (med["pt"] - med["vanilla"]) / max(abs(med["vanilla"]), 1e-9)
        advantages[env] = adv
        print(f"{env:22s} {n:3d} {med['vanilla']:9.1f} {med['ewc']:9.1f} {med['pt']:9.1f} "
              f"{med['pt'] - med['vanilla']:8.1f} {p_van:8.4f} "
              f"{med['pt'] - med['ewc']:8.1f} {p_ewc:8.4f}")
    return advantages


def headline_plot_data(root, env_dirs=ENV_DIRS):
    """Print and return (env, carry-over, pt's relative advantage over vanilla) per environment."""
    advantages = study_table(root, env_dirs)
    print("\n=== THE HEADLINE: carry-over vs pt's advantage over vanilla ===\n")
    print(f"{'environment':22s} {'carry-over':>11s} {'pt advantage':>14s}")
    pts = []
    for env in env_dirs:
        hits = glob.glob(os.path.join(root, env, "vanilla", "*_transfer_matrix.pkl"))
        if not hits or env not in advantages:
            continue
        # Median over seeds: a single transfer matrix (25 cells x 10 episodes) is noisy.
        cs = []
        for h in hits:
            t = load_pickle(h)
            c = carry_over(t["transfer_matrix"], t["baselines"])
            if np.isfinite(c):
                cs.append(c)
        if not cs:
            continue
        c = float(np.median(cs))
        pts.append((env, c, advantages[env]))
        print(f"{env:22s} {c:11.2f} {advantages[env] * 100:13.1f}%")
    if len(pts) >= 3:
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() >= 3:
            r = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
            print(f"\n  Pearson r over {ok.sum()} environments = {r:.3f}")
            print("  With six points this is a description, not a test - report it with the "
                  "scatter,\n  never as a p-value.")
    return pts


def main():
    p = argparse.ArgumentParser(description="the multi-environment family's numbers")
    p.add_argument("--gates", metavar="DIR", default=None,
                   help="a directory of one vanilla run per environment; print the gate table")
    p.add_argument("--results-dir", default="src_continuous_control/results/multienv",
                   help="the study's results tree: <env>/<arm>/*.pkl")
    p.add_argument("--envs", nargs="+", default=list(ENV_DIRS))
    args = p.parse_args()

    if args.gates:
        gate_table(args.gates, args.envs)
        return 0
    headline_plot_data(args.results_dir, args.envs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
