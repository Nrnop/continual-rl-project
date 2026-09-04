"""The multi-environment family's numbers: carry-over, disruption, and the per-arm comparison.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.scripts.report_multienv --gates <dir>    # the gate table
    python -m src_continuous_control.scripts.report_multienv                  # the full study

WHY THIS FILE EXISTS AT ALL. The carry-over number is the single measurement the project's central
explanation rests on — "`pt` pays off when consecutive tasks share structure", supported by 0.56 on
cartpole against 0.23 on HalfCheetah — and until now it was computed **by no committed script**. It
was worked out once, by hand, for CARTPOLE_RESULTS.md section 9, and the definition survived only
as a parenthesis in that document. The family study puts six of these numbers on the x-axis of the
plot that justifies it, so the definition is written down here, in code, with tests.

THE DEFINITION, spelled out because two reasonable people would not otherwise compute the same
thing:

    carry_over = mean over j != 0 of  (R[0, j] - b_j) / (R[0, 0] - b_0)

  * ROW 0 ONLY. R[0, j] is the policy that has trained on task 0 and nothing else, evaluated on
    task j. Using row 0 alone keeps the training budget behind every cell identical — later rows
    have seen more tasks AND more steps, so a difference between them confounds transfer with
    training time.
  * BASELINE-CORRECTED. Competence means return above what an untrained policy scores, and the
    untrained score is not negligible or constant: a random-init policy already scores ~139 on
    walker-stand by falling over slowly, and ~0.1 on cheetah-run. Dividing raw returns would make
    walker look like it transfers almost perfectly when most of what "transfers" is the floor.
  * b IS THE RANDOM-INIT POLICY, NOT A UNIFORM-RANDOM ONE. That is the convention `TransferMatrix`
    already stores and `fwt()` already uses, and the two differ a lot — measured on the gate runs,
    a random-init network scores 0.8 on nominal cartpole where uniform-random noise scores 32.9,
    because an initialised Gaussian actor emits small smooth correlated actions rather than white
    noise. Mixing the two conventions inside one column would make the column meaningless, which is
    the same lesson `report_tables.perm_p` records about rank-sum versus median-difference.

READING IT. carry_over = 1 means a task-0 policy is just as competent on the other physics as on
its own; 0 means the physics change destroys everything it learned, back to the floor; negative
means it is actively worse than an untrained policy on the other tasks. Values slightly above 1 are
possible and are not a bug — they mean the other tasks are, for that policy, easier than the one it
trained on.

ONE HONEST CAVEAT. Carry-over measured off a WEAKLY TRAINED row 0 runs high, because a policy that
has learned little has little that is specific to lose. The gate runs (204k steps per task) give
cartpole 1.13 where the committed full-length figure is 0.78. Compare carry-over only between runs
of the same length.
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


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------
def carry_over(matrix, baselines):
    """Fraction of its competence a task-0-only policy keeps on the other tasks.

    See the module docstring for why it is row 0 and why it is baseline-corrected. Returns nan
    when the task-0 policy learned nothing at all (denominator ~ 0), because "what fraction of
    nothing survives" has no answer and a huge ratio would otherwise be reported as a real result.
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
    """How far the physics move ONE FROZEN POLICY's return, as a fraction of its mean.

    The dynamic-range gate, and the covariate reported in place of the calibration that
    MULTIENV_TASK.md section 2.3 asks for — see scripts/make_multienv_configs.py for why that
    calibration is deliberately not done. Row 0, for the same budget reason as carry_over.
    """
    row = np.asarray(matrix, dtype=np.float64)[0]
    return float((row.max() - row.min()) / max(abs(row.mean()), 1e-9))


def percent_of_ceiling(value):
    return 100.0 * value / CEILING


def run_returns(path):
    """(whole-run mean, final-20% mean) from one run's training curve.

    Mirrors `report_tables.load`, including its exclusion of the two other `*returns.pkl` shapes,
    so a number here means the same thing as the same number there.
    """
    if any(path.endswith(s) for s in ("_ep_returns.pkl", "_eval_returns.pkl")):
        return None
    a = np.asarray(load_pickle(path), dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        return None
    return float(a[:, 1].mean()), float(a[int(0.8 * len(a)):, 1].mean()), float(a[-1, 0])


# ---------------------------------------------------------------------------
# The gate table — what the short pre-sweep runs answered
# ---------------------------------------------------------------------------
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
    print("\nOrdered by carry-over — this is the x-axis of the study's headline plot:")
    for env, _, c, _ in sorted(rows, key=lambda r: r[2]):
        print(f"  {c:6.2f}   {env}")
    return rows


# ---------------------------------------------------------------------------
# The study — medians and exact tests, per environment and across the family
# ---------------------------------------------------------------------------
def arm_returns(root, env, arm):
    """Every seed's (whole-run, final-20%) return for one arm of one environment."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, env, arm, "*_returns.pkl"))):
        got = run_returns(path)
        if got:
            out.append(got[:2])
    return np.asarray(out, dtype=float)


def study_table(root, env_dirs=ENV_DIRS):
    """Medians, not means, and exact rank-sum tests — the project's standing convention.

    ALWAYS ALONGSIDE HOW MUCH THE ARM LEARNED. Retention-flavoured metrics all improve when an
    agent simply learns less: on HalfCheetah corr(peak return, BWT) was -0.745, and a frozen arm
    scored BWT ~ 0 with a peak return of exactly 0.0. Every row here carries the return that the
    transfer numbers belong to.
    """
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
    """carry-over on x, pt's advantage over vanilla on y. Six points, one per environment.

    The plot the study exists to produce: it either draws a line or refutes the shared-structure
    explanation. Printed as a table so the numbers can be checked before anything is drawn.
    """
    advantages = study_table(root, env_dirs)
    print("\n=== THE HEADLINE: carry-over vs pt's advantage over vanilla ===\n")
    print(f"{'environment':22s} {'carry-over':>11s} {'pt advantage':>14s}")
    pts = []
    for env in env_dirs:
        hits = glob.glob(os.path.join(root, env, "vanilla", "*_transfer_matrix.pkl"))
        if not hits or env not in advantages:
            continue
        # MEDIAN OVER SEEDS, not the first file on disk. A transfer matrix is 25 cells of 10
        # evaluation episodes each and is noisy; reading one seed made ball_in_cup-catch look like
        # the lowest-carry-over environment in the family at 0.18 when the median across ten seeds
        # is 0.57. Everything else here is a median over seeds and this has to match, or the
        # figures and the report disagree about the study's own x-axis.
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
            print("  With six points this is a description, not a test — report it with the "
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
