"""Pilot: decay the transient by what the permanent absorbed, instead of by a fixed rho.

At consolidation the transient is decayed by (1 - rho), but the permanent absorbs only c*rho of it
(c ~ 0.56 on cheetah-run late in training), so (1 - c)*rho*f_T leaks from the composed function
every k updates. `--decay-match realised` decays by (1 - c*rho) instead. Two arms, same seeds,
run together on one machine:
  pt_fixed     the default `pt` (control)
  pt_realised  identical except --decay-match realised

    python -m src_continuous_control.scripts.run_matched_decay_pilot --seeds 0 1 2 3 4 --jobs 10

    # HalfCheetah instead of cheetah-run:
    python -m src_continuous_control.scripts.run_matched_decay_pilot --overlay phase2_hard \
        --results-dir src_continuous_control/results/matched_decay_pilot_halfcheetah \
        --runs-dir src_continuous_control/runs/matched_decay_pilot_halfcheetah \
        --log-dir src_continuous_control/results/matched_decay_pilot_halfcheetah/logs
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ARMS = {"pt_fixed": [], "pt_realised": ["--decay-match", "realised"]}
DEFAULT_OVERLAY = "multienv_cheetah_run_pt"


def _run(arm, seed, args):
    results_dir = os.path.join(args.results_dir, arm)
    os.makedirs(args.log_dir, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "src_continuous_control.train",
           "--agent", "pt", "--seed", str(seed), "--config", args.overlay,
           "--results-dir", results_dir, "--runs-dir", args.runs_dir,
           "--no-wandb", "--no-tb", "--async-envs", "false", *ARMS[arm]]
    if args.total_steps is not None:
        cmd += ["--total-steps", str(args.total_steps)]
    if args.switch is not None:
        cmd += ["--switch", str(args.switch)]
    log_path = os.path.join(args.log_dir, f"{arm}_seed_{seed}.log")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    print(f"{arm} seed {seed}: rc={rc} {(time.time() - started) / 60:.1f} min  {log_path}",
          flush=True)
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=10)
    # cheetah-run by default; `phase2_hard` for HalfCheetah (the overlay of results/clean/pt).
    p.add_argument("--overlay", default=DEFAULT_OVERLAY)
    p.add_argument("--results-dir", default="src_continuous_control/results/matched_decay_pilot")
    p.add_argument("--runs-dir", default="src_continuous_control/runs/matched_decay_pilot")
    p.add_argument("--log-dir", default="src_continuous_control/results/matched_decay_pilot/logs")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--switch", type=int, default=None)
    args = p.parse_args()
    jobs = [(arm, seed) for seed in args.seeds for arm in ARMS]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(lambda job: _run(*job, args), jobs))
    sys.exit(0 if all(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    main()
