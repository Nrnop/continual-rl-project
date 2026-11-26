"""Run the control runs on cartpole-swingup that separate output-layer decay from the split.

  pt, pt_frozen      the stored `pt` and frozen-permanent runs, re-run with `--log-adv-split`
                     (which only reads the rollout buffer, so each must reproduce its stored return
                     curve exactly on the same machine; `--check` verifies this)
  pt_frozen_nodecay  frozen permanent, transient not decayed  (cartpole_pt_frozen_nodecay)
  van_small          vanilla PPO at the transient's [30, 30]  (cartpole_van_small)
  van_outdecay       vanilla PPO, output layers halved every 10 updates  (cartpole_van_outdecay)

    python -m src_continuous_control.scripts.run_cartpole_controls --jobs 6
    python -m src_continuous_control.scripts.run_cartpole_controls --check
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

OUT = "src_continuous_control/results/cartpole_controls"
STORED = "src_continuous_control/results/cartpole"
#            arm                  agent      overlay                       extra flags
ARMS = [("pt",                "pt",      "cartpole_pt",                ["--log-adv-split"]),
        ("pt_frozen",         "pt",      "cartpole_pt_frozen",         ["--log-adv-split"]),
        ("pt_frozen_nodecay", "pt",      "cartpole_pt_frozen_nodecay", ["--log-adv-split"]),
        ("van_small",         "vanilla", "cartpole_van_small",         []),
        ("van_outdecay",      "vanilla", "cartpole_van_outdecay",      [])]
# the advantage-split re-runs and their stored originals
REPRO = {"pt": f"{STORED}/pt_ppo_seed_{{s}}_returns.pkl",
         "pt_frozen": f"{STORED}/ablation_frozen/pt_ppo_seed_{{s}}_returns.pkl"}


def _run(arm, agent, overlay, extra, seed):
    log_dir = os.path.join(OUT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "src_continuous_control.train",
           "--agent", agent, "--seed", str(seed), "--config", overlay,
           "--results-dir", os.path.join(OUT, arm),
           "--runs-dir", f"src_continuous_control/runs/cartpole_controls/{arm}",
           "--no-wandb", "--no-tb", *extra]
    log_path = os.path.join(log_dir, f"{arm}_seed_{seed}.log")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    print(f"{time.strftime('%H:%M')}  {arm} seed {seed}: rc={rc} "
          f"{(time.time() - started) / 60:.1f} min", flush=True)
    return rc


def _check(seeds):
    ok = True
    for arm, stored in REPRO.items():
        for s in seeds:
            new = os.path.join(OUT, arm, f"pt_ppo_seed_{s}_returns.pkl")
            old = stored.format(s=s)
            if not (os.path.exists(new) and os.path.exists(old)):
                print(f"{arm} seed {s}: MISSING")
                ok = False
                continue
            a, b = (np.asarray(pickle.load(open(p, "rb"))) for p in (new, old))
            same = a.shape == b.shape and np.array_equal(a, b)
            ok &= same
            print(f"{arm} seed {s}: {'IDENTICAL' if same else 'DIFFERS'} to the stored run")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.check:
        sys.exit(0 if _check(args.seeds) else 1)
    # the two re-runs first, so the reproduction check can run before the controls finish
    jobs = ([(*arm, s) for arm in ARMS[:2] for s in args.seeds]
            + [(*arm, s) for s in args.seeds for arm in ARMS[2:]])
    print(f"{time.strftime('%H:%M')}  {len(jobs)} runs, {args.jobs} at a time -> {OUT}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(lambda j: _run(*j), jobs))
    ok = _check(args.seeds)
    print("ALL DONE" if all(c == 0 for c in codes) and ok else "FINISHED WITH PROBLEMS", flush=True)
    sys.exit(0 if all(c == 0 for c in codes) and ok else 1)


if __name__ == "__main__":
    main()
