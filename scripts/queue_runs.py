"""Run a list of training jobs with a concurrency cap, skipping runs that are already done.

    python -m src_continuous_control.scripts.queue_runs --plan clean --jobs 7

Each child PID is recorded in `<results>/.queue_pids` so `--stop` kills the training processes,
not just a parent shell. A run whose `*_returns.pkl` already reaches full length is skipped, so an
interrupted plan resumes at run granularity (a killed run restarts from step 0). The cap counts
every training process on the machine, including ones this launcher did not start.
"""
import argparse
import os
import pickle
import subprocess
import sys
import time

import numpy as np

# plan name -> list of (arm, agent, config[, seeds])
#
# `arm` is a path relative to --results-dir and may contain a subdirectory ("clean/vanilla").
# Each arm directory holds exactly one experiment. An optional 4th element overrides the seed list
# for that entry.
PLANS = {
    "clean": [
        ("vanilla", "vanilla", "phase2_hard"),
        ("pt", "pt", "phase2_hard"),
        ("ewc", "ewc", "phase2_hard"),
        ("pt_double_actor", "pt", "pt_double_actor"),
        ("pt_frozen", "pt", "phase2_hard_ablation_frozen"),
    ],
    # Reward flips with the one-hot task label appended to the observation. PT hides the label
    # from its permanent so only the transient reads it. Compare with results/s14reset/* (same
    # sigma and reset, no label).
    "taskid": [
        ("taskid/van", "vanilla", "taskid_van", [0, 1, 2, 3, 4, 5]),
        ("taskid/pt",  "pt",      "taskid_pt",  [0, 1, 2, 3, 4, 5]),
        ("taskid/ewc", "ewc",     "taskid_ewc", [0, 1, 2, 3, 4, 5]),
    ],
    # P1: physics benchmark seeds 5-9 (at 5 vs 5 the exact permutation floor is p = 0.0079).
    #     EWC goes to clean/ewc_fixed, kept apart from clean/ewc, which holds an EWC variant with
    #     the penalty divided by parameter count and log_std anchored.
    # P2: the frozen-permanent ablation for the double-actor setup.
    # P3/P4: sigma sweep at 0.55 and 0.20, between and below the tested values 0.37 and 1.0.
    "overnight": [
        ("clean/vanilla",    "vanilla", "phase2_hard",   [5, 6, 7, 8, 9]),   # P1
        ("clean/pt",         "pt",      "phase2_hard",   [5, 6, 7, 8, 9]),
        ("clean/ewc_fixed",  "ewc",     "phase2_hard",   [5, 6, 7, 8, 9]),
        ("clean/pt_double_actor",     "pt",      "pt_double_actor", [5, 6, 7, 8, 9]),
        ("clean/pt_double_actor_frozen", "pt",   "pt_double_actor_frozen", [0, 1, 2, 3, 4]),   # P2
        ("sigma_sweep/s055_van", "vanilla", "s055_van",  [0, 1, 2, 3, 4]),   # P3
        ("sigma_sweep/s055_pt",  "pt",      "s055_pt",   [0, 1, 2, 3, 4]),
        ("sigma_sweep/s055_ewc", "ewc",     "s055_ewc",  [0, 1, 2, 3, 4]),
        ("sigma_sweep/s020_van", "vanilla", "s020_van",  [0, 1, 2, 3, 4]),   # P4
        ("sigma_sweep/s020_pt",  "pt",      "s020_pt",   [0, 1, 2, 3, 4]),
        ("sigma_sweep/s020_ewc", "ewc",     "s020_ewc",  [0, 1, 2, 3, 4]),
    ],
    # Config-only runs on existing code paths, ordered so the most useful finish first.
    "night_0817": [
        # LayerNorm on both sigma-0.37 benchmarks, all three arms. The unnormalised controls are
        # {van,pt,ewc}_physics_s037 and s14reset/*. Parameter parity with LN: pt/van 1.006.
        ("layernorm_physics_s037/vanilla_ln", "vanilla", "ln_physics_s037_van", [0, 1, 2, 3, 4]),
        ("layernorm_physics_s037/pt_ln",      "pt",      "ln_physics_s037_pt",  [0, 1, 2, 3, 4]),
        ("layernorm_physics_s037/ewc_ln",     "ewc",     "ln_physics_s037_ewc", [0, 1, 2, 3, 4]),
        ("layernorm_rewardflip_s037/vanilla_ln", "vanilla", "ln_flip_s037_van", [0, 1, 2, 3, 4]),
        ("layernorm_rewardflip_s037/pt_ln",      "pt",      "ln_flip_s037_pt",  [0, 1, 2, 3, 4]),
        ("layernorm_rewardflip_s037/ewc_ln",     "ewc",     "ln_flip_s037_ewc", [0, 1, 2, 3, 4]),
        # Stationary ceiling: task switching off, sigma learned vs frozen at 0.37, 3 seeds.
        ("no_switch_ceiling/vanilla_sigma_learned",
         "vanilla", "ceiling_learned",         [0, 1, 2]),
        ("no_switch_ceiling/vanilla_sigma_frozen_037",
         "vanilla", "ceiling_s037",            [0, 1, 2]),
        # Consolidation order: shuffled buffer vs the default visit order (pt_physics_s037).
        ("consolidation_order/pt_shuffled_vs_pt_physics_s037",
         "pt",      "pt_physics_s037_shuffle", [0, 1, 2, 3, 4]),
        # Capacity at [16,16] (Appendix C.3, "big world, small agent"). The [64,64] end is
        # {van,pt,ewc}_physics_s037. Last because it is the longest; rerunning the plan resumes it.
        ("small_agent_capacity_w16/vanilla_w64to16",
         "vanilla", "cap16_van",               [0, 1, 2, 3, 4]),
        ("small_agent_capacity_w16/pt_perm11_trans8",
         "pt",      "cap16_pt",                [0, 1, 2, 3, 4]),
        ("small_agent_capacity_w16/ewc_w64to16",
         "ewc",     "cap16_ewc",               [0, 1, 2, 3, 4]),
    ],
    # Allocation test. PPO's gradient reaches only pt's transient, which has 0.32x vanilla's
    # trainable parameters under total-parameter parity. Here the transient matches vanilla
    # (1.00x trainable, 1.68x total). Controls: pt_physics_s037 and van_physics_s037.
    "bigtrans": [
        ("allocation_test_bigtransient/pt_trainable_equals_vanilla",
         "pt", "bigtrans_physics_s037", [0, 1, 2, 3, 4]),
    ],
}


def _complete(results_dir, arm, agent, seed, min_steps):
    """Return True if the run's returns curve reaches `min_steps`."""
    path = os.path.join(results_dir, arm, f"{agent}_ppo_seed_{seed}_returns.pkl")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            arr = np.asarray(pickle.load(f), dtype=float)
        return arr.ndim == 2 and arr[-1, 0] >= min_steps
    except Exception:
        return False


def _running():
    """Count running training processes, including ones this launcher did not start."""
    try:
        if os.name == "nt":
            ps = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                  "Where-Object { $_.CommandLine -like '*src_continuous_control.train*' } | "
                  "Measure-Object | ForEach-Object { $_.Count }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True, timeout=60).stdout
            return int(out.strip() or 0)
        out = subprocess.run(["pgrep", "-fc", "src_continuous_control.train"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0)
    except Exception:
        return 0


def _stop(results_dir):
    pid_file = os.path.join(results_dir, ".queue_pids")
    if not os.path.exists(pid_file):
        print(f"no pid file at {pid_file}")
        return 0
    pids = [l.split()[0] for l in open(pid_file) if l.strip()]
    for pid in pids:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        else:
            subprocess.run(["kill", "-9", pid], capture_output=True)
    print(f"stopped {len(pids)} recorded processes")
    return 0


def main():
    p = argparse.ArgumentParser(description="Queue training runs with a concurrency cap")
    p.add_argument("--plan", default="clean", choices=list(PLANS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=7)
    p.add_argument("--results-dir", default="src_continuous_control/results/clean")
    p.add_argument("--min-steps", type=int, default=3_000_000,
                   help="a curve shorter than this counts as unfinished")
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--stop", action="store_true", help="kill the runs this launcher started")
    p.add_argument("--allow-existing", action="store_true",
                   help="queue behind training processes this launcher did not start")
    p.add_argument("--wait-for-idle", action="store_true",
                   help="block until no training processes remain, then start (chain after "
                        "another launcher instead of competing with it for slots)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.stop:
        return _stop(args.results_dir)

    # --wait-for-idle blocks until no training process is running, so two launchers never
    # compete for the same slots. Require three consecutive quiet polls: a launcher between
    # waves shows zero for a few seconds.
    if args.wait_for_idle:
        quiet = 0
        while quiet < 3:
            n = _running()
            quiet = quiet + 1 if n == 0 else 0
            if n:
                print(f"[queue] waiting for the machine to go idle - {n} training process(es) "
                      f"still alive", flush=True)
            time.sleep(args.poll)
        print("[queue] machine idle; starting", flush=True)

    existing = _running()
    if existing and not args.allow_existing:
        print(
            f"REFUSING TO START: {existing} training process(es) are already running.\n"
            f"  They may be orphans from a launcher that was stopped - killing a launcher does\n"
            f"  NOT kill its training children. Inspect them, then either wait, or run:\n"
            f"    python -m src_continuous_control.scripts.queue_runs --stop "
            f"--results-dir {args.results_dir}\n"
            f"  Pass --allow-existing to queue behind them deliberately (the cap counts them).")
        return 2

    todo, done = [], []
    for entry in PLANS[args.plan]:
        arm, agent, cfg = entry[0], entry[1], entry[2]
        seeds = entry[3] if len(entry) > 3 else args.seeds
        for s in seeds:
            (done if _complete(args.results_dir, arm, agent, s, args.min_steps)
             else todo).append((arm, agent, cfg, s))

    print(f"plan '{args.plan}': {len(done)} already complete, {len(todo)} to run")
    for arm, agent, cfg, s in done:
        print(f"  skip  {arm:10s} seed {s}")
    for arm, agent, cfg, s in todo:
        print(f"  queue {arm:10s} seed {s}  (--config {cfg})")
    if args.dry_run or not todo:
        return 0

    log_dir = os.path.join(args.results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    pid_file = os.path.join(args.results_dir, ".queue_pids")
    open(pid_file, "w").close()
    started = []

    for arm, agent, cfg, s in todo:
        while _running() >= args.jobs:
            time.sleep(args.poll)
        # Re-check before launching: another launcher may have finished this run while we waited.
        if _complete(args.results_dir, arm, agent, s, args.min_steps):
            print(f"[queue] skip {arm} seed {s} - completed while waiting", flush=True)
            continue
        out_dir = os.path.join(args.results_dir, arm)
        os.makedirs(out_dir, exist_ok=True)
        # arm may be a nested path ("clean/vanilla"); flatten it for the log filename.
        log = open(os.path.join(log_dir, f"{arm.replace('/', '_')}_seed_{s}.log"), "w")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "src_continuous_control.train",
             "--agent", agent, "--seed", str(s), "--config", cfg,
             "--async-envs", "false", "--no-wandb", "--no-tb",
             "--results-dir", out_dir, "--runs-dir", out_dir],
            stdout=log, stderr=subprocess.STDOUT, env=env)
        with open(pid_file, "a") as f:
            f.write(f"{proc.pid} {arm} seed {s}\n")
        started.append((proc, arm, s))
        print(f"[queue] started {arm} seed {s} (pid {proc.pid}); "
              f"{_running()} training processes now alive", flush=True)
        time.sleep(2)

    print("[queue] everything launched; waiting for completion", flush=True)
    failures = 0
    for proc, arm, s in started:
        code = proc.wait()
        failures += code != 0
        print(f"[queue] {arm} seed {s} exited {code}", flush=True)
    print(f"[queue] done: {len(started) - failures}/{len(started)} succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
