"""Run a list of training jobs with a concurrency cap, skipping work that is already done.

    python -m src_continuous_control.scripts.queue_runs --plan clean --jobs 7

Written after a bad afternoon, and it exists to prevent three specific things that happened:

  1. ORPHANED CHILDREN. Killing a bash launcher leaves its `python` grandchildren running. Two
     "stopped" studies kept training for over an hour, overlapping a third and putting 20
     processes on 8 cores. Every PID is recorded in `<results>/.queue_pids` as it starts, so
     `--stop` can kill the actual training processes rather than their parent.
  2. REDUNDANT WORK. A run whose `*_returns.pkl` already exists at full length is skipped, so an
     interrupted study resumes at run granularity instead of starting over.
  3. OVERSUBSCRIPTION. The cap counts training processes ACTUALLY RUNNING on the machine, not
     just the ones this launcher started, so it can be started while an earlier wave is still
     finishing and will simply fill slots as they free up.

There is still no mid-run resume: a killed run restarts from step 0. Run granularity is the unit.
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
# `arm` is a path RELATIVE TO --results-dir, so it may contain a subdirectory ("clean/vanilla").
# One arm == one directory == one experiment; nothing else may write there. The mapping from
# directory back to experiment is documented in results/MANIFEST.md, which must be updated
# whenever a plan here changes -- with this many concurrent studies, a result whose provenance
# has to be reconstructed from filenames is a result that cannot be trusted.
#
# A 4th element overrides the seed list for that entry alone (used by the 10-seed top-up, which
# adds seeds 5-9 to arms that already hold 0-4).
PLANS = {
    "clean": [
        ("vanilla", "vanilla", "phase2_hard"),
        ("pt", "pt", "phase2_hard"),
        ("ewc", "ewc", "phase2_hard"),
        ("pt_sup", "pt", "pt_supervisor"),
        ("pt_frozen", "pt", "phase2_hard_ablation_frozen"),
    ],
    # TASK-AWARE reward flips: the one-hot task label is appended to the observation, and PT hides
    # it from its permanent so only the transient can read it. Directly comparable to
    # results/s14reset/* (same sigma, same reset, 6 seeds, no label).
    "taskid": [
        ("taskid/van", "vanilla", "taskid_van", [0, 1, 2, 3, 4, 5]),
        ("taskid/pt",  "pt",      "taskid_pt",  [0, 1, 2, 3, 4, 5]),
        ("taskid/ewc", "ewc",     "taskid_ewc", [0, 1, 2, 3, 4, 5]),
    ],
    # Everything left on the "Runs still needed" list, in the order it should run.
    #
    # P1 takes the physics benchmark from 5 seeds to 10. At 5v5 the exact permutation floor is
    #    p = 0.0079, so a real effect there CANNOT reach significance however large it is; at
    #    10v10 the floor is ~1e-5. EWC goes to clean/ewc_fixed, NOT clean/ewc -- the latter holds
    #    the pre-bugfix EWC (penalty divided by parameter count, log_std anchored) and pooling the
    #    two would silently mix a broken arm with a fixed one.
    # P2 is the one ablation bar missing from the big-network setups.
    # P3/P4 are the sigma sweep. sigma 0.55 runs before 0.20 because it sits between the two
    #    values ever tested (0.37 where PT wins, 1.0 where it loses), so it locates the crossover;
    #    0.20 then says whether the advantage keeps growing below 0.37 or falls off.
    "overnight": [
        ("clean/vanilla",    "vanilla", "phase2_hard",   [5, 6, 7, 8, 9]),   # P1
        ("clean/pt",         "pt",      "phase2_hard",   [5, 6, 7, 8, 9]),
        ("clean/ewc_fixed",  "ewc",     "phase2_hard",   [5, 6, 7, 8, 9]),
        ("clean/pt_sup",     "pt",      "pt_supervisor", [5, 6, 7, 8, 9]),
        ("clean/pt_sup_frozen", "pt",   "pt_sup_frozen", [0, 1, 2, 3, 4]),   # P2
        ("sigma_sweep/s055_van", "vanilla", "s055_van",  [0, 1, 2, 3, 4]),   # P3
        ("sigma_sweep/s055_pt",  "pt",      "s055_pt",   [0, 1, 2, 3, 4]),
        ("sigma_sweep/s055_ewc", "ewc",     "s055_ewc",  [0, 1, 2, 3, 4]),
        ("sigma_sweep/s020_van", "vanilla", "s020_van",  [0, 1, 2, 3, 4]),   # P4
        ("sigma_sweep/s020_pt",  "pt",      "s020_pt",   [0, 1, 2, 3, 4]),
        ("sigma_sweep/s020_ewc", "ewc",     "s020_ewc",  [0, 1, 2, 3, 4]),
    ],
    # 2026-08-17 overnight. Ordered by value-if-the-night-is-cut-short: the ceiling recalibrates
    # every number already in HALFCHEETAH_RESULTS.md, the shuffle ablation tests a named suspect, the
    # capacity sweep is the new experiment. Every entry is CONFIG-ONLY on already-tested code
    # paths — no new agent or env code runs unattended.
    #
    # C1 the ceiling. Two arms, sigma learned vs frozen, task switching OFF. Answers "how good does
    #    our PPO get when nothing changes", which no run has ever measured, and as a by-product
    #    says whether the sigma collapse costs return without any boundaries to recover from.
    #    3 seeds: this is a reference number, not a comparison, so it does not need 5.
    # C2 the consolidation-order ablation. `consolidation_shuffle: true` against the existing
    #    `pt_physics_s037` (5 seeds, same everything else). Every pt run on disk fit its permanent
    #    in visit order; this is the first measurement of what that cost.
    # C3 the capacity sweep at [16,16], the small-agent end of Appendix C.3's boundary condition.
    #    The [64,64] end is the existing {van,pt,ewc}_physics_s037 runs, so this completes a
    #    two-point sweep. All three agents, since the claim is about pt RELATIVE to its baselines.
    # DIRECTORY NAMES SAY WHAT THE EXPERIMENT IS. A results tree read six months from now is the
    # only record that survives, and `cap16/van` does not tell anyone what was being tested or
    # what it should be compared against. Each name below states the manipulation, then the arm.
    "night_0817": [
        # C1 -- LAYERNORM, both sigma-0.37 benchmarks, all three arms. THE INTERVENTION: everything
        #       else queued tonight is diagnostic, this is the one change that could make pt work.
        #       Loss of plasticity under a changing task distribution is the project's exact
        #       failure signature and normalisation is its standard mitigation. The unnormalised
        #       controls already exist ({van,pt,ewc}_physics_s037 and s14reset/*), so only the
        #       normalised arms need running. Parity re-verified with LN on: pt/van 0.993 -> 1.006.
        ("layernorm_physics_s037/vanilla_ln", "vanilla", "ln_physics_s037_van", [0, 1, 2, 3, 4]),
        ("layernorm_physics_s037/pt_ln",      "pt",      "ln_physics_s037_pt",  [0, 1, 2, 3, 4]),
        ("layernorm_physics_s037/ewc_ln",     "ewc",     "ln_physics_s037_ewc", [0, 1, 2, 3, 4]),
        ("layernorm_rewardflip_s037/vanilla_ln", "vanilla", "ln_flip_s037_van", [0, 1, 2, 3, 4]),
        ("layernorm_rewardflip_s037/pt_ln",      "pt",      "ln_flip_s037_pt",  [0, 1, 2, 3, 4]),
        ("layernorm_rewardflip_s037/ewc_ln",     "ewc",     "ln_flip_s037_ewc", [0, 1, 2, 3, 4]),
        # C2 -- how good is our PPO with the non-stationarity switched OFF? The reference every
        #       continual number is read against, and nothing has ever measured it.
        ("no_switch_ceiling/vanilla_sigma_learned",
         "vanilla", "ceiling_learned",         [0, 1, 2]),
        ("no_switch_ceiling/vanilla_sigma_frozen_037",
         "vanilla", "ceiling_s037",            [0, 1, 2]),
        # C3 -- does fitting the permanent in VISIT ORDER (the default, and what every pt run on
        #       disk did) cost anything? Compare against results/pt_physics_s037.
        ("consolidation_order/pt_shuffled_vs_pt_physics_s037",
         "pt",      "pt_physics_s037_shuffle", [0, 1, 2, 3, 4]),
        # C4 -- Appendix C.3's "big world, small agent" prediction, reached by shrinking the agent.
        #       The [64,64] end of the sweep is the existing {van,pt,ewc}_physics_s037 runs.
        #       LAST deliberately: 56 runs is ~12 h and a night is ~8, so this is the tail that
        #       gets cut. queue_runs skips completed runs, so re-running the plan resumes it.
        ("small_agent_capacity_w16/vanilla_w64to16",
         "vanilla", "cap16_van",               [0, 1, 2, 3, 4]),
        ("small_agent_capacity_w16/pt_perm11_trans8",
         "pt",      "cap16_pt",                [0, 1, 2, 3, 4]),
        ("small_agent_capacity_w16/ewc_w64to16",
         "ewc",     "cap16_ewc",               [0, 1, 2, 3, 4]),
    ],
    # THE ALLOCATION TEST. pt's parity is matched on TOTAL parameters, but PPO's gradient reaches
    # only the transient, so pt has been learning with 0.32x vanilla's trainable capacity. This
    # arm sets the trainable halves EQUAL (1.00x) while handing pt 1.68x on totals. Controls are
    # the existing pt_physics_s037 and van_physics_s037 -- neither needs re-running.
    "bigtrans": [
        ("allocation_test_bigtransient/pt_trainable_equals_vanilla",
         "pt", "bigtrans_physics_s037", [0, 1, 2, 3, 4]),
    ],
}


def _complete(results_dir, arm, agent, seed, min_steps):
    """Has this run already finished? True only if the curve reaches full length."""
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
    """Training processes alive right now — anyone's, not just ours."""
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

    # GUARD: never start a study on top of one that is already running. On 2026-08-12 three
    # studies overlapped on an 8-core box (20 training processes) because a "stopped" launcher's
    # children survived; two of them wrote to the same result files. Refuse by default.
    # CHAINING. --wait-for-idle blocks until the machine is quiet, so this study can be armed while
    # an earlier one is still going without ever running two schedulers against the same slot
    # budget -- both see a free slot at the same instant and both fill it. Idle must be SUSTAINED:
    # a launcher between waves shows zero for a few seconds, and starting then would collide with
    # the runs it is about to dispatch. Three consecutive quiet polls means genuinely finished.
    if args.wait_for_idle:
        quiet = 0
        while quiet < 3:
            n = _running()
            quiet = quiet + 1 if n == 0 else 0
            if n:
                print(f"[queue] waiting for the machine to go idle — {n} training process(es) "
                      f"still alive", flush=True)
            time.sleep(args.poll)
        print("[queue] machine idle; starting", flush=True)

    existing = _running()
    if existing and not args.allow_existing:
        print(
            f"REFUSING TO START: {existing} training process(es) are already running.\n"
            f"  They may be orphans from a launcher that was stopped — killing a launcher does\n"
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
        # RE-CHECK immediately before launching, not just when the plan was built. Runs started by
        # an earlier wave may have finished while this queue was waiting for a slot, and launching
        # a second copy of one is exactly the duplication this script exists to stop.
        if _complete(args.results_dir, arm, agent, s, args.min_steps):
            print(f"[queue] skip {arm} seed {s} — completed while waiting", flush=True)
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
