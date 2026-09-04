"""T9 — the Phase 2 sweep. Four arms x 5 seeds, run locally with bounded parallelism.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.scripts.run_phase2_sweep --jobs 7
    python -m src_continuous_control.scripts.run_phase2_sweep --jobs 7 --overlay phase2_hard \
        --results-dir src_continuous_control/results/hard      # T11, the harder variant

Arms:
    vanilla, ewc, pt                    -- the three methods
    pt_frozen (--config phase2_ablation_frozen, its own results dir)
                                        -- bar 2 of figure (c); without it there is no ablation

THREAD PINNING IS NOT OPTIONAL. Training here is CPU-bound on MuJoCo physics, and torch will
happily open a thread pool per job; unpinned, throughput collapses when several runs share the
machine. Each job gets OMP/MKL_NUM_THREADS=1.

THE PRE-FLIGHT IS A GATE, NOT A SUGGESTION. The cheap checks (parameter parity, sigma parity, the
physics actually differing) run first and abort the sweep if any fails — a broken invariant found
after four hours of compute is four hours of compute. Pass --skip-preflight only when you have
just run it yourself. The dynamic-range check is separate and must be run once by hand
(`preflight --dynamic-range`), because it is a full-length run in its own right.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time

DEFAULT_ARMS = ("vanilla", "ewc", "pt", "pt_frozen")

# arm -> (agent, results subdirectory). The frozen arm's overlay comes from --frozen-config,
# because it has to change with the main overlay: the hard variant needs a frozen arm that is
# ALSO hard, or bar 2 of the ablation is measuring a different environment from bar 3.
ARM_SPEC = {
    "vanilla": ("vanilla", None),
    "ewc": ("ewc", None),
    "pt": ("pt", None),
    "pt_frozen": ("pt", "ablation_frozen"),
}

# --- per-arm overlays, for benchmarks where the arms cannot share one -----------------------
# HalfCheetah's arms share a single --overlay because they differ only in `--agent`. Cartpole's
# cannot: at obs 5 / act 1 the shipped `pt` widths land at 0.931x the baseline's parameters, so
# `pt` needs its own widths and therefore its own overlay. Handing every arm the same cartpole
# overlay would silently run the whole study with `pt` 7% down on capacity — which is precisely
# failure mode #3, a config key one arm reads and another ignores.
BENCHMARK_OVERLAYS = {
    "halfcheetah": None,        # arms share --overlay, as before
    "cartpole": {
        "vanilla": "cartpole_van",
        "ewc": "cartpole_ewc",
        "pt": "cartpole_pt",
        "pt_frozen": "cartpole_pt_frozen",
    },
    # Standard PPO exploration (log_std trainable from 1.0), same environment. No frozen-permanent
    # ablation here: that arm exists to decompose the mechanism, and it should be read against the
    # study it belongs to rather than duplicated across exploration settings.
    "cartpole_learned": {
        "vanilla": "cartpole_van_learned",
        "ewc": "cartpole_ewc_learned",
        "pt": "cartpole_pt_learned",
        "pt_frozen": None,
    },
}

# --- the multi-environment family ---------------------------------------------------------------
# One benchmark per dm_control environment, named for it. Registered from the env specs so a
# seventh environment cannot be added without its overlays, and so no invented short names can
# creep in — results/ already holds sixty-odd directories called things like `s14reset` and `sup`,
# and MANIFEST.md exists only because nobody could tell them apart.
from ..envs.dm_control_drift import SPECS as _DMC_SPECS      # noqa: E402

MULTIENV_BENCHMARKS = tuple(_DMC_SPECS)
for _env in MULTIENV_BENCHMARKS:
    BENCHMARK_OVERLAYS[_env] = {
        arm: "multienv_%s_%s" % (_env.replace("-", "_"), arm) for arm in ARM_SPEC
    }

# The two boundary-free drift settings, registered as "<setting>:<environment>" so one sweep call
# still means one environment in one setting. ball_in_cup-catch is deliberately absent -- it
# saturated in the boundary study (all three arms within 13 points of each other at 84% of the
# ceiling) and would measure nothing here either. See scripts/make_multienv_configs.py.
from .make_multienv_configs import DRIFT_ENVIRONMENTS as _DRIFT_ENVS   # noqa: E402
from .make_multienv_configs import SETTINGS as _SETTINGS               # noqa: E402

DRIFT_BENCHMARKS = tuple("%s:%s" % (s, e) for s in _SETTINGS for e in _DRIFT_ENVS)
for _key in DRIFT_BENCHMARKS:
    _setting, _env = _key.split(":", 1)
    BENCHMARK_OVERLAYS[_key] = {
        arm: "%s_%s_%s" % (_SETTINGS[_setting]["stem"], _env.replace("-", "_"), arm)
        for arm in ARM_SPEC
    }

# Benchmarks whose arms each get their OWN results directory, named for the arm.
#
# The rest of this project writes every arm of a sweep into one directory and tells them apart by
# the `<arm>_ppo_seed_<n>_` filename prefix. The family study does not, because it has 6 x 3 x 10
# runs and a flat tree of 720 files is unreadable and untraceable. The layout is
# results/multienv/<environment>/<arm>/, one sweep call per environment, so the folder alone says
# what it holds.
ARM_SUBDIR_BENCHMARKS = set(MULTIENV_BENCHMARKS) | set(DRIFT_BENCHMARKS)


def _results_dir(arm, args):
    """Where this arm's pickles go.

    For the family, `--results-dir` is the ENVIRONMENT's directory and each arm gets a folder
    inside it. Elsewhere the historical behaviour is unchanged: arms share a directory except
    `pt_frozen`, which has always had its own.
    """
    if args.benchmark in ARM_SUBDIR_BENCHMARKS:
        return os.path.join(args.results_dir, arm)
    subdir = ARM_SPEC[arm][1]
    return os.path.join(args.results_dir, subdir) if subdir else args.results_dir


def _job_command(arm, seed, args):
    agent, _subdir = ARM_SPEC[arm]
    results_dir = _results_dir(arm, args)
    # -u: unbuffered stdout. Redirected to a file, Python block-buffers in ~8KB chunks,
    # which at ~110 bytes per progress line means a healthy run writes nothing for half an
    # hour — making a long sweep impossible to monitor and a wedged run indistinguishable
    # from a busy one.
    cmd = [sys.executable, "-u", "-m", "src_continuous_control.train",
           "--agent", agent, "--seed", str(seed),
           "--results-dir", results_dir, "--runs-dir", args.runs_dir,
           "--no-wandb", "--no-tb", "--async-envs", str(args.async_envs).lower()]
    # Only ONE --config is applied, so the frozen arm's overlay must already contain whatever the
    # main overlay says. `phase2_hard_ablation_frozen.yaml` is that merge for the hard variant.
    per_arm = BENCHMARK_OVERLAYS.get(args.benchmark)
    if per_arm is not None:
        overlay = per_arm[arm]
    else:
        overlay = args.frozen_config if arm == "pt_frozen" else args.overlay
    if overlay:
        cmd += ["--config", overlay]
    # Shortened runs, for exercising the pipeline end to end before committing hours to it.
    if args.total_steps is not None:
        cmd += ["--total-steps", str(args.total_steps)]
    if args.switch is not None:
        cmd += ["--switch", str(args.switch)]
    return cmd, results_dir


def _run_one(arm, seed, args, log_dir):
    cmd, _ = _job_command(arm, seed, args)
    log_path = os.path.join(log_dir, f"{arm}_seed_{seed}.log")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc.returncode, time.time() - started, log_path


def main():
    p = argparse.ArgumentParser(description="Phase 2 sweep")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARM_SPEC))
    p.add_argument("--jobs", type=int, default=7, help="concurrent runs")
    p.add_argument("--overlay", type=str, default=None,
                   help="config overlay for the vanilla/ewc/pt arms, e.g. phase2_hard")
    p.add_argument("--frozen-config", type=str, default="phase2_ablation_frozen",
                   help="overlay for the pt_frozen arm; must already include whatever --overlay "
                        "says (use phase2_hard_ablation_frozen with --overlay phase2_hard)")
    # SyncVectorEnv, not Async. AsyncVectorEnv spawns num_envs SUBPROCESSES per run, so 7 parallel
    # runs would be ~63 processes; measured on this 8-core box, sync is also slightly FASTER per
    # run (1518 vs 1432 sps). On a many-core machine, pass --async-envs true.
    p.add_argument("--async-envs", type=lambda v: str(v).lower() in ("true", "1", "yes"),
                   default=False)
    p.add_argument("--results-dir", type=str, default="src_continuous_control/results")
    p.add_argument("--runs-dir", type=str, default="src_continuous_control/runs")
    p.add_argument("--total-steps", type=int, default=None,
                   help="override the config's run length (for a short end-to-end rehearsal)")
    p.add_argument("--switch", type=int, default=None,
                   help="override the task length; keep total_steps = 5 x switch")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--benchmark", choices=list(BENCHMARK_OVERLAYS), default="halfcheetah",
                   help="which environment to sweep. 'cartpole' selects the per-arm cartpole_* "
                        "overlays and gates on the cartpole pre-flight; --overlay is then ignored.")
    p.add_argument("--dry-run", action="store_true", help="print the job list and stop")
    args = p.parse_args()

    # SEED-MAJOR, not arm-major. The queue is drained in order, so an arm-major list
    # ([vanilla x 10, ewc x 10, pt x 10]) finishes every vanilla seed before starting a single pt
    # one — and a sweep stopped, crashed or interrupted halfway then yields a complete baseline and
    # no method to compare it against, which is worth nothing. Interleaving by seed keeps the arms
    # within about one seed of each other at every moment, so ANY prefix of the sweep is a usable
    # balanced study. Ordering affects execution only; each run's result is unchanged.
    jobs = [(arm, seed) for seed in args.seeds for arm in args.arms]
    print(f"=== Phase 2 sweep: {len(jobs)} runs, {args.jobs} at a time ===")
    for arm, seed in jobs:
        cmd, _ = _job_command(arm, seed, args)
        print(f"  {arm:<10} seed {seed}: {' '.join(cmd[2:])}")
    if args.dry_run:
        return 0

    if not args.skip_preflight:
        from .preflight import (BENCHMARKS, MULTIENV_PARITY_TOLERANCE, check_non_inert,
                                check_parameter_parity, check_physics_change, check_sigma_parity)
        # preflight registers the family as "multienv:<env>"; the sweep takes the bare environment
        # name because that is also the results directory's name.
        is_drift = args.benchmark in DRIFT_BENCHMARKS
        is_family = args.benchmark in MULTIENV_BENCHMARKS or is_drift
        # preflight registers environments as "multienv:<env>"; a drift benchmark is
        # "<setting>:<env>", and its GATES are the environment's -- parity, sigma and
        # non-inertness are properties of the body, not of the schedule.
        env_name = args.benchmark.split(":", 1)[1] if is_drift else args.benchmark
        bench_key = ("multienv:" + env_name) if is_family else args.benchmark
        spec = BENCHMARKS[bench_key]
        obs_dim, act_dim = spec["obs_dim"], spec["act_dim"]
        # Gate the overlays this sweep will ACTUALLY use, not the boundary ones that share
        # the environment: a drift config with the wrong widths would otherwise pass.
        overlays = ({a: BENCHMARK_OVERLAYS[args.benchmark][a] for a in ("vanilla", "ewc", "pt")}
                    if is_drift else spec["overlays"])
        # The family's widths were re-derived per environment to within 0.5%, and the failure this
        # gate exists to catch is a quiet 7% capacity handicap — which the loose default waves
        # through. Tighten it for the family and leave the older benchmarks as they were.
        tolerance = MULTIENV_PARITY_TOLERANCE if is_family else 1.40
        gates = {"parameter parity": check_parameter_parity(obs_dim, act_dim, tolerance=tolerance,
                                                            overlays=overlays),
                 "sigma parity": check_sigma_parity(obs_dim, act_dim, overlays=overlays),
                 "physics change": check_physics_change(bench_key, overlays=overlays)}
        if is_family:
            # An inert parameter passes "physics change" — the model numbers differ — while
            # changing nothing about the simulation. That distinction cost Phase 1 a week.
            gates["non-inert"] = check_non_inert(bench_key, overlays=overlays)
        failed = [name for name, ok in gates.items() if not ok]
        if failed:
            print(f"\nABORTING before the sweep: {', '.join(failed)} failed the pre-flight.")
            return 1
        print("\n[sweep] pre-flight gates passed.")
        print("[sweep] REMINDER: the dynamic-range check is separate — if vanilla's return varies "
              "by\n        less than ~20% across the task sequence, this sweep cannot separate the "
              "methods.")

    log_dir = os.path.join(args.runs_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    results, lock = [], threading.Lock()

    def worker():
        while True:
            try:
                arm, seed = pending.get_nowait()
            except queue.Empty:
                return
            code, secs, log_path = _run_one(arm, seed, args, log_dir)
            with lock:
                results.append((arm, seed, code, secs))
                status = "ok" if code == 0 else f"FAILED ({code})"
                print(f"[sweep] {arm} seed {seed}: {status} in {secs / 60:.1f} min -> {log_path}",
                      flush=True)

    started = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(args.jobs, 1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failures = [(a, s) for a, s, code, _ in results if code != 0]
    print(f"\n=== {len(results) - len(failures)}/{len(results)} runs finished in "
          f"{(time.time() - started) / 60:.1f} min ===")
    if failures:
        print("FAILED: " + ", ".join(f"{a} seed {s}" for a, s in failures))
        print(f"Check the logs in {log_dir}.")
        return 1
    print("\nNext: python -m src_continuous_control.plots.make_phase2_figures "
          f"--seeds {' '.join(map(str, args.seeds))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
