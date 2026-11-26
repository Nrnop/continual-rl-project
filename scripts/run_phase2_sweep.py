"""Run a sweep of arms x seeds with bounded parallelism.

Run from the parent directory of the package:

    python -m src_continuous_control.scripts.run_phase2_sweep --jobs 7
    python -m src_continuous_control.scripts.run_phase2_sweep --jobs 7 --overlay phase2_hard \
        --results-dir src_continuous_control/results/hard
    python -m src_continuous_control.scripts.run_phase2_sweep --benchmark walker-walk \
        --arms vanilla ewc pt --seeds 0 1 2 3 4 5 6 7 8 9 --jobs 47 \
        --results-dir src_continuous_control/results/multienv/walker_walk

Default arms are vanilla, ewc, pt and pt_frozen (lr_perm = 0, the ablation's second bar). Each job
runs with OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1: training is CPU-bound and unpinned torch thread
pools make throughput collapse when runs share the machine. The parity, sigma and physics gates
from `preflight` run first and abort the sweep on failure (skip with --skip-preflight). The
dynamic-range gate is a full-length run and must be run separately with `preflight --dynamic-range`.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time

DEFAULT_ARMS = ("vanilla", "ewc", "pt", "pt_frozen")

# arm -> (agent, results subdirectory). The frozen arm's overlay comes from --frozen-config so it
# can follow the main overlay; otherwise the ablation bars would compare different environments.
ARM_SPEC = {
    "vanilla": ("vanilla", None),
    "ewc": ("ewc", None),
    "pt": ("pt", None),
    "pt_frozen": ("pt", "ablation_frozen"),
}

# The k/rho sweep's arms are all the `pt` agent with different consolidation interval k and
# transfer fraction rho. They are arms, not benchmarks, so one sweep call is one environment and
# each lands in a results folder named for its settings.
from .make_multienv_configs import K_RHO_SWEEP as _K_RHO          # noqa: E402
from .make_multienv_configs import SWEEP_ENVIRONMENTS as _SWEEP_ENVS   # noqa: E402
from .make_multienv_configs import (k_sweep_stem_for,             # noqa: E402
                                    stationary_stem_for)

K_SWEEP_ARMS = tuple("pt_k%d_rho%03d" % (v["k"], round(v["rho"] * 100)) for v in _K_RHO)
for _arm in K_SWEEP_ARMS:
    ARM_SPEC[_arm] = ("pt", None)

# Per-arm overlays, for benchmarks where the arms cannot share one. HalfCheetah's arms share
# --overlay. On cartpole (obs 5 / act 1) the HalfCheetah `pt` widths give 0.931x the baseline's
# parameters, so `pt` needs its own widths and therefore its own overlay.
BENCHMARK_OVERLAYS = {
    "halfcheetah": None,        # arms share --overlay
    "cartpole": {
        "vanilla": "cartpole_van",
        "ewc": "cartpole_ewc",
        "pt": "cartpole_pt",
        "pt_frozen": "cartpole_pt_frozen",
    },
    # Standard PPO exploration (log_std trainable from 1.0). No frozen-permanent ablation arm.
    "cartpole_learned": {
        "vanilla": "cartpole_van_learned",
        "ewc": "cartpole_ewc_learned",
        "pt": "cartpole_pt_learned",
        "pt_frozen": None,
    },
}

# The dm_control family: one benchmark per environment, named for it. Registered from the env
# specs so a new environment cannot be added without its overlays.
from ..envs.dm_control_drift import SPECS as _DMC_SPECS      # noqa: E402

MULTIENV_BENCHMARKS = tuple(_DMC_SPECS)
for _env in MULTIENV_BENCHMARKS:
    BENCHMARK_OVERLAYS[_env] = {
        arm: "multienv_%s_%s" % (_env.replace("-", "_"), arm) for arm in ARM_SPEC
    }

# The two boundary-free drift settings, registered as "<setting>:<environment>" so one sweep call
# is one environment in one setting. ball_in_cup-catch is excluded because it saturates (all arms
# within 13 points of each other at 84% of the ceiling). See scripts/make_multienv_configs.py.
from .make_multienv_configs import DRIFT_ENVIRONMENTS as _DRIFT_ENVS   # noqa: E402
from .make_multienv_configs import SETTINGS as _SETTINGS               # noqa: E402

DRIFT_BENCHMARKS = tuple("%s:%s" % (s, e) for s in _SETTINGS for e in _DRIFT_ENVS)
for _key in DRIFT_BENCHMARKS:
    _setting, _env = _key.split(":", 1)
    BENCHMARK_OVERLAYS[_key] = {
        arm: "%s_%s_%s" % (_SETTINGS[_setting]["stem"], _env.replace("-", "_"), arm)
        for arm in ARM_SPEC
    }

# The k/rho sweep and the stationary control, both "<setting>:<environment>" over
# SWEEP_ENVIRONMENTS. `ksweep` has only `pt` arms, so its pre-flight uses the Lipschitz2 trio for
# the same environment, whose widths are identical.
#
# The k sweep carries its own baselines: `pt_k10_rho050` (the Lipschitz2 `pt` overlay, k = 10,
# rho = 0.5) and `ewc` are re-run alongside the new arms. A 3.07M-step PPO run is chaotic across
# CPUs, so every cell of one comparison must run on the same hardware.
KSWEEP_BASELINE_ARM = "pt_k10_rho050"
ARM_SPEC[KSWEEP_BASELINE_ARM] = ("pt", None)

KSWEEP_BENCHMARKS = tuple("ksweep:%s" % e for e in _SWEEP_ENVS)
for _key in KSWEEP_BENCHMARKS:
    _env = _key.split(":", 1)[1]
    _l2 = "%s_%s" % (_SETTINGS["lipschitz2"]["stem"], _env.replace("-", "_"))
    BENCHMARK_OVERLAYS[_key] = {
        arm: k_sweep_stem_for(_env, v) for arm, v in zip(K_SWEEP_ARMS, _K_RHO)
    }
    # The baseline arms reuse the Lipschitz2 overlays unchanged.
    BENCHMARK_OVERLAYS[_key][KSWEEP_BASELINE_ARM] = _l2 + "_pt"
    BENCHMARK_OVERLAYS[_key]["ewc"] = _l2 + "_ewc"

STATIONARY_BENCHMARKS = tuple("stationary:%s" % e for e in _SWEEP_ENVS)
for _key in STATIONARY_BENCHMARKS:
    _env = _key.split(":", 1)[1]
    BENCHMARK_OVERLAYS[_key] = {
        arm: stationary_stem_for(_env, arm) for arm in ("vanilla", "ewc", "pt")
    }

# Benchmarks whose arms each get their own results directory, <results-dir>/<arm>/, with one
# sweep call per environment. Other benchmarks put all arms in one directory, told apart by the
# `<arm>_ppo_seed_<n>_` filename prefix.
ARM_SUBDIR_BENCHMARKS = (set(MULTIENV_BENCHMARKS) | set(DRIFT_BENCHMARKS)
                        | set(KSWEEP_BENCHMARKS) | set(STATIONARY_BENCHMARKS))


def _check_k_rho(args):
    """Check each ksweep arm's merged config has the expected k and rho, and decay_rho == rho."""
    import argparse as _argparse

    from ..train import build_config
    ok = True
    print()
    print("[preflight] k / rho, read back off the merged config:")
    expected = list(zip(K_SWEEP_ARMS, _K_RHO))
    expected.append((KSWEEP_BASELINE_ARM, dict(k=10, rho=0.50)))
    for arm, variant in expected:
        if arm not in args.arms:
            continue
        overlay = BENCHMARK_OVERLAYS[args.benchmark][arm]
        cfg = build_config(_argparse.Namespace(agent="pt", config=overlay))
        got_k, got_rho = int(cfg["k"]), float(cfg["rho"])
        # decay_rho != rho breaks composition preservation: the policy jumps at each consolidation.
        got_decay = float(cfg.get("decay_rho", got_rho))
        good = (got_k == variant["k"] and abs(got_rho - variant["rho"]) < 1e-9
                and abs(got_decay - got_rho) < 1e-9)
        ok = ok and good
        print("  %-16s k=%-3d rho=%.2f decay_rho=%.2f   %s"
              % (arm, got_k, got_rho, got_decay, "ok" if good else "MISMATCH"))
    return ok


def _results_dir(arm, args):
    """Return this arm's results directory.

    For ARM_SUBDIR_BENCHMARKS, `--results-dir` is the environment's directory and each arm gets a
    folder inside it. Otherwise arms share it, except `pt_frozen`, which gets a subdirectory.
    """
    if args.benchmark in ARM_SUBDIR_BENCHMARKS:
        return os.path.join(args.results_dir, arm)
    subdir = ARM_SPEC[arm][1]
    return os.path.join(args.results_dir, subdir) if subdir else args.results_dir


def _job_command(arm, seed, args):
    agent, _subdir = ARM_SPEC[arm]
    results_dir = _results_dir(arm, args)
    # -u: unbuffered stdout, so progress lines reach the log file as they are printed.
    cmd = [sys.executable, "-u", "-m", "src_continuous_control.train",
           "--agent", agent, "--seed", str(seed),
           "--results-dir", results_dir, "--runs-dir", args.runs_dir,
           "--no-wandb", "--no-tb", "--async-envs", str(args.async_envs).lower()]
    # Only one --config is applied, so the frozen arm's overlay must already include the main
    # overlay's settings (e.g. phase2_hard_ablation_frozen.yaml for phase2_hard).
    per_arm = BENCHMARK_OVERLAYS.get(args.benchmark)
    if per_arm is not None:
        overlay = per_arm[arm]
    else:
        overlay = args.frozen_config if arm == "pt_frozen" else args.overlay
    if overlay:
        cmd += ["--config", overlay]
    # Shortened runs, for testing the pipeline end to end.
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
    p = argparse.ArgumentParser(description="Arms x seeds sweep")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    # Default resolved after parsing, since it depends on the benchmark.
    p.add_argument("--arms", nargs="+", default=None, choices=list(ARM_SPEC))
    p.add_argument("--jobs", type=int, default=7, help="concurrent runs")
    p.add_argument("--overlay", type=str, default=None,
                   help="config overlay for the vanilla/ewc/pt arms, e.g. phase2_hard")
    p.add_argument("--frozen-config", type=str, default="phase2_ablation_frozen",
                   help="overlay for the pt_frozen arm; must already include whatever --overlay "
                        "says (use phase2_hard_ablation_frozen with --overlay phase2_hard)")
    # SyncVectorEnv by default: AsyncVectorEnv spawns num_envs subprocesses per run, which
    # oversubscribes a small machine when several runs share it. Pass --async-envs true on a
    # many-core machine.
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

    # ksweep: the (k, rho) variants plus its baselines. stationary: no frozen ablation arm.
    if args.arms is None:
        if args.benchmark in KSWEEP_BENCHMARKS:
            # Baselines run alongside, on the same hardware (see KSWEEP_BASELINE_ARM).
            args.arms = list(K_SWEEP_ARMS) + [KSWEEP_BASELINE_ARM, "ewc"]
        elif args.benchmark in STATIONARY_BENCHMARKS:
            args.arms = ["vanilla", "ewc", "pt"]
        else:
            args.arms = list(DEFAULT_ARMS)
    unknown = [a for a in args.arms if a not in BENCHMARK_OVERLAYS.get(args.benchmark, {a: None})]
    if unknown:
        p.error("benchmark %s has no overlay for arm(s): %s. Available: %s"
                % (args.benchmark, ", ".join(unknown),
                   ", ".join(sorted(BENCHMARK_OVERLAYS[args.benchmark]))))

    # Seed-major order keeps the arms within about one seed of each other, so a sweep stopped
    # partway still gives a balanced comparison. Order does not affect any run's result.
    jobs = [(arm, seed) for seed in args.seeds for arm in args.arms]
    print(f"=== Sweep: {len(jobs)} runs, {args.jobs} at a time ===")
    for arm, seed in jobs:
        cmd, _ = _job_command(arm, seed, args)
        print(f"  {arm:<10} seed {seed}: {' '.join(cmd[2:])}")
    if args.dry_run:
        return 0

    if not args.skip_preflight:
        from .preflight import (BENCHMARKS, MULTIENV_PARITY_TOLERANCE, check_non_inert,
                                check_parameter_parity, check_physics_change, check_sigma_parity)
        # preflight registers the family as "multienv:<env>"; the sweep takes the bare name.
        is_drift = args.benchmark in DRIFT_BENCHMARKS
        is_ksweep = args.benchmark in KSWEEP_BENCHMARKS
        is_stationary = args.benchmark in STATIONARY_BENCHMARKS
        # "<setting>:<environment>" benchmarks gate on the environment: parity, sigma and
        # non-inertness do not depend on the schedule.
        is_qualified = is_drift or is_ksweep or is_stationary
        is_family = args.benchmark in MULTIENV_BENCHMARKS or is_qualified
        env_name = args.benchmark.split(":", 1)[1] if is_qualified else args.benchmark
        bench_key = ("multienv:" + env_name) if is_family else args.benchmark
        spec = BENCHMARKS[bench_key]
        obs_dim, act_dim = spec["obs_dim"], spec["act_dim"]
        # Gate the overlays this sweep will actually use. `ksweep` gates the Lipschitz2 trio for
        # the same environment: render_k_sweep emits the same widths and drift keys.
        if is_ksweep:
            gate_key = "lipschitz2:" + env_name
        else:
            gate_key = args.benchmark
        overlays = ({a: BENCHMARK_OVERLAYS[gate_key][a] for a in ("vanilla", "ewc", "pt")}
                    if is_qualified else spec["overlays"])
        # The family's widths match to within 0.5%, so use the tight tolerance there.
        tolerance = MULTIENV_PARITY_TOLERANCE if is_family else 1.40
        gates = {"parameter parity": check_parameter_parity(obs_dim, act_dim, tolerance=tolerance,
                                                            overlays=overlays),
                 "sigma parity": check_sigma_parity(obs_dim, act_dim, overlays=overlays),
                 "physics change": check_physics_change(bench_key, overlays=overlays)}
        if is_family:
            # An inert parameter passes "physics change" while leaving the trajectory unchanged.
            gates["non-inert"] = check_non_inert(bench_key, overlays=overlays)
        if is_ksweep:
            gates["k/rho realised"] = _check_k_rho(args)
        failed = [name for name, ok in gates.items() if not ok]
        if failed:
            print(f"\nABORTING before the sweep: {', '.join(failed)} failed the pre-flight.")
            return 1
        print("\n[sweep] pre-flight gates passed.")
        print("[sweep] REMINDER: the dynamic-range check is separate - if vanilla's return varies "
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
