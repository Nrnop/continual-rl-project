"""Pre-run gates that check a benchmark and its arms are set up for a fair comparison.

Run from the parent directory of the package:

    python -m src_continuous_control.scripts.preflight
    python -m src_continuous_control.scripts.preflight --smoke-steps 60000
    python -m src_continuous_control.scripts.preflight --dynamic-range --total-steps 3072000
    python -m src_continuous_control.scripts.preflight --cartpole
    python -m src_continuous_control.scripts.preflight --multienv walker-walk

Gates:
  parameter parity  every arm has a comparable number of parameters.
  sigma parity      every arm starts at the same realised exploration noise and follows the same
                    schedule for it (all frozen or all learned).
  physics change    the physics read back from the live model differ between tasks (or over
                    time, for a continuous drift schedule).
  non-inert         the physics change actually changes the simulated trajectory.
  smoke             each agent trains to completion; pt's permanent absorbs something; EWC's
                    penalty is non-zero after a boundary.
  dynamic range     the physics move a fixed policy's return enough for the methods to be
                    separable. A full-length run, so opt in with --dynamic-range.

Every gate asserts on realised values (constructed modules, live tensors, logged scalars), never on
config keys, since a key that one agent reads and another ignores would otherwise go unnoticed.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

from ..agents import AGENTS
from ..envs.dm_control_drift import SPECS as DMC_SPECS
from ..envs.drift_half_cheetah import LipschitzDriftHalfCheetah
from ..train import build_config

AGENT_NAMES = ("vanilla", "ewc", "pt")

# Benchmark selection. Each gate runs against the config an arm would actually get, and the
# benchmarks differ in observation/action dimensions as well as overlays, so the benchmark is a
# parameter of the gates. The dimensions matter: pt's HalfCheetah widths give 0.99x parity at
# obs 17 / act 6 but 0.931x at cartpole's obs 5 / act 1.
BENCHMARKS = {
    "halfcheetah": dict(obs_dim=17, act_dim=6, overlays={n: None for n in AGENT_NAMES}),
    "cartpole": dict(obs_dim=5, act_dim=1,
                     overlays={"vanilla": "cartpole_van", "ewc": "cartpole_ewc",
                               "pt": "cartpole_pt"}),
    # Standard PPO exploration (log_std trainable from 1.0). Sigma parity reports learned=True for
    # all three arms, which passes: parity means a shared schedule, not a frozen one.
    "cartpole_learned": dict(obs_dim=5, act_dim=1,
                             overlays={"vanilla": "cartpole_van_learned",
                                       "ewc": "cartpole_ewc_learned",
                                       "pt": "cartpole_pt_learned"}),
}

# The dm_control family, registered as "multienv:<dm_control name>". Built from the env specs so a
# new environment cannot skip its gates and the dimensions always match the env.
#
# pt's widths are derived per environment to within 0.5% of the baseline, so the family uses a
# tight parity tolerance. The default 1.40x would let a 7% capacity difference through.
MULTIENV_PARITY_TOLERANCE = 1.02

for _name, _spec in DMC_SPECS.items():
    BENCHMARKS["multienv:" + _name] = dict(
        obs_dim=_spec.obs_dim, act_dim=_spec.act_dim,
        overlays={arm: "multienv_%s_%s" % (_name.replace("-", "_"), arm)
                  for arm in AGENT_NAMES},
        dmc_env=_name,
    )

MULTIENV_BENCHMARKS = tuple("multienv:" + n for n in DMC_SPECS)


def _cfg_for(agent, overlay=None, **overrides):
    """Return the merged config a real run of `agent` would get, using the same merge as train.py.

    The overlay must be passed explicitly: on cartpole and the dm_control family every
    environment-defining key lives there.
    """
    cli = argparse.Namespace(agent=agent, config=overlay, **overrides)
    return build_config(cli)


def _ok(passed):
    return "PASS" if passed else "**FAIL**"


# Parameter parity

def _ppo_trainable(agent):
    """Return (parameters PPO's gradient can reach, total parameters).

    For pt the permanent mean is detached in the training forward, so the PPO update moves only the
    transient networks (and log_std if it is learned). The permanent is moved by consolidation.
    """
    total = sum(p.numel() for p in agent.actor.parameters()) + \
        sum(p.numel() for p in agent.critic.parameters())
    trans = 0
    found_split = False
    for module, attr in ((agent.actor, "trans_mean"), (agent.critic, "trans")):
        sub = getattr(module, attr, None)
        if sub is not None:
            found_split = True
            trans += sum(p.numel() for p in sub.parameters())
    if not found_split:
        return total, total
    # log_std is shared, not split; count it only when it is trainable.
    log_std = getattr(agent.actor, "log_std", None)
    if log_std is not None and log_std.requires_grad:
        trans += log_std.numel()
    return trans, total


def check_parameter_parity(obs_dim=17, act_dim=6, tolerance=1.40, overlays=None):
    """Check every agent's total parameter count is within `tolerance` of vanilla's.

    Both the total and the PPO-trainable count are printed, since they differ for pt by
    construction. The default tolerance allows a permanent at baseline width plus a half-width
    transient (1.32x) and still catches a large mismatch.
    """
    print("\n=== 3. PARAMETER PARITY ===")
    overlays = overlays or {n: None for n in AGENT_NAMES}
    counts, trainables = {}, {}
    for name in AGENT_NAMES:
        torch.manual_seed(0)
        agent = AGENTS[name](obs_dim, act_dim, _cfg_for(name, overlays.get(name)),
                             torch.device("cpu"))
        actor = sum(p.numel() for p in agent.actor.parameters())
        critic = sum(p.numel() for p in agent.critic.parameters())
        counts[name] = (actor, critic)
        trainables[name], _ = _ppo_trainable(agent)
        print(f"  {name:<8} actor={actor:>8,}  critic={critic:>8,}  total={actor + critic:>8,}"
              f"  PPO-trainable={trainables[name]:>8,}")

    base = sum(counts["vanilla"])
    base_trainable = trainables["vanilla"]
    passed = True
    for name in AGENT_NAMES:
        ratio = sum(counts[name]) / base
        tr_ratio = trainables[name] / base_trainable
        if ratio > tolerance or ratio < 1.0 / tolerance:
            passed = False
        print(f"  {name:<8} total/vanilla = {ratio:.4f}x    PPO-trainable/vanilla = {tr_ratio:.4f}x")
    print(f"  {_ok(passed)}: every agent within {tolerance:.2f}x of the baseline's parameter count")
    print("  NOTE: the two ratios differ for `pt` by construction, and both belong in the write-up "
          "-\n        matching totals is the paper's convention; the trainable split is what PPO "
          "actually sees.")
    return passed


# Sigma parity

def check_sigma_parity(obs_dim=17, act_dim=6, overlays=None):
    """Check every agent has the same realised initial sigma and the same log_std trainability."""
    print("\n=== 4. SIGMA PARITY (realised, not configured) ===")
    overlays = overlays or {n: None for n in AGENT_NAMES}
    sigmas, trainable = {}, {}
    for name in AGENT_NAMES:
        torch.manual_seed(0)
        agent = AGENTS[name](obs_dim, act_dim, _cfg_for(name, overlays.get(name)),
                             torch.device("cpu"))
        log_std = agent.actor.log_std.detach()
        sigmas[name] = float(torch.exp(log_std).mean())
        trainable[name] = bool(agent.actor.log_std.requires_grad)
        print(f"  {name:<8} sigma={sigmas[name]:.4f}  learned={trainable[name]}")

    same_start = np.allclose(list(sigmas.values()), sigmas["vanilla"], atol=1e-6)
    same_schedule = len(set(trainable.values())) == 1
    print(f"  {_ok(same_start)}: identical exploration level at initialisation")
    print(f"  {_ok(same_schedule)}: identical exploration SCHEDULE (all frozen or all learned)")
    if not same_schedule:
        print("           log_std is frozen for pt by construction. An arm that "
              "LEARNS it\n           runs a different exploration schedule, so a return difference "
              "confounds the PT\n           mechanism with sigma. Set freeze_log_std: true for "
              "every arm.")
    return same_start and same_schedule


# Physics change

def check_physics_change_over_time(benchmark, cfg):
    """Check a continuous drift schedule moves the physics with time and ignores task boundaries.

    Under `sin`/`linear` the multiplier is a function of elapsed steps and `set_task` is a no-op,
    so the boundary version of this gate does not apply. Instead, realise the physics at four
    phases of the slow cycle and require the witness parameter to vary, then require that
    `set_task` leaves the physics unchanged.
    """
    from ..envs.dm_control_drift import SPECS, DmControlDrift
    print("\n=== 1c. PHYSICS CHANGE OVER TIME (continuous schedule, no boundaries) ===")
    env_name = benchmark.split(":", 1)[1]
    period = int(cfg["drift_period"])
    env = DmControlDrift(
        env_name=env_name, drift_targets=tuple(cfg["drift_targets"]),
        schedule=cfg["drift_schedule"],
        amplitude=cfg["drift_amplitude"], period=period,
        amplitude2=cfg.get("drift_amplitude2", 0.0),
        period2=cfg.get("drift_period2", 30720), phase=cfg.get("drift_phase", 0.0),
        reload_tol=cfg.get("dmc_reload_tol", 0.005), max_episode_steps=50)
    witness = SPECS[env_name].probes[cfg["drift_targets"][0]][0]

    seen = []
    for frac in (0.0, 0.25, 0.5, 0.75):
        # Set the env's step clock directly, then apply the physics through the normal code path.
        env.t = int(frac * period)
        env._apply(env.multiplier())
        seen.append((frac, env.current_params()))
    for frac, params in seen:
        print(f"  t = {frac:.2f} of the slow period: "
              + "  ".join(f"{k}={v:.4f}" for k, v in params.items()))
    values = [p.get(witness) for _, p in seen]
    moves = len(set(np.round(values, 9))) > 1
    print(f"  {_ok(moves)}: the physics move with the clock (witness: {witness})")

    env.t = 0
    env._apply(env.multiplier())
    before = env.current_params()
    env.set_task(3)
    after = env.current_params()
    no_boundary = before == after
    print(f"  {_ok(no_boundary)}: set_task does NOT change the physics - this benchmark has no "
          f"boundaries")
    if not no_boundary:
        print("           A drift run that still responds to task boundaries is the boundary "
              "benchmark\n           wearing a drift filename.")

    lip = env.lipschitz_constant()
    print(f"  Lipschitz bound on the multiplier: {lip:.3e} per env step")
    env.close()
    return moves and no_boundary and lip > 0


def check_physics_change(benchmark="halfcheetah", overlays=None):
    """Check the physics read back from the live model differ across the task sequence.

    Reads the model, not the config, so a key the env ignores cannot pass.
    """
    overlays = overlays or {n: None for n in AGENT_NAMES}
    cfg = _cfg_for("vanilla", overlays.get("vanilla"))
    if str(cfg.get("drift_schedule", "step")) != "step" and benchmark.startswith("multienv:"):
        return check_physics_change_over_time(benchmark, cfg)
    print("\n=== 1c. PHYSICS DIFFER BETWEEN TASKS ===")
    mults = list(cfg["task_multipliers"])
    # Match on the prefix: "cartpole_learned" is the cartpole env with different exploration.
    if benchmark.startswith("multienv:"):
        from ..envs.dm_control_drift import SPECS, DmControlDrift
        env_name = benchmark.split(":", 1)[1]
        env = DmControlDrift(
            env_name=env_name, drift_targets=tuple(cfg["drift_targets"]),
            schedule=cfg["drift_schedule"], task_multipliers=mults, max_episode_steps=50)
        # The witness is the first drift target's probe, which differs per environment.
        witness = SPECS[env_name].probes[cfg["drift_targets"][0]][0]
    elif benchmark.startswith("cartpole"):
        from ..envs.cartpole_swingup import DriftCartpoleSwingup
        env = DriftCartpoleSwingup(
            task_name=cfg.get("cartpole_task", "swingup"),
            drift_targets=tuple(cfg["drift_targets"]),
            schedule=cfg["drift_schedule"], task_multipliers=mults, max_episode_steps=50)
        witness = "drift_pole_length"
    else:
        env = LipschitzDriftHalfCheetah(
            env_id=cfg.get("env_id", "HalfCheetah-v5"),
            drift_targets=tuple(cfg["drift_targets"]),
            schedule=cfg["drift_schedule"], task_multipliers=mults, max_episode_steps=50)
        witness = "drift_damping"
    seen = []
    for i in range(len(mults)):
        env.set_task(i)
        seen.append(env.current_params())
    for i, params in enumerate(seen):
        print(f"  task {i}: " + "  ".join(f"{k}={v:.4f}" for k, v in params.items()))
    values = [p.get(witness) for p in seen]
    passed = len(set(np.round(values, 9))) > 1
    print(f"  {_ok(passed)}: the physics are not constant across the task sequence "
          f"(witness: {witness})")
    env.close()
    return passed


# Non-inert physics

def check_non_inert(benchmark, overlays=None, n_steps=200, threshold=1e-3):
    """Check that scaled physics change the trajectory under the same seed and actions.

    A changed model parameter can still have no effect: `dof_armature` is 0.0 on several bodies and
    `dof_damping` is 0.0005 on cartpole, so a multiplier on either changes nothing. The actions
    are independent of the physics, so any divergence is caused by the physics alone. Only defined
    for the dm_control family.
    """
    print("\n=== 1d. THE PHYSICS ARE NON-INERT (trajectories diverge) ===")
    if not benchmark.startswith("multienv:"):
        print("  only defined for the multienv family; skipping")
        return True
    from ..envs.dm_control_drift import DmControlDrift
    overlays = overlays or {n: None for n in AGENT_NAMES}
    cfg = _cfg_for("vanilla", overlays.get("vanilla"))
    env_name = benchmark.split(":", 1)[1]
    targets = tuple(cfg["drift_targets"])

    if str(cfg.get("drift_schedule", "step")) != "step":
        # Continuous drift: `set_task` is a no-op, so compare the real schedule against a
        # zero-amplitude copy with the same env, seed and actions.
        def drift_trace(amp, amp2, steps):
            env = DmControlDrift(
                env_name=env_name, drift_targets=targets, schedule=cfg["drift_schedule"],
                amplitude=amp, period=int(cfg["drift_period"]), amplitude2=amp2,
                period2=cfg.get("drift_period2", 30720),
                reload_tol=cfg.get("dmc_reload_tol", 0.005), max_episode_steps=steps + 10)
            env.reset(seed=7)
            rng = np.random.RandomState(11)
            out = []
            for _ in range(steps):
                env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
                out.append(np.array(env._dm_env.physics.data.qpos, copy=True))
            env.close()
            return np.asarray(out)

        # Rebuild-based environments only re-apply the physics once the multiplier has moved by
        # `reload_tol`, so a short window can see no change at all on a slow schedule. Size the
        # window from the schedule's Lipschitz bound so the multiplier crosses the threshold five
        # times.
        amp, amp2 = cfg["drift_amplitude"], cfg.get("drift_amplitude2", 0.0)
        probe = DmControlDrift(
            env_name=env_name, drift_targets=targets, schedule=cfg["drift_schedule"],
            amplitude=amp, period=int(cfg["drift_period"]), amplitude2=amp2,
            period2=cfg.get("drift_period2", 30720), max_episode_steps=50)
        lip = probe.lipschitz_constant()
        probe.close()
        tol = float(cfg.get("dmc_reload_tol", 0.005))
        needed = int(np.ceil(5.0 * tol / lip)) if lip > 0 else n_steps
        window = max(n_steps, needed)
        print(f"  schedule moves the multiplier {lip:.3e}/step; {window:,} steps needed to clear "
              f"the {tol} rebuild threshold five times over")

        still = drift_trace(0.0, 0.0, window)
        moving = drift_trace(amp, amp2, window)
        divergence = float(np.abs(moving - still).max())
        # Only asks whether the drift does anything; the size of the effect is the dynamic-range
        # gate's job.
        ok = divergence > 1e-6
        print(f"  amplitude {amp} + {amp2} vs a motionless copy: max|dqpos| = {divergence:.6f}"
              f"   {_ok(ok)}")
        if not ok:
            print("           The drift schedule is not moving this environment's physics at all.")
        return ok

    def trace(mult):
        env = DmControlDrift(env_name=env_name, drift_targets=targets,
                             task_multipliers=[mult, 999.0], max_episode_steps=n_steps + 10)
        env.set_task(0)
        env.reset(seed=7)
        rng = np.random.RandomState(11)
        out = []
        for _ in range(n_steps):
            env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
            out.append(np.array(env._dm_env.physics.data.qpos, copy=True))
        env.close()
        return np.asarray(out)

    base = trace(1.0)
    passed = True
    for mult in sorted({m for m in cfg["task_multipliers"] if m != 1.0}):
        divergence = float(np.abs(trace(mult) - base).max())
        ok = divergence > threshold
        passed &= ok
        print(f"  x{mult:<4} max|dqpos| vs nominal = {divergence:8.4f}   {_ok(ok)}")
    if not passed:
        print("           An INERT parameter looks exactly like a working experiment: it runs, it "
              "logs a\n           multiplier, and it measures nothing. Choose a different physics "
              "parameter for this\n           environment - do not shrug and run the sweep.")
    return passed


# Config drift against a known-good reference config.
#
# Diffs the live effective config against a reference config that produced a working run and
# prints every difference. Differences can be intended, but they should be seen before a sweep.
# Keys in DANGEROUS have each invalidated a sweep before and are flagged with the reason.
DANGEROUS = {
    "log_std_init": "sigma level; 1.0 is heavy exploration noise for HalfCheetah",
    "anneal_lr": "true silently overwrites pt's Robbins-Monro schedule and kills rm_power",
    "freeze_log_std": "must be identical on every arm or the comparison is confounded with sigma",
    "rho": "the transfer split",
    "k": "consolidation cadence",
    "hidden_sizes": "capacity",
    "critic_hidden_sizes": "capacity",
    "lr_perm": "alpha_P; at 1e-5/sgd the permanent is inert",
    "rm_power": "Theorem 5 requires a decreasing alpha_P",
}


def check_config_drift(agent="pt", reference="archive/phase1/configs/stage14_pt.yaml"):
    """Print every key where the live config differs from the reference config. Never fails."""
    import yaml
    print(f"\n=== 5. CONFIG DRIFT vs {os.path.basename(reference)} ===")
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), reference)
    if not os.path.exists(path):
        print(f"  reference not found at {path} - skipping")
        return True
    with open(path) as f:
        ref = yaml.safe_load(f) or {}
    live = _cfg_for(agent)
    risky = []
    for key in sorted(set(ref) | set(DANGEROUS)):
        a, b = ref.get(key, "(absent)"), live.get(key, "(absent)")
        if a == b:
            continue
        mark = "  <-- " + DANGEROUS[key] if key in DANGEROUS else ""
        print(f"  {key:26s} reference={str(a):18s} live={str(b)}{mark}")
        if key in DANGEROUS:
            risky.append(key)
    if risky:
        print(f"  {len(risky)} difference(s) on settings that have previously invalidated a sweep: "
              f"{', '.join(risky)}")
        print("  This is NOT a failure - the current benchmark changes some of these "
              "deliberately. It is a\n  prompt to confirm every one of them is intended BEFORE spending hours.")
    else:
        print("  no differences on the high-risk settings")
    return True


# Smoke: all three agents train and their mechanisms fire

def _run_training(agent, steps, seed, results_dir, extra=(), overlay=None):
    cmd = [sys.executable, "-m", "src_continuous_control.train",
           "--agent", agent, "--seed", str(seed),
           "--total-steps", str(steps), "--switch", str(max(steps // 5, 1)),
           "--no-wandb", "--no-tb", "--results-dir", results_dir,
           "--runs-dir", results_dir, *extra]
    if overlay:
        cmd += ["--config", overlay]
    print(f"  $ {' '.join(cmd[2:])}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
    return proc


def check_smoke(steps, results_dir, seed=0, overlays=None):
    """Run each agent briefly; check it completes and that pt's and EWC's mechanisms fire."""
    print(f"\n=== 1. SMOKE ({steps:,} steps x 3 agents) ===")
    overlays = overlays or {n: None for n in AGENT_NAMES}
    passed = True
    for name in AGENT_NAMES:
        # Transfer matrix off: it costs a flat ~250k evaluation steps and the smoke test does
        # not use it.
        proc = _run_training(name, steps, seed, f"{results_dir}/{name}",
                             extra=["--transfer-eval-episodes", "0"],
                             overlay=overlays.get(name))
        trained = proc.returncode == 0 and "[train] Done." in proc.stdout
        print(f"  {name:<8} {_ok(trained)}: ran to completion")
        passed &= trained
        if not trained:
            continue
        if name == "pt":
            passed &= _check_absorption(f"{results_dir}/{name}", seed)
        if name == "ewc":
            passed &= _check_ewc_penalty(f"{results_dir}/{name}", proc.stdout)
    return passed


def _load_pickle(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _check_absorption(results_dir, seed):
    """Check actor_absorbed_frac > 0.01; below that the permanent policy is inert."""
    import glob
    hits = glob.glob(f"{results_dir}/*_consolidation_records.pkl")
    if not hits:
        print("  pt       **FAIL**: no consolidation records - the mechanism never ran")
        return False
    records = _load_pickle(hits[0])
    fracs = [r["actor_absorbed_frac"] for r in records if r["actor_absorbed_frac"] is not None]
    critic = [r["absorbed_frac"] for r in records if r["absorbed_frac"] is not None]
    if not fracs:
        print("  pt       **FAIL**: actor_absorbed_frac never recorded")
        return False
    worst = min(fracs)
    passed = worst > 0.01
    print(f"  pt       {_ok(passed)}: actor_absorbed_frac min={worst:.4f} "
          f"mean={np.mean(fracs):.4f} over {len(records)} consolidations "
          f"(critic mean={np.mean(critic):.4f} over {len(critic)})")
    if not passed:
        print("           Below 0.01 the permanent is INERT: PT has no slow timescale and this "
              "arm\n           measures nothing. Raise lr_perm / consolidation_epochs before the "
              "sweep.")
    return passed


def _check_ewc_penalty(results_dir, stdout):
    """Check the logged EWC penalty is non-zero after a boundary.

    Without a boundary no Fisher is accumulated and EWC reduces to vanilla PPO. Reads the logged
    `train/ewc_penalty`, not `ewc_lambda`.
    """
    import glob
    switched = "SWITCH to task" in stdout
    print(f"  ewc      {_ok(switched)}: saw at least one task boundary")
    hits = glob.glob(f"{results_dir}/*_scalars.pkl")
    if not hits:
        print("  ewc      **FAIL**: no scalars pickle to read the penalty from")
        return False
    history = _load_pickle(hits[0])
    series = history.get("train/ewc_penalty")
    if series is None or len(series) == 0:
        print("  ewc      **FAIL**: train/ewc_penalty was never logged")
        return False
    values = np.asarray(series, dtype=np.float64)[:, 1]
    nonzero = values[values > 0]
    passed = switched and nonzero.size > 0
    print(f"  ewc      {_ok(passed)}: penalty non-zero on {nonzero.size}/{values.size} updates "
          f"(max={values.max():.3e})")
    if not passed:
        print("           A penalty that is identically zero means no Fisher was accumulated: "
              "the EWC\n           arm is vanilla PPO and the comparison says nothing.")
    return passed


# Dynamic range

def check_dynamic_range(total_steps, results_dir, seed=0, threshold=0.20, overlay=None):
    """Run one full-length vanilla run and report its dynamic range."""
    print(f"\n=== 2. DYNAMIC RANGE (vanilla, {total_steps:,} steps) ===")
    cfg = _cfg_for("vanilla", overlay)
    switch = int(cfg["switch"])
    proc = _run_training("vanilla", total_steps, seed, results_dir,
                         extra=["--switch", str(switch)], overlay=overlay)
    if proc.returncode != 0:
        print("  **FAIL**: the run did not complete")
        return False
    return report_dynamic_range(results_dir, switch, threshold)


def report_dynamic_range(results_dir, switch, threshold=0.20):
    """Check the physics change moves the return enough for the methods to be separable.

    Two measurements:
      (a) plateau return per task along the training curve. Learning progress inflates this
          spread even in a stationary environment, so it is printed for context only.
      (b) one frozen policy evaluated on every physics setting (a row of the transfer matrix).
          Training progress is fixed, so the spread comes from the physics alone. This is the
          gate: it passes if some row's spread is at least `threshold` of its mean.

    (b) needs a run with transfer_eval_episodes > 0; without it the check fails.
    """
    import glob
    # `*_returns.pkl` also matches `*_ep_returns.pkl` and `*_eval_returns.pkl`, which are flat
    # lists of episode returns rather than (step, value) rows.
    hits = [h for h in glob.glob(f"{results_dir}/*_returns.pkl")
            if not os.path.basename(h).endswith(("_ep_returns.pkl", "_eval_returns.pkl"))]
    if not hits:
        print("  **FAIL**: no returns pickle found")
        return False
    curve = np.asarray(_load_pickle(hits[0]), dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] < 2:
        print(f"  **FAIL**: {hits[0]} is not a (step, return) curve")
        return False
    steps, rets = curve[:, 0], curve[:, 1]
    n_phases = max(int(np.ceil(steps[-1] / switch)), 1)
    # A step landing exactly on total_steps would open an extra final bucket; clamp it.
    task = np.minimum((steps // switch).astype(int), n_phases - 1)
    print("  (a) plateau return per phase along the training curve "
          "[CONTAMINATED BY LEARNING, context only]")
    per_task = []
    for i in sorted(set(task)):
        sel = task == i
        # Last 25% of each task: the plateau, not the dip right after a switch.
        tail = rets[sel][int(0.75 * sel.sum()):]
        per_task.append(float(np.mean(tail)) if len(tail) else float("nan"))
        print(f"      phase {i}: {per_task[-1]:.1f}")
    naive = (max(per_task) - min(per_task)) / max(abs(np.mean(per_task)), 1e-9)
    print(f"      spread = {naive * 100:.1f}% of the mean (an improving agent inflates this)")

    hits = glob.glob(f"{results_dir}/*_transfer_matrix.pkl")
    if not hits:
        print("  (b) NOT AVAILABLE: no transfer matrix in this run, so the physics-only spread "
              "could not be\n      measured. Re-run with transfer_eval_episodes > 0; (a) alone "
              "does not answer the question.")
        return False
    matrix = np.asarray(_load_pickle(hits[0])["transfer_matrix"], dtype=np.float64)
    print("  (b) one FROZEN policy across every physics setting [the gate]")
    spreads = []
    for i, row in enumerate(matrix):
        if np.isnan(row).any():
            continue
        spread = (row.max() - row.min()) / max(abs(row.mean()), 1e-9)
        spreads.append(spread)
        print(f"      policy after task {i}: {row.min():8.1f} .. {row.max():8.1f}   "
              f"spread = {spread * 100:5.1f}%")
    if not spreads:
        print("      **FAIL**: the transfer matrix is incomplete")
        return False
    best = max(spreads)
    passed = best >= threshold
    print(f"  {_ok(passed)}: the physics move a fixed policy's return by up to "
          f"{best * 100:.0f}% (need >= {threshold * 100:.0f}%)")
    if not passed:
        print("           The physics change is too small for the benchmark to separate the "
              "methods.\n           Raise the amplitude of task_multipliers before spending hours "
              "on a sweep.")
    return passed


def _main_multienv(args):
    """Run every gate against each selected dm_control environment and print one summary table.

    An environment that fails the non-inert gate needs a different physics parameter; one that
    fails the dynamic-range gate should be dropped from the study.
    """
    names = args.multienv or list(MULTIENV_BENCHMARKS)
    names = [n if n.startswith("multienv:") else "multienv:" + n for n in names]
    unknown = [n for n in names if n not in BENCHMARKS]
    if unknown:
        print("unknown environment(s): %s\nvalid: %s"
              % (unknown, [n.split(":", 1)[1] for n in MULTIENV_BENCHMARKS]))
        return 1

    results = {}
    for bench in names:
        spec = BENCHMARKS[bench]
        obs_dim, act_dim, overlays = spec["obs_dim"], spec["act_dim"], spec["overlays"]
        env_name = bench.split(":", 1)[1]
        print(f"\n{'=' * 94}\n=== ENVIRONMENT: {env_name}  (obs {obs_dim}, act {act_dim}) ===")
        for name in AGENT_NAMES:
            print(f"  {name:<8} config = default.yaml <- ppo_{name}.yaml <- {overlays[name]}.yaml")
        per_env = {
            "parameter parity": check_parameter_parity(
                obs_dim, act_dim, tolerance=MULTIENV_PARITY_TOLERANCE, overlays=overlays),
            "sigma parity": check_sigma_parity(obs_dim, act_dim, overlays=overlays),
            "physics change": check_physics_change(bench, overlays=overlays),
            "non-inert": check_non_inert(bench, overlays=overlays),
        }
        if not args.skip_smoke:
            per_env["smoke"] = check_smoke(
                args.smoke_steps, f"{args.results_dir}/multienv/{env_name}/smoke",
                args.seed, overlays=overlays)
        if args.dynamic_range:
            total = args.total_steps or int(
                _cfg_for("vanilla", overlays["vanilla"])["total_steps"])
            per_env["dynamic range"] = check_dynamic_range(
                total, f"{args.results_dir}/multienv/{env_name}/range", args.seed,
                overlay=overlays["vanilla"])
        results[env_name] = per_env

    print(f"\n{'=' * 94}\n=== FAMILY SUMMARY ===")
    gates = sorted({g for per_env in results.values() for g in per_env})
    print("  %-22s %s" % ("environment", "  ".join(f"{g:>16s}" for g in gates)))
    for env_name, per_env in results.items():
        cells = "  ".join(f"{('PASS' if per_env[g] else 'FAIL') if g in per_env else '-':>16s}"
                          for g in gates)
        print(f"  {env_name:<22s} {cells}")
    failed = [(e, g) for e, per_env in results.items() for g, ok in per_env.items() if not ok]
    if failed:
        print("\nDO NOT START THE SWEEP:")
        for env_name, gate in failed:
            print(f"  {env_name}: {gate}")
        return 1
    print("\nAll gates passed on %d environment(s)." % len(results))
    return 0


def main():
    p = argparse.ArgumentParser(description="Pre-flight checks")
    p.add_argument("--smoke-steps", type=int, default=60000)
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--dynamic-range", action="store_true",
                   help="also run the full-length vanilla run (check 2). Takes hours.")
    p.add_argument("--total-steps", type=int, default=None,
                   help="length of the dynamic-range run; defaults to the config's total_steps")
    p.add_argument("--results-dir", type=str, default="src_continuous_control/results/preflight")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cartpole", action="store_true",
                   help="run every check against the cartpole-swingup benchmark (obs 5 / act 1, "
                        "the cartpole_* overlays) instead of HalfCheetah")
    p.add_argument("--multienv", nargs="*", default=None, metavar="ENV",
                   help="run every check against the dm_control FAMILY. With no arguments, all "
                        "six environments; otherwise the named ones (e.g. --multienv walker-walk "
                        "cheetah-run). A family study with one environment ungated answers "
                        "nothing, so the default is all of them.")
    args = p.parse_args()

    if args.multienv is not None:
        return _main_multienv(args)

    bench = "cartpole" if args.cartpole else "halfcheetah"
    spec = BENCHMARKS[bench]
    obs_dim, act_dim, overlays = spec["obs_dim"], spec["act_dim"], spec["overlays"]
    print(f"=== BENCHMARK: {bench}  (obs {obs_dim}, act {act_dim}) ===")
    for name in AGENT_NAMES:
        print(f"  {name:<8} config = default.yaml <- ppo_{name}.yaml"
              + (f" <- {overlays[name]}.yaml" if overlays[name] else ""))

    results = {}
    results["parameter parity"] = check_parameter_parity(obs_dim, act_dim, overlays=overlays)
    results["sigma parity"] = check_sigma_parity(obs_dim, act_dim, overlays=overlays)
    results["physics change"] = check_physics_change(bench, overlays=overlays)
    if not args.cartpole:
        # The reference config is a HalfCheetah run, so diffing cartpole against it is noise.
        check_config_drift()
    if not args.skip_smoke:
        results["smoke"] = check_smoke(args.smoke_steps, f"{args.results_dir}/smoke", args.seed,
                                       overlays=overlays)
    if args.dynamic_range:
        total = args.total_steps or int(_cfg_for("vanilla", overlays["vanilla"])["total_steps"])
        results["dynamic range"] = check_dynamic_range(
            total, f"{args.results_dir}/range", args.seed, overlay=overlays["vanilla"])

    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"  {_ok(passed)}  {name}")
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\nDO NOT START THE SWEEP: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
