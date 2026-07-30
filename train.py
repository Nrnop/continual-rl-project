"""Training entry point for the continuous-control PT extension.

Usage:
    python -m src_continuous_control.train --agent pt   --seed 0
    python -m src_continuous_control.train --agent vanilla --seed 0

    # Quick smoke test
    python -m src_continuous_control.train --agent pt --seed 0 \
        --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb

Config resolution: default.yaml ← agent-specific YAML ← CLI overrides.
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import yaml
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from .agents import AGENTS
from .envs.directional_half_cheetah import (
    DirectionalHalfCheetah,
    make_directional_env,
    make_vector_env,
)
from .envs.drift_half_cheetah import make_drift_env, make_drift_vector_env
from .utils.logger import Logger
from .utils.metrics import ValueDriftProbe, BoundaryReturnTracker
from .utils.seeding import seed_everything, seed_env


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
_CFG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_config(cli_args):
    """Merge default.yaml ← agent-specific YAML ← --config overlay ← CLI overrides."""
    cfg = _load_yaml(os.path.join(_CFG_DIR, "default.yaml"))

    agent_yaml = os.path.join(_CFG_DIR, f"ppo_{cli_args.agent}.yaml")
    if os.path.exists(agent_yaml):
        cfg.update(_load_yaml(agent_yaml))

    # Optional extra overlay (e.g. --config cleanrl_match) — applied on top of the
    # agent YAML so a single file can pin a full experiment's hyper-params.
    extra = getattr(cli_args, "config", None)
    if extra:
        if not extra.endswith(".yaml"):
            extra = extra + ".yaml"
        path = extra if os.path.isabs(extra) else os.path.join(_CFG_DIR, extra)
        if os.path.exists(path):
            cfg.update(_load_yaml(path))
        else:
            raise FileNotFoundError(f"--config overlay not found: {path}")

    # CLI overrides (only non-None values). 'config' is a meta-key consumed above.
    for key, val in vars(cli_args).items():
        if key == "config":
            continue
        if val is not None:
            cfg[key] = val

    # Ensure agent is set
    cfg.setdefault("agent", cli_args.agent)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Continuous-control PT-PPO training")
    p.add_argument("--agent", type=str, default="pt", choices=list(AGENTS.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=str, default=None,
                   help="extra YAML overlay in configs/ (e.g. cleanrl_match)")

    # Env
    p.add_argument("--env-id", type=str, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--switch", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--num-envs", type=int, default=None,
                   help="parallel envs (batch = n_steps * num_envs)")
    p.add_argument("--async-envs", type=lambda v: str(v).lower() in ("true","1","yes","y"),
                   nargs="?", const=True, default=None,
                   help="use AsyncVectorEnv (subprocesses) when num_envs>1")

    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if str(v).lower() in ("true", "1", "yes", "t", "y"):
            return True
        elif str(v).lower() in ("false", "0", "no", "f", "n"):
            return False
        return bool(v)

    p.add_argument("--step-by-step", type=_str2bool, nargs="?", const=True, default=None)

    # PPO
    p.add_argument("--lr-actor", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--minibatch-size", type=int, default=None)
    p.add_argument("--clip-coef", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--target-kl", type=float, default=None)

    # PT-specific
    p.add_argument("--lr-trans", type=float, default=None)
    p.add_argument("--lr-perm", type=float, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--decay", type=float, default=None)
    p.add_argument("--lr-critic", type=float, default=None)

    # EWC-specific
    p.add_argument("--ewc-lambda", type=float, default=None)
    p.add_argument("--ewc-gamma", type=float, default=None)

    # Logging & evaluation
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-tb", action="store_true")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default=None)
    p.add_argument("--eval-interval-updates", type=int, default=None)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--disable-task-switch", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--render-freq", type=int, default=None)

    args = p.parse_args()

    # Normalise hyphenated CLI keys → underscored config keys
    raw = vars(args)
    normalised = {}
    for k, v in raw.items():
        normalised[k.replace("-", "_")] = v
    return argparse.Namespace(**normalised)


def _run_offline_eval(agent, eval_env, n_episodes=5):
    """Run standardized zero-momentum offline evaluation from stationary standstill."""
    eval_returns = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                if hasattr(agent, "actor"):
                    action, _ = agent.actor.act(obs_t)
                else:
                    action, _ = agent.select_action(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            obs, r, term, trunc, info = eval_env.step(action_np)
            # True reward even if a reward normalizer is present upstream.
            ep_ret += info.get("directional_reward", r)
            done = term or trunc
        eval_returns.append(ep_ret)
    return float(np.mean(eval_returns))


# ---------------------------------------------------------------------------
# Helper functions for traversing wrappers (e.g. RecordVideo) cleanly
# ---------------------------------------------------------------------------
def _set_env_task(env, direction):
    """Find the DirectionalHalfCheetah wrapper across any stack and call set_task."""
    if hasattr(env, "set_task"):
        return env.set_task(direction)
    if hasattr(env, "get_wrapper_attr"):
        try:
            return env.get_wrapper_attr("set_task")(direction)
        except AttributeError:
            pass
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
        if hasattr(cur, "set_task"):
            return cur.set_task(direction)
    return getattr(env.unwrapped, "set_task", lambda d: None)(direction)


def _find_normalize_obs(env):
    """Return the NormalizeObservation wrapper in the stack, or None.

    Matches both the single-env and vector variants so the training (vector) stats
    can be copied onto the single-env eval wrapper — both expose obs_rms with the
    same per-feature (obs_dim,) mean/var/count.
    """
    norm_types = (gym.wrappers.NormalizeObservation, gym.wrappers.vector.NormalizeObservation)
    cur = env
    while cur is not None:
        if isinstance(cur, norm_types):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _sync_obs_stats(train_env, eval_env):
    """Copy the training obs running-mean/std onto the eval env and freeze it.

    The policy was trained on observations normalized by the *training* stats;
    evaluating with a different normalizer would inject a distribution shift. We
    copy the stats and stop the eval env from updating its own (CleanRL evaluates
    with the training normalizer's statistics).
    """
    src = _find_normalize_obs(train_env)
    dst = _find_normalize_obs(eval_env)
    if src is None or dst is None:
        return
    dst.obs_rms.mean = np.array(src.obs_rms.mean, copy=True)
    dst.obs_rms.var = np.array(src.obs_rms.var, copy=True)
    dst.obs_rms.count = src.obs_rms.count
    dst.update_running_mean = False


def _drift_multiplier(cfg, t):
    """The drift multiplier at global env step t — same formula as the wrapper, no env access."""
    amp = float(cfg.get("drift_amplitude", 0.5))
    period = max(int(cfg.get("drift_period", 1228800)), 1)
    if str(cfg.get("drift_schedule", "sin")).lower() == "sin":
        m = 1.0 + amp * float(np.sin(2.0 * np.pi * t / period + float(cfg.get("drift_phase", 0.0))))
    else:
        m = 1.0 + amp * (t / period)
    amp2 = float(cfg.get("drift_amplitude2", 0.0))
    if amp2:
        p2 = max(int(cfg.get("drift_period2", 30720)), 1)
        m += amp2 * float(np.sin(2.0 * np.pi * t / p2 + float(cfg.get("drift_phase2", 0.0))))
    return m


def _drift_lipschitz(cfg):
    """Max |change in the multiplier| per env step: the eps of ||P_{t+1} - P_t|| <= eps."""
    amp = abs(float(cfg.get("drift_amplitude", 0.5)))
    period = max(int(cfg.get("drift_period", 1228800)), 1)
    slow = amp * 2.0 * np.pi / period if str(cfg.get("drift_schedule", "sin")).lower() == "sin"         else amp / period
    amp2 = abs(float(cfg.get("drift_amplitude2", 0.0)))
    fast = amp2 * 2.0 * np.pi / max(int(cfg.get("drift_period2", 30720)), 1)
    return slow + fast


def _sync_drift_clock(eval_env, t):
    """Put the eval env at the same point of the drift schedule as the training env.

    The drift clock counts global env steps and never resets, so an eval env that started at t=0
    would be evaluated on DIFFERENT physics than the policy is currently training on — which would
    silently confound every eval point. Copy the training clock across and re-apply the drift.
    """
    cur = eval_env
    while cur is not None:
        if hasattr(cur, "multiplier") and hasattr(cur, "_apply"):
            cur.t = t
            cur._apply(cur.multiplier())
            return
        cur = getattr(cur, "env", None)


def _get_env_direction(env):
    """Find the running direction across any wrapper stack."""
    if hasattr(env, "direction"):
        return env.direction
    if hasattr(env, "get_wrapper_attr"):
        try:
            return env.get_wrapper_attr("direction")
        except AttributeError:
            pass
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
        if hasattr(cur, "direction"):
            return cur.direction
    return getattr(env.unwrapped, "direction", 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    cli = parse_args()
    cfg = build_config(cli)

    seed = cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    # --- Environment (vectorized) ---
    render_mode = "rgb_array" if cfg.get("render", False) else None
    normalize_obs = cfg.get("normalize_obs", False)
    normalize_reward = cfg.get("normalize_reward", False)
    num_envs = int(cfg.get("num_envs", 1))
    async_envs = cfg.get("async_envs", True)
    if cfg.get("render", False) and num_envs > 1:
        print("[train] render is not supported with num_envs>1; disabling train-env video.")

    # env_mode selects WHERE the non-stationarity lives:
    #   "directional" (default) -- the REWARD flips sign at discrete task boundaries
    #   "drift"                 -- the REWARD is fixed and the PHYSICS drift smoothly, with no
    #                              boundaries at all (the setting the proposal specifies)
    env_mode = str(cfg.get("env_mode", "directional")).lower()
    drift_mode = env_mode == "drift"
    if env_mode not in ("directional", "drift"):
        raise ValueError(f"unknown env_mode {env_mode!r}; valid: 'directional', 'drift'")

    drift_kwargs = dict(
        drift_targets=tuple(cfg.get("drift_targets", ["damping", "friction"])),
        amplitude=cfg.get("drift_amplitude", 0.5),
        period=cfg.get("drift_period", 1228800),
        schedule=cfg.get("drift_schedule", "sin"),
        phase=cfg.get("drift_phase", 0.0),
        amplitude2=cfg.get("drift_amplitude2", 0.0),
        period2=cfg.get("drift_period2", 30720),
        phase2=cfg.get("drift_phase2", 0.0),
    )
    if drift_mode:
        env = make_drift_vector_env(
            env_id=cfg.get("env_id", "HalfCheetah-v5"),
            num_envs=num_envs,
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            gamma=cfg["gamma"],
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            clip_obs=cfg.get("clip_obs", 10.0),
            clip_reward=cfg.get("clip_reward", 10.0),
            asynchronous=async_envs,
            **drift_kwargs,
        )
    else:
        env = make_vector_env(
            env_id=cfg.get("env_id", "HalfCheetah-v5"),
            num_envs=num_envs,
            direction=cfg["tasks"][0],
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            gamma=cfg["gamma"],
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            clip_obs=cfg.get("clip_obs", 10.0),
            clip_reward=cfg.get("clip_reward", 10.0),
            asynchronous=async_envs,
        )
    obs, _ = env.reset(seed=seed)
    done = np.zeros(num_envs, dtype=np.float32)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    # --- Agent ---
    AgentClass = AGENTS[cfg["agent"]]
    agent = AgentClass(obs_dim, act_dim, cfg, device)

    # --- Logger ---
    backend = "none"
    if not cfg.get("no_wandb", False) and not cfg.get("no_tb", False):
        backend = "both"
    elif cfg.get("no_wandb", False) and not cfg.get("no_tb", False):
        backend = "tb"
    elif not cfg.get("no_wandb", False) and cfg.get("no_tb", False):
        backend = "wandb"

    exp_name = f"{cfg['agent']}_ppo"
    logger = Logger(
        exp_name=exp_name,
        seed=seed,
        backend=backend,
        runs_dir=cfg.get("runs_dir", "src_continuous_control/runs"),
        results_dir=cfg.get("results_dir", "src_continuous_control/results"),
        wandb_project=cfg.get("wandb_project", "pt-continuous-control"),
        config=cfg,
    )

    # --- Metrics ---
    tasks = cfg["tasks"]
    switch_interval = cfg["switch"]
    total_steps = cfg["total_steps"]
    n_steps = cfg["n_steps"]

    drift_probe = ValueDriftProbe()
    boundary_tracker = BoundaryReturnTracker(post_window_steps=n_steps * 5)
    probe_states = None  # sampled lazily

    # --- Training state ---  (obs/done initialized from env.reset above)
    avg_return = 0.0
    global_step = 0
    update_idx = 0
    task_idx = 0

    returns_curve = []
    all_episode_returns = []
    eval_returns_curve = []
    velocity_curve = []

    eval_env = None
    if not cfg.get("no_eval", False):
        eval_env = make_drift_env(
            env_id=cfg.get("env_id", "HalfCheetah-v5"),
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            render_mode=render_mode,
            normalize_obs=normalize_obs,
            normalize_reward=False,
            clip_obs=cfg.get("clip_obs", 10.0),
            **drift_kwargs,
        ) if drift_mode else make_directional_env(
            env_id=cfg.get("env_id", "HalfCheetah-v5"),
            direction=cfg["tasks"][0],
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            render_mode=render_mode,
            normalize_obs=normalize_obs,      # stats synced from train env before each eval
            normalize_reward=False,           # eval reports the true (un-normalized) return
            clip_obs=cfg.get("clip_obs", 10.0),
        )
        if cfg.get("render", False):
            video_folder_eval = os.path.join(cfg.get("runs_dir", "src_continuous_control/runs"), "videos", f"{cfg['agent']}_seed_{seed}_eval")
            eval_env = RecordVideo(
                eval_env,
                video_folder=video_folder_eval,
                episode_trigger=lambda ep_id: ep_id % 5 == 0,
                disable_logger=True,
            )
        seed_env(eval_env, seed + 10000)

    print(f"[train] agent={cfg['agent']}  seed={seed}  device={device}")
    print(f"[train] total_steps={total_steps}  switch={switch_interval}  n_steps={n_steps}  "
          f"num_envs={num_envs}  async={async_envs and num_envs > 1}  batch={n_steps * num_envs}")
    print(f"[train] step_by_step={cfg.get('step_by_step', False)}")
    if drift_mode:
        print(f"[train] env_mode=drift  targets={drift_kwargs['drift_targets']}  "
              f"amplitude={drift_kwargs['amplitude']}  period={drift_kwargs['period']}  "
              f"schedule={drift_kwargs['schedule']}")
        print(f"[train] Lipschitz bound on the drift multiplier: "
              f"{_drift_lipschitz(cfg):.3e} per env step "
              f"({cfg['total_steps'] / max(int(cfg.get('drift_period', 1)), 1):.2f} full cycles)")
    if cfg.get("disable_task_switch", False):
        print("[train] Task switching disabled (Single-Task Baseline)")
    if cfg.get("render", False):
        print(f"[train] Video rendering enabled (render_freq={cfg.get('render_freq', 25)})")
    t0 = time.time()

    anneal_lr = cfg.get("anneal_lr", False)
    steps_per_update = n_steps * num_envs
    num_updates = max(total_steps // steps_per_update, 1)

    while global_step < total_steps:
        # ---- LR annealing (CleanRL: linear decay to 0 over training) ----
        if anneal_lr:
            frac = max(1.0 - update_idx / num_updates, 0.0)
            agent.anneal_lr(frac)

        # ---- Task switching ----
        if not drift_mode and not cfg.get("disable_task_switch", False):
            next_switch = (task_idx + 1) * switch_interval
            if global_step >= next_switch:
                task_idx += 1
                direction = tasks[task_idx % len(tasks)]
                env.unwrapped.call("set_task", direction)  # propagates to every sub-env
                agent.on_task_switch(global_step)

                # Value-drift measurement at boundary
                if probe_states is not None:
                    def val_fn(s):
                        t = torch.as_tensor(s, dtype=torch.float32, device=device)
                        with torch.no_grad():
                            if hasattr(agent.critic, "value"):
                                return agent.critic.value(t).cpu().numpy()
                            else:
                                return agent.critic(t).cpu().numpy()
                        return np.zeros(len(s))
                    drift = drift_probe.snapshot(val_fn, probe_states)
                    if drift is not None:
                        logger.log_scalar("boundary/value_drift", drift, global_step)

                boundary_tracker.on_switch(global_step, avg_return)
                print(f"[train] step {global_step}: SWITCH to task {direction}  avg_return={avg_return:.1f}")

        # ---- Collect rollout ----
        obs, done, episode_returns = agent.collect_rollout(env, obs, done)
        all_episode_returns.extend(episode_returns)
        global_step += steps_per_update

        # Track velocity
        velocity_curve.append((global_step, float(np.mean(agent._velocities))))

        # Track episodic returns (EMA, mirrors baseline's avg_return = 0.99 * avg_return + 0.01 * epi_return)
        ema = cfg.get("eval_ema", 0.99)
        for ep_ret in episode_returns:
            avg_return = ema * avg_return + (1 - ema) * ep_ret

        returns_curve.append((global_step, avg_return))

        # ---- PPO update (batch PPO over the flattened rollout) ----
        metrics = agent.update(obs, done, update_idx)
        update_idx += 1

        # ---- Zero-momentum offline evaluation & checkpointing ----
        eval_interval = cfg.get("eval_interval_updates")
        if eval_interval is None:
            eval_interval = 50
        if eval_env is not None and (update_idx % eval_interval == 0 or update_idx == 1):
            if drift_mode:
                _sync_drift_clock(eval_env, global_step)  # evaluate on the CURRENT physics
            else:
                _set_env_task(eval_env, tasks[task_idx % len(tasks)])  # match current train task
            _sync_obs_stats(env, eval_env)  # evaluate with the training normalizer's stats
            clean_eval_ret = _run_offline_eval(agent, eval_env, n_episodes=5)
            eval_returns_curve.append((global_step, clean_eval_ret))
            logger.log_scalar("eval/zero_momentum_return", clean_eval_ret, global_step)
            if cfg.get("save_checkpoints", False):
                logger.save_checkpoint(agent.state_dict(), step=global_step)

        # ---- Logging ----
        if update_idx % cfg.get("log_interval_updates", 1) == 0:
            scalars = {
                "train/avg_return": avg_return,
                "train/actor_loss": metrics["actor_loss"],
                "train/critic_loss": metrics["critic_loss"],
                "train/entropy": metrics["entropy"],
                "train/approx_kl": metrics["approx_kl"],
                "train/global_step": global_step,
            }
            if "ewc_penalty" in metrics:
                scalars["train/ewc_penalty"] = metrics["ewc_penalty"]
            # PT only: % change in the acting value across the last consolidation (0 = preserved).
            consol_err = getattr(agent, "last_consolidation_error", None)
            if consol_err is not None:
                scalars["train/consolidation_error_pct"] = consol_err
            consol_ho = getattr(agent, "last_consolidation_error_holdout", None)
            if consol_ho is not None:
                scalars["train/consolidation_error_holdout_pct"] = consol_ho
            # Consolidation-regression diagnostics (PT, separate-trunk variant only).
            for attr, tag in (("last_consolidation_loss_first", "consol/loss_first"),
                              ("last_consolidation_loss_last", "consol/loss_last"),
                              ("last_consolidation_loss_mean", "consol/loss_mean"),
                              ("last_perm_mean_before", "consol/perm_mean_before"),
                              ("last_perm_mean_after", "consol/perm_mean_after"),
                              ("last_perm_l2_before", "consol/perm_l2_before"),
                              ("last_perm_l2_after", "consol/perm_l2_after"),
                              ("last_trans_mean_before", "consol/trans_mean_before"),
                              ("last_trans_mean_after", "consol/trans_mean_after"),
                              ("last_trans_l2_before", "consol/trans_l2_before"),
                              ("last_trans_l2_after", "consol/trans_l2_after")):
                val = getattr(agent, attr, None)
                if val is not None:
                    scalars[f"train/{tag}"] = val
            if drift_mode:
                # Record where on the drift schedule we are, so return curves can be read against
                # the physics that produced them.
                scalars["drift/multiplier"] = _drift_multiplier(cfg, global_step)
            logger.log_scalars(scalars, step=global_step)

        # Sample probe states lazily (first rollout provides good coverage)
        if probe_states is None:
            flat_obs = agent.buffer.obs.reshape(-1, obs_dim)
            n_probe = min(cfg.get("probe_states", 256), flat_obs.shape[0])
            probe_states = flat_obs[:n_probe].copy()

        # Boundary return tracker
        rec = boundary_tracker.update(global_step, avg_return)
        if rec is not None:
            logger.log_scalars({
                "boundary/return_drop": rec["drop"],
                "boundary/pre_return": rec["pre"],
                "boundary/post_trough": rec["trough"],
            }, step=rec["step"])

        # Progress
        elapsed = time.time() - t0
        sps = global_step / max(elapsed, 1)
        if update_idx % 10 == 0:
            print(f"[train] step {global_step}/{total_steps}  "
                  f"return={avg_return:.1f}  sps={sps:.0f}  "
                  f"actor_loss={metrics['actor_loss']:.4f}  "
                  f"critic_loss={metrics['critic_loss']:.4f}")

    # --- Finalize ---
    fname = logger.save_returns(returns_curve)
    if all_episode_returns:
        logger.save_returns(all_episode_returns, suffix="ep_returns")
    if eval_returns_curve:
        logger.save_returns(eval_returns_curve, suffix="eval_returns")
    if velocity_curve:
        logger.save_returns(velocity_curve, suffix="velocities")
    mean_drop = boundary_tracker.mean_drop()
    if mean_drop is not None:
        logger.log_scalar("boundary/mean_drop", mean_drop, global_step)
        print(f"[train] mean boundary drop: {mean_drop:.2f}")

    if hasattr(env, "close"):
        env.close()
    if eval_env is not None and hasattr(eval_env, "close"):
        eval_env.close()
    logger.close()
    elapsed = time.time() - t0
    print(f"[train] Done. {global_step} steps in {elapsed:.0f}s ({global_step/elapsed:.0f} sps). "
          f"Returns saved to {fname}")


if __name__ == "__main__":
    main()
