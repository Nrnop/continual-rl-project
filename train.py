"""Training entry point for the continuous-control PT extension.

Usage:
    python -m src_continuous_control.train --agent pt   --seed 0
    python -m src_continuous_control.train --agent vanilla --seed 0

    # Quick smoke test
    python -m src_continuous_control.train --agent pt --seed 0 \
        --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb

Run from the parent directory of the package. Config layers, later wins:
default.yaml, ppo_<agent>.yaml, --config overlay, CLI flags.
"""
import argparse
import contextlib
import copy
import os
import random
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
from .envs.cartpole_swingup import make_cartpole_env, make_cartpole_vector_env
from .envs.dm_control_drift import SPECS as DMC_SPECS, make_dmc_env, make_dmc_vector_env
from .utils.logger import Logger
from .utils.metrics import (ValueDriftProbe, BoundaryReturnTracker,
                            JumpstartTracker, RetentionProbe, TransferMatrix,
                            evaluate_policy_on_tasks)
from .utils.seeding import seed_everything, seed_env


# Config helpers
_CFG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_config(cli_args):
    """Merge default.yaml, the agent YAML, the --config overlay and CLI overrides, in that order."""
    cfg = _load_yaml(os.path.join(_CFG_DIR, "default.yaml"))

    agent_yaml = os.path.join(_CFG_DIR, f"ppo_{cli_args.agent}.yaml")
    if os.path.exists(agent_yaml):
        cfg.update(_load_yaml(agent_yaml))

    # Optional overlay applied on top of the agent YAML, so one file can pin an experiment.
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

    # The CLI agent wins over any `agent` field in an overlay.
    cfg["agent"] = cli_args.agent
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
    p.add_argument("--normalizer-freeze-after", type=int, default=None,
                   help="freeze observation/reward normalizer statistics after this many env steps")
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
    # alpha_P needs tuning per domain: the paper's search ranges span seven orders of magnitude
    # (tabular 0.8..1e-3, deep prediction 1e-3..3e-5, MinAtar 1e-7..1e-9).
    p.add_argument("--perm-optimizer", type=str, default=None, choices=["sgd", "adam"])
    p.add_argument("--lr-perm-actor", type=float, default=None)
    p.add_argument("--consolidation-epochs", type=int, default=None)
    # Robbins-Monro annealing of alpha_P, required by Theorem 5 (rm_power = 0 makes it constant).
    p.add_argument("--rm-power", type=float, default=None)
    p.add_argument("--alpha-p-rm-power", type=float, default=None, help="alias for --rm-power")
    # The permanent absorbs rho and the transient keeps (1 - rho), which preserves the composed
    # output. --decay-rho decouples the two halves and is meant only as a control.
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--decay-rho", type=float, default=None)
    # "realised": decay the transient by what the permanent actually absorbed, (1 - c*rho) per
    # output unit, instead of the fixed (1 - rho). The YAML default is "fixed".
    p.add_argument("--decay-match", type=str, default=None, choices=["fixed", "realised"])
    p.add_argument("--log-adv-split", action="store_true", default=None,
                   help="log the advantage split by value component (Table 4.1); changes no training")
    p.add_argument("--kl-prior-coef", type=float, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--lr-critic", type=float, default=None)

    # Env non-stationarity
    p.add_argument("--drift-schedule", type=str, default=None,
                   choices=["step", "sin", "linear"],
                   help="'step' = physics change at observable boundaries; "
                        "'sin'/'linear' = continuous drift, no boundaries")
    p.add_argument("--task-multipliers", type=float, nargs="+", default=None,
                   help="drift_schedule=step: one physics multiplier per task, cycled")

    # EWC-specific
    p.add_argument("--ewc-lambda", type=float, default=None)
    p.add_argument("--ewc-gamma", type=float, default=None)

    # Logging & evaluation
    #
    # Every store_true flag below needs default=None. build_config only overrides the YAML with
    # non-None CLI values, so default=False would overwrite the YAML and these keys could not be
    # set from a config file. Consumers read them with cfg.get(key, False).
    p.add_argument("--no-wandb", action="store_true", default=None)
    p.add_argument("--no-tb", action="store_true", default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default=None)
    p.add_argument("--eval-interval-updates", type=int, default=None)
    # Episodes per cell of the transfer matrix (0 disables it) and per side of the decay probe.
    # Both add environment steps on top of training.
    p.add_argument("--transfer-eval-episodes", type=int, default=None)
    p.add_argument("--decay-gain-episodes", type=int, default=None)
    p.add_argument("--no-eval", action="store_true", default=None)
    p.add_argument("--save-checkpoints", action="store_true", default=None)
    p.add_argument("--disable-task-switch", action="store_true", default=None)
    p.add_argument("--render", action="store_true", default=None)
    p.add_argument("--render-freq", type=int, default=None)

    args = p.parse_args()

    # Hyphenated CLI keys to underscored config keys.
    raw = vars(args)
    normalised = {}
    for k, v in raw.items():
        normalised[k.replace("-", "_")] = v
    return argparse.Namespace(**normalised)


@contextlib.contextmanager
def _isolated_rng():
    """Run a block without disturbing the torch, numpy and random RNG states.

    Evaluation samples from the global torch generator, which also produces training actions.
    Restoring the RNG state afterwards keeps training trajectories identical with eval on or off.
    """
    t_state = torch.get_rng_state()
    n_state = np.random.get_state()
    p_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(t_state)
        np.random.set_state(n_state)
        random.setstate(p_state)


def _run_offline_eval(agent, eval_env, n_episodes=5):
    """Mean return of the stochastic policy over `n_episodes`, each started from reset.

    Wrap in `_isolated_rng()`: this samples actions from the global torch generator.
    """
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
            # True reward from info, unaffected by any reward normalizer.
            ep_ret += info.get("directional_reward", r)
            done = term or trunc
        eval_returns.append(ep_ret)
    return float(np.mean(eval_returns))


# Evaluation probes and wrapper-traversal helpers
def _run_deterministic_eval(agent, env, n_episodes=3, max_steps=1000):
    """Mean return of the deterministic policy mean, with no gradient steps.

    Wrap in `_isolated_rng()`: env.reset() consumes randomness.
    """
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action = agent.actor.act_deterministic(obs_t).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(info.get("directional_reward", reward))
            done = bool(terminated or truncated)
            steps += 1
        returns.append(total)
    return float(np.mean(returns))


def _decay_gain_probe(agent, probe_env, n_episodes=3, max_steps=1000):
    """Effect of the transient decay alone: evaluate, decay mu_T, evaluate again.

    There is no gradient step in between, so the difference comes from the decay operator only.
    It is 0 for a split critic (the critic does not act), so it measures what the split actor adds.
    mu_T's output layer is snapshotted and restored, leaving the run unchanged.
    Returns {"before", "after", "gain"}, or None for agents without a split actor.
    """
    trans = getattr(getattr(agent, "actor", None), "trans_mean", None)
    if trans is None or not hasattr(agent.actor, "decay_transient"):
        return None
    with _isolated_rng():
        before = _run_deterministic_eval(agent, probe_env, n_episodes, max_steps)
        snapshot = (trans[-1].weight.detach().clone(), trans[-1].bias.detach().clone())
        agent.actor.decay_transient(1.0 - agent.decay_rho)
        try:
            after = _run_deterministic_eval(agent, probe_env, n_episodes, max_steps)
        finally:
            with torch.no_grad():
                trans[-1].weight.copy_(snapshot[0])
                trans[-1].bias.copy_(snapshot[1])
    return {"before": before, "after": after, "gain": after - before}


def _actor_perm_trans_corr(agent, states_np, max_states=1024):
    """Correlation between mu_P and mu_T over visited states.

    Near -1 means the two components cancel, and a two-component ablation would show no effect.
    """
    actor = getattr(agent, "actor", None)
    if actor is None or not hasattr(actor, "trans_mean"):
        return None
    states = torch.as_tensor(states_np[:max_states], dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        mu_p = actor.perm_forward(states).flatten()
        mu_t = actor.trans_mean(states).flatten()
    if float(mu_p.std()) < 1e-8 or float(mu_t.std()) < 1e-8:
        return None            # mu_T is still zero (init), so the correlation is undefined
    stacked = torch.stack([mu_p, mu_t])
    return float(torch.corrcoef(stacked)[0, 1])


def _set_task_id_obs(env, task_idx):
    """Set the one-hot task label on the TaskIDObservation wrapper. Returns True if found.

    Walks the wrapper stack, since the wrapper may sit under RecordVideo or a vector wrapper.
    """
    cur = env
    for _ in range(16):
        if hasattr(cur, "set_task_id"):
            cur.set_task_id(task_idx)
            return True
        cur = getattr(cur, "env", None)
        if cur is None:
            return False
    return False


def _set_env_task(env, task):
    """Find the wrapper that owns `set_task` anywhere in the stack and call it.

    `DirectionalHalfCheetah.set_task(direction)` flips the reward sign;
    `LipschitzDriftHalfCheetah.set_task(i)` selects task i's physics. RecordVideo and the vector
    wrappers hide the inner env, so the stack is walked explicitly.
    """
    if hasattr(env, "set_task"):
        return env.set_task(task)
    if hasattr(env, "get_wrapper_attr"):
        try:
            return env.get_wrapper_attr("set_task")(task)
        except AttributeError:
            pass
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
        if hasattr(cur, "set_task"):
            return cur.set_task(task)
    return getattr(env.unwrapped, "set_task", lambda d: None)(task)


def _find_normalize_obs(env):
    """Return the NormalizeObservation wrapper in the stack, or None.

    Matches the single-env and vector variants, which both expose obs_rms with per-feature
    (obs_dim,) mean/var, so training stats can be copied onto the eval env.
    """
    norm_types = (gym.wrappers.NormalizeObservation, gym.wrappers.vector.NormalizeObservation)
    cur = env
    while cur is not None:
        if isinstance(cur, norm_types):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _normalizer_wrappers(env):
    """Return observation and reward normalizers in a wrapper stack."""
    obs_types = (gym.wrappers.NormalizeObservation, gym.wrappers.vector.NormalizeObservation)
    reward_types = (gym.wrappers.NormalizeReward, gym.wrappers.vector.NormalizeReward)
    found = []
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, obs_types):
            found.append(("observation", current))
        elif isinstance(current, reward_types):
            found.append(("reward", current))
        current = getattr(current, "env", None)
    return found


def _normalizer_snapshot(kind, wrapper):
    stats = getattr(wrapper, "obs_rms", None)
    if stats is None:
        stats = getattr(wrapper, "return_rms", None)
    if stats is None:
        return {f"{kind}_count": 0.0}
    mean = np.asarray(getattr(stats, "mean", 0.0), dtype=np.float64)
    var = np.asarray(getattr(stats, "var", 0.0), dtype=np.float64)
    return {
        f"{kind}_mean_l2": float(np.linalg.norm(mean)),
        f"{kind}_var_l2": float(np.linalg.norm(var)),
        f"{kind}_count": float(getattr(stats, "count", 0.0)),
    }


def _freeze_normalizers(env):
    """Freeze running statistics and return a numeric snapshot for logging."""
    snapshot = {}
    for kind, wrapper in _normalizer_wrappers(env):
        wrapper.update_running_mean = False
        snapshot.update(_normalizer_snapshot(kind, wrapper))
    return snapshot


def _normalizer_freeze_due(global_step, threshold):
    """Return whether a nonnegative freeze threshold has been reached."""
    return threshold is not None and int(threshold) >= 0 and global_step >= int(threshold)


def _sync_obs_stats(train_env, eval_env):
    """Copy the training observation mean/var onto the eval env and freeze it.

    The policy expects observations normalized with the training statistics (as in CleanRL).
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
    """Drift multiplier at global env step t, using the wrapper's formula without the env."""
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
    """Set the eval env's drift clock to global step t and re-apply the drift.

    The drift clock counts global env steps and never resets, so without this the eval env would
    run on different physics from the training env.
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


# Main training loop
def main():
    cli = parse_args()
    cfg = build_config(cli)

    seed = cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    # Environment (vectorized)
    render_mode = "rgb_array" if cfg.get("render", False) else None
    normalize_obs = cfg.get("normalize_obs", False)
    normalize_reward = cfg.get("normalize_reward", False)
    num_envs = int(cfg.get("num_envs", 1))
    async_envs = cfg.get("async_envs", True)
    if cfg.get("render", False) and num_envs > 1:
        print("[train] render is not supported with num_envs>1; disabling train-env video.")

    # env_mode selects the source of non-stationarity:
    #   "drift" (default): HalfCheetah physics change, reward fixed. drift_schedule="step" changes
    #                      them at observable boundaries; "sin"/"linear" drift continuously.
    #   "directional":     HalfCheetah reward-direction flip.
    #   "cartpole":        dm_control cartpole-swingup, a physics multiplier per task, cycled.
    #                      Reward is in [0, 1] per step, so the return ceiling is 1000.
    #                      See envs/cartpole_swingup.py.
    #   "dmc":             one of the dm_control family, selected by `dmc_env`, same structure.
    #                      See envs/dm_control_drift.py.
    env_mode = str(cfg.get("env_mode", "drift")).lower()
    if env_mode not in ("directional", "drift", "cartpole", "dmc"):
        raise ValueError(
            f"unknown env_mode {env_mode!r}; valid: 'directional', 'drift', 'cartpole', 'dmc'")
    cartpole_mode = env_mode == "cartpole"
    dmc_mode = env_mode == "dmc"
    # `dmc` requires an explicit `dmc_env`; there is no default environment.
    dmc_env_name = str(cfg.get("dmc_env", "")) if dmc_mode else ""
    if dmc_mode:
        if dmc_env_name not in DMC_SPECS:
            raise ValueError(
                f"env_mode 'dmc' needs `dmc_env` set to one of {tuple(DMC_SPECS)}; "
                f"got {dmc_env_name!r}")
    # drift_mode: the non-stationarity is a physics multiplier indexed by task. Task labels,
    # boundary bookkeeping, the transfer matrix and the decay-gain probe are shared across these
    # modes; only env construction differs.
    drift_mode = env_mode in ("drift", "cartpole", "dmc")

    # Default drift targets per benchmark. For `dmc` they come from the environment's spec, since
    # the parameter that matters differs per body (e.g. ball mass on ball_in_cup).
    if dmc_mode:
        _default_targets = list(DMC_SPECS[dmc_env_name].default_targets)
    elif cartpole_mode:
        _default_targets = ["pole_length", "pole_mass"]
    else:
        _default_targets = ["damping", "friction"]

    drift_kwargs = dict(
        drift_targets=tuple(cfg.get("drift_targets", _default_targets)),
        amplitude=cfg.get("drift_amplitude", 0.5),
        period=cfg.get("drift_period", 1228800),
        schedule=cfg.get("drift_schedule", "sin"),
        phase=cfg.get("drift_phase", 0.0),
        amplitude2=cfg.get("drift_amplitude2", 0.0),
        period2=cfg.get("drift_period2", 30720),
        phase2=cfg.get("drift_phase2", 0.0),
        # schedule="step" only: one multiplier per task, cycled.
        task_multipliers=tuple(cfg.get("task_multipliers", [1.0, 1.6, 0.6, 1.6, 0.6])),
    )
    if dmc_mode:
        env = make_dmc_vector_env(
            env_name=dmc_env_name,
            num_envs=num_envs,
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            gamma=cfg["gamma"],
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            clip_obs=cfg.get("clip_obs", 10.0),
            clip_reward=cfg.get("clip_reward", 10.0),
            asynchronous=async_envs,
            reload_tol=cfg.get("dmc_reload_tol", 0.005),
            task_id_obs=bool(cfg.get("task_id_obs", False)),
            n_task_ids=len(drift_kwargs["task_multipliers"]),
            **drift_kwargs,
        )
    elif cartpole_mode:
        env = make_cartpole_vector_env(
            task_name=cfg.get("cartpole_task", "swingup"),
            num_envs=num_envs,
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            gamma=cfg["gamma"],
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            clip_obs=cfg.get("clip_obs", 10.0),
            clip_reward=cfg.get("clip_reward", 10.0),
            asynchronous=async_envs,
            reload_tol=cfg.get("cartpole_reload_tol", 0.005),
            task_id_obs=bool(cfg.get("task_id_obs", False)),
            n_task_ids=len(drift_kwargs["task_multipliers"]),
            **drift_kwargs,
        )
    elif drift_mode:
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
            task_id_obs=bool(cfg.get("task_id_obs", False)),
            n_task_ids=len(cfg["tasks"]),
        )
    obs, _ = env.reset(seed=seed)
    done = np.zeros(num_envs, dtype=np.float32)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    # The number of trailing task-label dims is derived inside the agent from `task_id_obs` and
    # `tasks`, not passed in through cfg.

    # Agent
    AgentClass = AGENTS[cfg["agent"]]
    agent = AgentClass(obs_dim, act_dim, cfg, device)

    # Logger
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

    # Metrics
    tasks = cfg["tasks"]
    switch_interval = cfg["switch"]
    total_steps = cfg["total_steps"]
    n_steps = cfg["n_steps"]

    # Task boundaries. Observable boundaries correspond to the paper's semi-continual setting.
    # Drift modes have boundaries only with the piecewise-constant "step" schedule.
    drift_schedule = str(drift_kwargs["schedule"]).lower()
    boundaries_enabled = (not cfg.get("disable_task_switch", False)
                          and (not drift_mode or drift_schedule == "step"))
    _task_multipliers = list(drift_kwargs["task_multipliers"])

    def _task_label(idx):
        """Identifier for the task at index `idx`, shared by every revisit of the same physics.

        With multipliers [1.0, 1.6, 0.6, 1.6, 0.6], tasks 1/3 and 2/4 share a label. Retention and
        backward transfer rely on this.
        """
        if drift_mode:
            return _task_multipliers[idx % len(_task_multipliers)]
        return tasks[idx % len(tasks)]

    n_tasks = len(_task_multipliers) if drift_mode else len(tasks)

    def _task_arg(idx):
        """Argument to `set_task` for task `idx`: the index (drift) or reward direction."""
        return idx if drift_mode else tasks[idx % len(tasks)]

    def _task_physics_note(idx):
        if drift_mode:
            return f" (physics x{_task_label(idx):g})"
        return f" (reward direction {_task_label(idx)})"

    drift_probe = ValueDriftProbe()
    # Window in whole PPO updates (batch = n_steps * num_envs); see BoundaryReturnTracker.
    boundary_window = int(cfg.get("boundary_window_updates", 5)) * n_steps * num_envs
    boundary_tracker = BoundaryReturnTracker(post_window_steps=boundary_window,
                                             min_useful_steps=n_steps * num_envs)
    # Theorem 8's advantage sits right after a switch, so this window is short relative to a task
    # (default 20 updates).
    jumpstart_window = int(cfg.get("jumpstart_window_updates", 20)) * n_steps * num_envs
    jumpstart_tracker = JumpstartTracker(window_steps=jumpstart_window)
    retention_probe = RetentionProbe()
    probe_states = None  # sampled lazily

    def _v_perm(s):
        return agent.get_value(s)[0]

    def _v_full(s):
        vp, vt = agent.get_value(s)
        return vp + vt

    # Both are scored against the acting value at the end of each finished task (Theorem 7).
    # Vanilla and EWC put the whole value in the perm slot and zero in trans, so the two coincide.
    _value_fns = {"perm": _v_perm, "full": _v_full}

    # Training state (obs/done come from env.reset above)
    avg_return = 0.0
    global_step = 0
    update_idx = 0
    task_idx = 0

    returns_curve = []
    consol_loss_traces = []      # PT only: (global_step, within-consolidation loss curve)
    _n_consol_seen = 0
    all_episode_returns = []
    eval_returns_curve = []
    velocity_curve = []
    actor_consol_loss_traces = []
    normalizers_frozen = False

    eval_env = None
    if not cfg.get("no_eval", False):
        if dmc_mode:
            eval_env = make_dmc_env(
                env_name=dmc_env_name,
                max_episode_steps=cfg.get("max_episode_steps", 1000),
                render_mode=render_mode,
                normalize_obs=normalize_obs,
                normalize_reward=False,       # eval reports the true (un-normalized) return
                clip_obs=cfg.get("clip_obs", 10.0),
                reload_tol=cfg.get("dmc_reload_tol", 0.005),
                # Must match the training env's observation layout exactly.
                task_id_obs=bool(cfg.get("task_id_obs", False)),
                n_task_ids=len(drift_kwargs["task_multipliers"]),
                **drift_kwargs,
            )
        elif cartpole_mode:
            eval_env = make_cartpole_env(
            task_name=cfg.get("cartpole_task", "swingup"),
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            render_mode=render_mode,
            normalize_obs=normalize_obs,
            normalize_reward=False,           # eval reports the true (un-normalized) return
            clip_obs=cfg.get("clip_obs", 10.0),
            reload_tol=cfg.get("cartpole_reload_tol", 0.005),
            # Must match the training env's observation layout exactly.
            task_id_obs=bool(cfg.get("task_id_obs", False)),
            n_task_ids=len(drift_kwargs["task_multipliers"]),
            **drift_kwargs,
            )
        elif drift_mode:
            eval_env = make_drift_env(
            env_id=cfg.get("env_id", "HalfCheetah-v5"),
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            render_mode=render_mode,
            normalize_obs=normalize_obs,
            normalize_reward=False,
            clip_obs=cfg.get("clip_obs", 10.0),
            **drift_kwargs,
            )
        else:
            eval_env = make_directional_env(
            env_id=cfg.get("env_id", "HalfCheetah-v5"),
            direction=cfg["tasks"][0],
            max_episode_steps=cfg.get("max_episode_steps", 1000),
            render_mode=render_mode,
            normalize_obs=normalize_obs,      # stats synced from train env before each eval
            normalize_reward=False,           # eval reports the true (un-normalized) return
            clip_obs=cfg.get("clip_obs", 10.0),
            # Must match the training env's observation layout exactly.
            task_id_obs=bool(cfg.get("task_id_obs", False)),
            n_task_ids=len(cfg["tasks"]),
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

    # probe/decay_gain (see _decay_gain_probe). Uses its own env so it never touches the eval env,
    # which may be wrapped in RecordVideo.
    probe_env = None
    if (drift_mode and cfg.get("decay_gain_probe", True)
            and hasattr(agent, "decay_rho") and not cfg.get("no_eval", False)):
        if dmc_mode:
            probe_env = make_dmc_env(
                env_name=dmc_env_name,
                max_episode_steps=cfg.get("max_episode_steps", 1000),
                normalize_obs=normalize_obs, normalize_reward=False,
                clip_obs=cfg.get("clip_obs", 10.0),
                reload_tol=cfg.get("dmc_reload_tol", 0.005),
                task_id_obs=bool(cfg.get("task_id_obs", False)),
                n_task_ids=len(drift_kwargs["task_multipliers"]),
                **drift_kwargs)
        elif cartpole_mode:
            probe_env = make_cartpole_env(
                task_name=cfg.get("cartpole_task", "swingup"),
                max_episode_steps=cfg.get("max_episode_steps", 1000),
                normalize_obs=normalize_obs, normalize_reward=False,
                clip_obs=cfg.get("clip_obs", 10.0),
                reload_tol=cfg.get("cartpole_reload_tol", 0.005),
                task_id_obs=bool(cfg.get("task_id_obs", False)),
                n_task_ids=len(drift_kwargs["task_multipliers"]),
                **drift_kwargs)
        else:
            probe_env = make_drift_env(
                env_id=cfg.get("env_id", "HalfCheetah-v5"),
                max_episode_steps=cfg.get("max_episode_steps", 1000),
                normalize_obs=normalize_obs, normalize_reward=False,
                clip_obs=cfg.get("clip_obs", 10.0), **drift_kwargs)
        seed_env(probe_env, seed + 20000)

    # Forward / backward transfer (Lopez-Paz & Ranzato 2017).
    # R[i, j] = mean return on task j with the policy frozen at the end of task i. Rows are filled
    # at boundaries. 0 episodes disables it.
    transfer_episodes = int(cfg.get("transfer_eval_episodes", 10))
    transfer = None
    if boundaries_enabled and transfer_episodes > 0 and eval_env is not None:
        transfer = TransferMatrix(n_tasks)

    def _transfer_row():
        """Mean return and standard error on every task for the current deterministic policy.

        Runs on the eval env, since `set_task` changes physics. The eval env is put back on the
        current task afterwards, and the RNG is isolated from training.
        """
        _sync_obs_stats(env, eval_env)

        def _act(obs_np):
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                return agent.actor.act_deterministic(obs_t).cpu().numpy()[0]

        with _isolated_rng():
            return evaluate_policy_on_tasks(
                _act, eval_env,
                lambda j: (_set_env_task(eval_env, _task_arg(j)),
                           _set_task_id_obs(eval_env, j) if cfg.get("task_id_obs", False) else None),
                n_tasks, n_episodes=transfer_episodes,
                max_steps=cfg.get("max_episode_steps", 1000),
                # A task index; the set_task lambda maps it through _task_arg.
                restore_task=task_idx % n_tasks, return_std=True,
            )

    if transfer is not None:
        # b_j: the random-init policy's return on each task, measured before the first update.
        transfer.set_baselines(_transfer_row()[0])
        print(f"[train] transfer baselines (random init) b_j = "
              f"{np.array2string(transfer.baselines, precision=1)}")

    print(f"[train] agent={cfg['agent']}  seed={seed}  device={device}")
    print(f"[train] total_steps={total_steps}  switch={switch_interval}  n_steps={n_steps}  "
          f"num_envs={num_envs}  async={async_envs and num_envs > 1}  batch={n_steps * num_envs}")
    print(f"[train] step_by_step={cfg.get('step_by_step', False)}")
    print(f"[train] grad_clip={'JOINT (actor+critic together)' if cfg.get('joint_grad_clip', False) else 'separate (actor | critic)'}")
    if cfg["agent"] == "pt":
        # Print the settings that decide whether the PT mechanism is active.
        _rho = float(cfg.get("rho", cfg.get("transfer_rate", 0.5)))
        _rm = float(cfg.get("rm_power", cfg.get("alpha_p_rm_power", 0.6)))
        _decay_rho = float(cfg.get("decay_rho", _rho))
        print(f"[train] PT: rho={_rho} k={cfg.get('k')} "
              f"lr_perm={cfg.get('lr_perm')} lr_perm_actor={cfg.get('lr_perm_actor')} "
              f"opt={cfg.get('perm_optimizer')} rm_power={_rm} "
              f"kl_prior_coef={cfg.get('kl_prior_coef')} "
              f"critic_hidden={cfg.get('critic_hidden_sizes', cfg.get('hidden_sizes'))}")
        # Warn about settings that change what the run measures.
        if _rm <= 0.0:
            print("[train] PT:   ^ CONSTANT alpha_P (Robbins-Monro OFF) - Theorem 5's premise "
                  "is not satisfied")
        if _decay_rho != _rho:
            print(f"[train] PT:   ^ decay_rho={_decay_rho} != rho={_rho} - the absorb and the "
                  "decay are DECOUPLED, so the transfer is not composition-preserving and the "
                  "acting value jumps at every consolidation. Watch for divergence.")
    if drift_mode:
        _sched = str(drift_kwargs["schedule"])
        print(f"[train] env_mode={env_mode}  targets={drift_kwargs['drift_targets']}  "
              f"schedule={_sched}")
        if cartpole_mode:
            # Reward is in [0, 1] over max_episode_steps steps with no early termination, so the
            # return ceiling is known.
            print(f"[train] cartpole task={cfg.get('cartpole_task', 'swingup')}  "
                  f"return ceiling = {cfg.get('max_episode_steps', 1000)} "
                  f"(reward in [0,1], no early termination)")
        if dmc_mode:
            # Same ceiling across the family. Log the environment name, since all six share one
            # code path.
            _spec = DMC_SPECS[dmc_env_name]
            print(f"[train] dm_control env={dmc_env_name}  obs={_spec.obs_dim} act={_spec.act_dim}"
                  f"  return ceiling = {cfg.get('max_episode_steps', 1000)} "
                  f"(reward in [0,1], no early termination)")
        if _sched == "step":
            # Piecewise-constant physics: the multiplier is a function of the task index.
            print(f"[train] task_multipliers={list(drift_kwargs['task_multipliers'])} "
                  f"(cycled; switch every {switch_interval} steps)")
        else:
            print(f"[train] amplitude={drift_kwargs['amplitude']}  "
                  f"period={drift_kwargs['period']}")
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
        # LR annealing (CleanRL: linear decay to 0 over training)
        if anneal_lr:
            frac = max(1.0 - update_idx / num_updates, 0.0)
            agent.anneal_lr(frac)

        # Task switching
        if boundaries_enabled:
            next_switch = (task_idx + 1) * switch_interval
            if global_step >= next_switch:
                # Snapshot the finishing task's values (Theorem 7's v_i) before on_task_switch,
                # which consolidates and decays and so changes V.
                if probe_states is not None:
                    retention_probe.snapshot(_task_label(task_idx), _v_full, probe_states)
                # R[i, :] is the policy at the end of task i, so measure it before the switch.
                if transfer is not None:
                    transfer.add_row(task_idx, *_transfer_row())

                task_idx += 1
                # `directional` passes the reward direction; drift modes pass the task index, which
                # selects that task's physics (the reward function is fixed).
                switch_arg = task_idx if drift_mode else tasks[task_idx % len(tasks)]
                env.unwrapped.call("set_task", switch_arg)  # propagates to every sub-env

                # The task-label wrapper sits on the outer vector env, which `call` above does not
                # reach, so set it here.
                if cfg.get("task_id_obs", False):
                    if not _set_task_id_obs(env, task_idx):
                        raise RuntimeError(
                            "task_id_obs is on but no TaskIDObservation wrapper was found - the "
                            "observation would carry a task label that never changes")

                # Optionally reset at the boundary so each task starts from a defined state rather
                # than the body's state at the switch. The drift clock survives the reset, and the
                # in-flight episode is discarded (RecordEpisodeStatistics reports no return for
                # it). Off by default; the reported HalfCheetah runs did not reset.
                if cfg.get("reset_on_task_switch", False):
                    obs, _ = env.reset()
                    done = np.zeros(num_envs, dtype=np.float32)

                agent.on_task_switch(global_step)

                # Value drift at the boundary
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

                # Effect of the decay alone, measured after the switch on the new task's physics.
                if probe_env is not None:
                    _sync_obs_stats(env, probe_env)
                    _set_env_task(probe_env, _task_arg(task_idx))
                    rec = _decay_gain_probe(agent, probe_env,
                                            n_episodes=int(cfg.get("decay_gain_episodes", 3)))
                    if rec is not None:
                        logger.log_scalars({"probe/decay_gain": rec["gain"],
                                            "probe/decay_before": rec["before"],
                                            "probe/decay_after": rec["after"]}, global_step)
                        print(f"[train]   probe/decay_gain = {rec['gain']:+.1f} "
                              f"({rec['before']:.1f} -> {rec['after']:.1f}, no gradient step)")

                boundary_tracker.on_switch(global_step, avg_return)
                jumpstart_tracker.on_switch(global_step)
                print(f"[train] step {global_step}: SWITCH to task {task_idx}"
                      f"{_task_physics_note(task_idx)}  avg_return={avg_return:.1f}")

        # Collect rollout
        obs, done, episode_returns = agent.collect_rollout(env, obs, done)
        all_episode_returns.extend(episode_returns)
        global_step += steps_per_update

        freeze_after = cfg.get("normalizer_freeze_after")
        if not normalizers_frozen and _normalizer_freeze_due(global_step, freeze_after):
            normalizer_snapshot = _freeze_normalizers(env)
            normalizers_frozen = True
            print(f"[train] normalizers frozen at step {global_step} "
                  f"(threshold={int(freeze_after)}) {normalizer_snapshot}", flush=True)
            logger.log_scalars(
                {f"normalizer/{name}": value for name, value in normalizer_snapshot.items()},
                global_step,
            )

        fresh_drift = None
        measure_drift = getattr(agent, "measure_post_consolidation_drift", None)
        if measure_drift is not None:
            fresh_drift = measure_drift(agent.buffer.obs.reshape(-1, obs_dim))
            if fresh_drift is not None:
                logger.log_scalars({
                    "consol/delta_v": fresh_drift["delta_v"],
                    "consol/delta_pi": fresh_drift["delta_pi"],
                }, global_step)

        # HalfCheetah reports x_velocity in info; other envs leave the list empty.
        if agent._velocities:
            velocity_curve.append((global_step, float(np.mean(agent._velocities))))

        # EMA of episodic returns: avg_return = 0.99 * avg_return + 0.01 * epi_return
        ema = cfg.get("eval_ema", 0.99)
        for ep_ret in episode_returns:
            avg_return = ema * avg_return + (1 - ema) * ep_ret

        returns_curve.append((global_step, avg_return))

        # PPO update over the flattened rollout
        metrics = agent.update(obs, done, update_idx)
        update_idx += 1

        # PT only: keep new consolidation loss traces, tagged with the global step.
        _curves = getattr(agent, "consolidation_loss_curves", None)
        if _curves is not None and len(_curves) > _n_consol_seen:
            for c in _curves[_n_consol_seen:]:
                consol_loss_traces.append((global_step, np.asarray(c, dtype=np.float32)))
            _n_consol_seen = len(_curves)
        _actor_curves = getattr(agent, "actor_consolidation_loss_curves", None)
        if _actor_curves is not None:
            seen_actor = len(actor_consol_loss_traces)
            for c in _actor_curves[seen_actor:]:
                actor_consol_loss_traces.append((global_step, np.asarray(c, dtype=np.float32)))

        # Periodic evaluation and checkpointing
        eval_interval = cfg.get("eval_interval_updates")
        if eval_interval is None:
            eval_interval = 50
        if eval_env is not None and (update_idx % eval_interval == 0 or update_idx == 1):
            if drift_mode and drift_schedule == "step":
                _set_env_task(eval_env, task_idx)        # evaluate on this task's physics
            elif drift_mode:
                _sync_drift_clock(eval_env, global_step)  # evaluate on the current physics
            else:
                _set_env_task(eval_env, tasks[task_idx % len(tasks)])  # match current train task
                # Keep the eval task label in sync with the task.
                if cfg.get("task_id_obs", False):
                    _set_task_id_obs(eval_env, task_idx)
            _sync_obs_stats(env, eval_env)  # evaluate with the training normalizer's stats
            with _isolated_rng():          # the eval must not perturb the training RNG stream
                clean_eval_ret = _run_offline_eval(agent, eval_env, n_episodes=5)
            eval_returns_curve.append((global_step, clean_eval_ret))
            logger.log_scalar("eval/zero_momentum_return", clean_eval_ret, global_step)
            if cfg.get("save_checkpoints", False):
                logger.save_checkpoint(agent.state_dict(), step=global_step)

        # Logging
        if update_idx % cfg.get("log_interval_updates", 1) == 0:
            scalars = {
                "train/avg_return": avg_return,
                "train/actor_loss": metrics["actor_loss"],
                "train/critic_loss": metrics["critic_loss"],
                "train/entropy": metrics["entropy"],
                "train/approx_kl": metrics["approx_kl"],
                "train/global_step": global_step,
            }
            corr = _actor_perm_trans_corr(agent, agent.buffer.obs.reshape(-1, obs_dim))
            if corr is not None:
                scalars["diag/actor_perm_trans_corr"] = corr
            for metric_name in (
                    "kl_prior", "clip_fraction", "value_perm_l2", "value_trans_l2",
                    "policy_perm_l2", "policy_trans_l2", "log_std_mean", "log_std_min",
                    "grad_norm_trans_actor", "grad_norm_perm_actor",
                    "grad_norm_trans_critic", "grad_norm_perm_critic",
                    # Advantage split by component (`log_adv_split: true`, Table 4.1)
                    "adv_var_perm_ratio", "adv_var_trans_ratio", "adv_var_reward_ratio",
                    "adv_corr_perm_trans", "adv_split_residual"):
                if metric_name in metrics:
                    scalars[f"train/{metric_name}"] = metrics[metric_name]
            if "ewc_penalty" in metrics:
                scalars["train/ewc_penalty"] = metrics["ewc_penalty"]
            # PT only: % change in the acting value across the last consolidation (0 = preserved).
            consol_err = getattr(agent, "last_consolidation_error", None)
            if consol_err is not None:
                scalars["train/consolidation_error_pct"] = consol_err
            consol_ho = getattr(agent, "last_consolidation_error_holdout", None)
            if consol_ho is not None:
                scalars["train/consolidation_error_holdout_pct"] = consol_ho
            # Consolidation diagnostics (PT only).
            for attr, tag in (("last_alpha_p", "consol/alpha_p"),
                              ("last_absorbed_frac", "consol/absorbed_frac"),
                              ("last_absorbed_align", "consol/absorbed_align"),
                              ("last_absorbed_frac_holdout", "consol/absorbed_frac_holdout"),
                              ("last_absorbed_align_holdout", "consol/absorbed_align_holdout"),
                              ("last_consolidation_loss_first", "consol/loss_first"),
                              ("last_consolidation_loss_last", "consol/loss_last"),
                              ("last_consolidation_loss_mean", "consol/loss_mean"),
                              ("last_actor_absorbed_frac", "consol/actor_absorbed_frac"),
                              ("last_actor_absorbed_align", "consol/actor_absorbed_align"),
                              ("last_alpha_p_actor", "consol/alpha_p_actor"),
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
                # Current physics multiplier. `_drift_multiplier` handles only the clock-driven
                # schedules; under "step" the multiplier is the task label.
                scalars["drift/multiplier"] = (
                    float(_task_label(task_idx)) if drift_schedule == "step"
                    else _drift_multiplier(cfg, global_step))
            logger.log_scalars(scalars, step=global_step)

        # Sample probe states from the first rollout.
        if probe_states is None:
            flat_obs = agent.buffer.obs.reshape(-1, obs_dim)
            n_probe = min(cfg.get("probe_states", 256), flat_obs.shape[0])
            probe_states = flat_obs[:n_probe].copy()
            # Control baselines for the retention metric (see RetentionProbe).
            retention_probe.set_baseline("perm_init", _v_perm(probe_states))
            retention_probe.set_baseline("zero", np.zeros(len(probe_states), dtype=np.float32))

        # perm/frac_of_value is the permanent's share of |V|; perm/drift_from_init is how far it
        # has moved since initialisation. Drift near 0 means the permanent is not learning.
        if probe_states is not None and update_idx % 10 == 0:
            vp_now = np.asarray(_v_perm(probe_states), dtype=np.float32)
            vt_now = np.asarray(agent.get_value(probe_states)[1], dtype=np.float32)
            denom = np.abs(vp_now).mean() + np.abs(vt_now).mean() + 1e-8
            logger.log_scalars({
                "perm/frac_of_value": float(np.abs(vp_now).mean() / denom),
                "perm/drift_from_init": float(np.sqrt(np.mean(
                    (vp_now - retention_probe.baselines["perm_init"]) ** 2))),
                "perm/abs_mean": float(np.abs(vp_now).mean()),
                "trans/abs_mean": float(np.abs(vt_now).mean()),
            }, step=global_step)

        # Boundary return tracker
        rec = boundary_tracker.update(global_step, avg_return)
        if rec is not None:
            logger.log_scalars({
                "boundary/return_drop": rec["drop"],
                "boundary/pre_return": rec["pre"],
                "boundary/post_trough": rec["trough"],
            }, step=rec["step"])

        # Jumpstart (Theorems 6 and 8): return in the short window after each switch.
        jrec = jumpstart_tracker.update(global_step, avg_return)
        if jrec is not None:
            logger.log_scalars({
                "boundary/jumpstart_first": jrec["first"],
                "boundary/jumpstart_mean": jrec["mean"],
                "boundary/jumpstart_end": jrec["end"],
                "boundary/jumpstart_gain": jrec["gain"],
            }, step=jrec["step"])

        # Retention (Theorem 7): MSE against the inactive tasks' values, for the permanent and the
        # full value. Empty until the first task has finished.
        if probe_states is not None and retention_probe.snapshots:
            ret = retention_probe.measure(_task_label(task_idx), _value_fns, probe_states)
            if ret:
                logger.log_scalars(
                    {f"retention/mse_{name}": v for name, v in ret.items()}, step=global_step)

        # Progress
        elapsed = time.time() - t0
        sps = global_step / max(elapsed, 1)
        if update_idx % 10 == 0:
            print(f"[train] step {global_step}/{total_steps}  "
                  f"return={avg_return:.1f}  sps={sps:.0f}  "
                  f"actor_loss={metrics['actor_loss']:.4f}  "
                  f"critic_loss={metrics['critic_loss']:.4f}")

    # Finalize
    if transfer is not None:
        # The last task has no boundary, so its row is measured here. If the run passed through
        # the cycle more than once, no row is left; bwt()/fwt() then return None.
        if task_idx < n_tasks:
            transfer.add_row(task_idx, *_transfer_row())
        summary = transfer.summary()
        # Two evaluation-noise estimates: the per-cell standard error and the disagreement
        # between columns of the same (revisited) task.
        summary["repeat_noise"] = transfer.repeat_noise(
            [_task_label(j) for j in range(n_tasks)])
        logger.save_object(summary, "transfer_matrix")
        print("[train] transfer matrix R[i, j] (return on task j after finishing task i):")
        print(np.array2string(summary["transfer_matrix"], precision=1, suppress_small=True))
        if summary["bwt"] is not None:
            logger.log_scalar("transfer/bwt", summary["bwt"], global_step)
            print(f"[train] BWT = {summary['bwt']:.2f}")
        if summary["fwt"] is not None:
            logger.log_scalar("transfer/fwt", summary["fwt"], global_step)
            print(f"[train] FWT = {summary['fwt']:.2f}")
        mean_se = float(np.nanmean(summary["cell_standard_errors"]))
        print(f"[train] transfer cell noise: mean SE = {mean_se:.1f}"
              + (f", repeated-task disagreement = {summary['repeat_noise']:.1f}"
                 if summary["repeat_noise"] is not None else ""))

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
    mean_js = jumpstart_tracker.mean_jumpstart()
    if mean_js is not None:
        logger.log_scalar("boundary/mean_jumpstart", mean_js, global_step)
        print(f"[train] mean post-switch jumpstart return "
              f"({cfg.get('jumpstart_window_updates', 20)}-update window): {mean_js:.2f}")
    if retention_probe.snapshots and probe_states is not None:
        final_ret = retention_probe.measure(_task_label(task_idx),
                                            _value_fns, probe_states)
        if final_ret:
            print("[train] final retention MSE vs inactive task(s): "
                  + "  ".join(f"{k}={v:.4f}" for k, v in final_ret.items()))

    # Save all logged scalars regardless of which logging backends were enabled.
    scalars_file = logger.save_scalars()
    if consol_loss_traces:
        f = logger.save_object(consol_loss_traces, "consol_loss_traces")
        print(f"[train] Consolidation loss traces ({len(consol_loss_traces)} cycles) -> {f}")
    if actor_consol_loss_traces:
        f = logger.save_object(actor_consol_loss_traces, "actor_consol_loss_traces")
        print(f"[train] Actor consolidation loss traces ({len(actor_consol_loss_traces)} cycles) -> {f}")
    records = getattr(agent, "consolidation_records", None)
    if records:
        f = logger.save_object(records, "consolidation_records")
        print(f"[train] Consolidation records ({len(records)} cycles) -> {f}")

    if hasattr(env, "close"):
        env.close()
    if eval_env is not None and hasattr(eval_env, "close"):
        eval_env.close()
    if probe_env is not None and hasattr(probe_env, "close"):
        probe_env.close()
    logger.close()
    print(f"[train] Scalars ({len(logger.history)} series) saved to {scalars_file}")
    elapsed = time.time() - t0
    print(f"[train] Done. {global_step} steps in {elapsed:.0f}s ({global_step/elapsed:.0f} sps). "
          f"Returns saved to {fname}")


if __name__ == "__main__":
    main()
