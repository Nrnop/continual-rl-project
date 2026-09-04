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

    # The explicit CLI agent wins over an overlay's descriptive/default agent field.
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
    # alpha_P must be TUNED PER DOMAIN. The paper's own search ranges span seven orders of
    # magnitude (tabular 0.8..1e-3, deep prediction 1e-3..3e-5, MinAtar 1e-7..1e-9), and at
    # HalfCheetah's value scale the inherited sgd/1e-5 transfers 0.04% per consolidation, i.e. the
    # permanent never learns. These two flags exist so the sweep can be driven from the CLI.
    p.add_argument("--perm-optimizer", type=str, default=None, choices=["sgd", "adam"])
    p.add_argument("--lr-perm-actor", type=float, default=None)
    p.add_argument("--consolidation-epochs", type=int, default=None)
    # Settable from the CLI so a missing/stale config cannot SILENTLY disable Robbins-Monro
    # annealing, which Theorem 5's premise requires (rm_power = 0 makes alpha_P constant).
    p.add_argument("--rm-power", type=float, default=None)
    p.add_argument("--alpha-p-rm-power", type=float, default=None, help="alias for --rm-power")
    # The transfer split: the permanent absorbs rho, the transient retains (1-rho). ONE knob, on
    # purpose — see CLAUDE.md. --decay-rho decouples the two halves and exists only to reproduce
    # the divergence that follows.
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--decay-rho", type=float, default=None)
    p.add_argument("--kl-prior-coef", type=float, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--lr-critic", type=float, default=None)

    # Env non-stationarity
    p.add_argument("--drift-schedule", type=str, default=None,
                   choices=["step", "sin", "linear"],
                   help="'step' = physics change at observable boundaries (Phase 2a); "
                        "'sin'/'linear' = continuous drift, no boundaries (Phase 2b)")
    p.add_argument("--task-multipliers", type=float, nargs="+", default=None,
                   help="drift_schedule=step: one physics multiplier per task, cycled")

    # EWC-specific
    p.add_argument("--ewc-lambda", type=float, default=None)
    p.add_argument("--ewc-gamma", type=float, default=None)

    # Logging & evaluation
    #
    # EVERY store_true FLAG BELOW MUST CARRY default=None. `build_config` merges the CLI over the
    # YAML with `if val is not None: cfg[key] = val`, so a flag left at argparse's usual
    # default=False is NOT absent — it is the value False, and it silently overwrites whatever the
    # config file said. The effect is that these keys CANNOT BE SET FROM YAML AT ALL.
    #
    # Found 2026-08-17 building the ceiling test: a config with `disable_task_switch: true` ran and
    # printed `SWITCH to task 1 (physics x1.6)` anyway. CLAUDE.md failure mode #1 exactly — a
    # control that was not actually on, and one that looks like a working experiment in every log
    # line except the ones nobody reads.
    #
    # No live Phase 2 config sets any of these, so no Phase 2 result is affected. Phase 1's
    # `archive/phase1/configs/cleanrl_match.yaml` DOES set `disable_task_switch: true`, so its
    # "single-task baseline" was switching tasks throughout; and ~30 archived stage configs set
    # `no_eval`, which was likewise ignored (harmless — eval runs under _isolated_rng()).
    #
    # With default=None the flag is absent unless actually passed, the YAML wins when the flag is
    # not given, and passing the flag still wins over the YAML. Every consumer reads these through
    # cfg.get(key, False), so an absent key behaves exactly as False did.
    p.add_argument("--no-wandb", action="store_true", default=None)
    p.add_argument("--no-tb", action="store_true", default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default=None)
    p.add_argument("--eval-interval-updates", type=int, default=None)
    # Episodes per cell of the transfer matrix (0 disables it) and per side of the decay probe.
    # Both are extra environment steps on top of training, so they are worth being able to trim
    # when rehearsing the pipeline.
    p.add_argument("--transfer-eval-episodes", type=int, default=None)
    p.add_argument("--decay-gain-episodes", type=int, default=None)
    p.add_argument("--no-eval", action="store_true", default=None)
    p.add_argument("--save-checkpoints", action="store_true", default=None)
    p.add_argument("--disable-task-switch", action="store_true", default=None)
    p.add_argument("--render", action="store_true", default=None)
    p.add_argument("--render-freq", type=int, default=None)

    args = p.parse_args()

    # Normalise hyphenated CLI keys → underscored config keys
    raw = vars(args)
    normalised = {}
    for k, v in raw.items():
        normalised[k.replace("-", "_")] = v
    return argparse.Namespace(**normalised)


@contextlib.contextmanager
def _isolated_rng():
    """Run a block without disturbing the training RNG streams.

    WHY THIS EXISTS. `_run_offline_eval` samples actions with `actor.act()`, which draws from the
    GLOBAL torch generator — up to 5 episodes x 1000 steps = 5000 draws, every 50 updates. Those
    draws advance the same generator that produces training actions, so the evaluation silently
    rewrote the trajectory it was supposed to be observing. Concretely: the same (agent, config,
    seed) produced different results with and without `--no-eval`, because the eval fires at
    update_idx == 1 and every subsequent training action is shifted.

    That broke reproducibility for every paired-per-seed comparison in this project whenever two
    runs were launched with different eval settings. Snapshotting and restoring the RNG state makes
    the evaluation a true read-only probe: identical training trajectories with eval on or off.
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
    """Run standardized zero-momentum offline evaluation from stationary standstill.

    Caller MUST wrap this in `_isolated_rng()` — it samples actions and would otherwise perturb
    the training RNG stream (see that context manager's docstring).
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
            # True reward even if a reward normalizer is present upstream.
            ep_ret += info.get("directional_reward", r)
            done = term or trunc
        eval_returns.append(ep_ret)
    return float(np.mean(eval_returns))


# ---------------------------------------------------------------------------
# Helper functions for traversing wrappers (e.g. RecordVideo) cleanly
# ---------------------------------------------------------------------------
def _run_deterministic_eval(agent, env, n_episodes=3, max_steps=1000):
    """Mean return of the policy MEAN — no exploration noise, no gradient steps.

    Separate from `_run_offline_eval`, which samples: a measurement that must attribute a return
    difference to one specific edit of the policy cannot have sampling noise sitting on top of it.
    Wrap in `_isolated_rng()`; env.reset() consumes randomness even though the policy does not.
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
    """What the DECAY alone does to behaviour: evaluate, decay mu_T, evaluate again.

    No gradient step in between, so the difference is caused by the decomposition's decay operator
    and nothing else. It is provably 0 for a split CRITIC (the critic does not act), so this
    isolates what the split ACTOR adds — and with the Phase 1 shrinkage control dropped it is the
    only thing that can separate "the decomposition works" from "the decay works".

    mu_T's output layer is snapshotted and restored, so the probe cannot perturb the run it is
    measuring. Returns None for agents without a split actor.
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

    Near -1 means the two components are cancelling: the composed policy is the small residue of
    two large opposed functions, and the two-component ablation will read flat no matter what the
    mechanism is doing. Worth seeing on run one rather than after the sweep.
    """
    actor = getattr(agent, "actor", None)
    if actor is None or not hasattr(actor, "trans_mean"):
        return None
    states = torch.as_tensor(states_np[:max_states], dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        mu_p = actor.perm_forward(states).flatten()
        mu_t = actor.trans_mean(states).flatten()
    if float(mu_p.std()) < 1e-8 or float(mu_t.std()) < 1e-8:
        return None            # mu_T is still exactly zero (init); the correlation is undefined
    stacked = torch.stack([mu_p, mu_t])
    return float(torch.corrcoef(stacked)[0, 1])


def _set_task_id_obs(env, task_idx):
    """Set the one-hot task label on the TaskIDObservation wrapper. Returns True if found.

    Traversed defensively for the same reason as `_set_env_task`: the wrapper may sit under a
    RecordVideo or another vector wrapper depending on the config, and a label that silently fails
    to update is indistinguishable in the logs from one that works.
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
    """Find the wrapper that owns `set_task` across any stack and call it.

    Works for both benchmarks: `DirectionalHalfCheetah.set_task(direction)` flips the reward sign,
    `LipschitzDriftHalfCheetah.set_task(i)` selects task i's physics. `RecordVideo` and the vector
    wrappers hide the inner env, hence the defensive traversal.
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
    #   "drift" (default) -- the HalfCheetah PHYSICS change; the reward is fixed. Phase 2's design.
    #                        With drift_schedule="step" they change at observable boundaries; with
    #                        "sin"/"linear" they drift continuously and there are none.
    #   "directional"     -- Phase 1's reward-flip benchmark. Retired as a design, kept runnable.
    # The two point-mass modes were retired with the rest of the point-mass benchmark (T4); they
    # are recoverable from git tag `phase1-archive`.
    #   "cartpole"        -- dm_control cartpole-swingup, the SECOND environment. Structurally
    #                        identical to "drift" (a physics multiplier per task, cycled), on a
    #                        maximally different task: 1 actuator vs 6, a point attractor vs a
    #                        gait, and a reward bounded in [0,1] so the return ceiling is exactly
    #                        1000 by construction. It exists to separate "a property of the PT
    #                        method" from "a property of HalfCheetah", which no HalfCheetah run
    #                        can do. See envs/cartpole_swingup.py.
    #   "dmc"             -- the FAMILY of dm_control benchmarks, selected by `dmc_env`. Same
    #                        structure again (a physics multiplier per task, cycled), across six
    #                        environments instead of one. Two environments can only ever give two
    #                        points, and cartpole and HalfCheetah differ in size AND task type at
    #                        once, so neither can say whether a finding belongs to the method or
    #                        to the environment. See envs/dm_control_drift.py.
    env_mode = str(cfg.get("env_mode", "drift")).lower()
    if env_mode not in ("directional", "drift", "cartpole", "dmc"):
        raise ValueError(
            f"unknown env_mode {env_mode!r}; valid: 'directional', 'drift', 'cartpole', 'dmc'")
    cartpole_mode = env_mode == "cartpole"
    dmc_mode = env_mode == "dmc"
    # Named, not defaulted silently: `dmc` without a `dmc_env` would quietly run cartpole and the
    # results directory would be the only record of which environment was actually swept.
    dmc_env_name = str(cfg.get("dmc_env", "")) if dmc_mode else ""
    if dmc_mode:
        if dmc_env_name not in DMC_SPECS:
            raise ValueError(
                f"env_mode 'dmc' needs `dmc_env` set to one of {tuple(DMC_SPECS)}; "
                f"got {dmc_env_name!r}")
    # `drift_mode` means "the non-stationarity is a physics multiplier indexed by task", which is
    # true of both physics benchmarks. Everything downstream — task labels, boundary bookkeeping,
    # the transfer matrix, the decay-gain probe — is shared, so the two envs differ only where
    # they are actually constructed.
    drift_mode = env_mode in ("drift", "cartpole", "dmc")

    # Each benchmark's own documented defaults. For `dmc` they come from the environment's spec
    # rather than from a literal here, because the parameter that is non-inert differs per body —
    # pole length on cartpole, ball mass on ball_in_cup, joint damping on cheetah.
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
        # schedule="step" only: one multiplier per task, cycled (see LipschitzDriftHalfCheetah).
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
    # NOTE: how many trailing dims are the task label is derived inside the agent (ppo_pt.py) from
    # `task_id_obs` + `tasks`, deliberately NOT injected into cfg from here — see the comment there.

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

    # --- Task boundaries ---
    # Phase 2a is the paper's SEMI-CONTINUAL setting: boundaries are observable, and what changes
    # at one is the physics. So `drift` has boundaries when its schedule is piecewise-constant, and
    # none when it is continuous ("sin"/"linear", which is Phase 2b).
    drift_schedule = str(drift_kwargs["schedule"]).lower()
    boundaries_enabled = (not cfg.get("disable_task_switch", False)
                          and (not drift_mode or drift_schedule == "step"))
    _task_multipliers = list(drift_kwargs["task_multipliers"])

    def _task_label(idx):
        """Stable identifier for the task at index `idx`, shared by every REVISIT of it.

        Retention and backward transfer compare a task against itself later in the sequence, so
        the label must repeat when the physics do. With multipliers [1.0, 1.6, 0.6, 1.6, 0.6],
        tasks 1/3 and 2/4 get the same label because they are the same task.
        """
        if drift_mode:
            return _task_multipliers[idx % len(_task_multipliers)]
        return tasks[idx % len(tasks)]

    n_tasks = len(_task_multipliers) if drift_mode else len(tasks)

    def _task_arg(idx):
        """What `set_task` wants for task `idx` on this benchmark (index vs reward direction)."""
        return idx if drift_mode else tasks[idx % len(tasks)]

    def _task_physics_note(idx):
        if drift_mode:
            return f" (physics x{_task_label(idx):g})"
        return f" (reward direction {_task_label(idx)})"

    drift_probe = ValueDriftProbe()
    # NOTE: the window MUST be measured in whole PPO updates. It used to be `n_steps * 5`, written
    # when n_steps was the entire batch (single env). Under vectorised envs the batch is
    # n_steps * num_envs, so `n_steps * 5` = 1280 < one 2048-step update: the tracker finalised on
    # its first post-switch sample and reported drop = 0.00 by construction. Every boundary_drop
    # number produced before 2026-08-04 is an artifact of that, not a measurement.
    boundary_window = int(cfg.get("boundary_window_updates", 5)) * n_steps * num_envs
    boundary_tracker = BoundaryReturnTracker(post_window_steps=boundary_window,
                                             min_useful_steps=n_steps * num_envs)
    # Theorem 8's advantage lives in a window right after a switch and decays to nothing, so this
    # window must be SHORT relative to the phase (default 20 updates ~= 41k env steps vs a 614k
    # phase). Theorem 7's retention measure needs a snapshot of each task's converged values.
    jumpstart_window = int(cfg.get("jumpstart_window_updates", 20)) * n_steps * num_envs
    jumpstart_tracker = JumpstartTracker(window_steps=jumpstart_window)
    retention_probe = RetentionProbe()
    probe_states = None  # sampled lazily

    def _v_perm(s):
        return agent.get_value(s)[0]

    def _v_full(s):
        vp, vt = agent.get_value(s)
        return vp + vt

    # Both are scored against the same reference (the converged acting value of the finished task),
    # which is the comparison Theorem 7 makes. Vanilla/EWC put the whole value in the perm slot and
    # zero in trans, so the two coincide there — a single critic has no separately retained
    # component, which is exactly what the theorem contrasts against.
    _value_fns = {"perm": _v_perm, "full": _v_full}

    # --- Training state ---  (obs/done initialized from env.reset above)
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

    # --- probe/decay_gain (see _decay_gain_probe) ---
    # Its OWN env: the eval env may be wrapped in RecordVideo, which would file the probe's
    # episodes as evaluations, and the probe must never touch the state of anything it measures.
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

    # --- Forward / backward transfer (Lopez-Paz & Ranzato) ---
    # R[i, j] = mean return on task j with the policy frozen at the END of task i. Rows are filled
    # in at boundaries, so no checkpoints are needed. 0 episodes disables the whole measurement.
    transfer_episodes = int(cfg.get("transfer_eval_episodes", 10))
    transfer = None
    if boundaries_enabled and transfer_episodes > 0 and eval_env is not None:
        transfer = TransferMatrix(n_tasks)

    def _transfer_row():
        """Mean return on every task for the CURRENT policy: frozen, and with no exploration noise.

        Runs on the eval env, never the training env — `set_task` changes physics, and doing that
        to the training env mid-run would corrupt the very thing being measured. The eval env is
        put back on the training task afterwards, and the RNG is isolated so the extra episodes
        cannot shift the training action stream.
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
                # A task INDEX: the set_task lambda above maps it through _task_arg itself.
                restore_task=task_idx % n_tasks, return_std=True,
            )

    if transfer is not None:
        # b_j: a RANDOM-INIT policy's return on each task. Measured before the first update, which
        # is exactly what "random init" means, and what FWT subtracts.
        transfer.set_baselines(_transfer_row()[0])
        print(f"[train] transfer baselines (random init) b_j = "
              f"{np.array2string(transfer.baselines, precision=1)}")

    print(f"[train] agent={cfg['agent']}  seed={seed}  device={device}")
    print(f"[train] total_steps={total_steps}  switch={switch_interval}  n_steps={n_steps}  "
          f"num_envs={num_envs}  async={async_envs and num_envs > 1}  batch={n_steps * num_envs}")
    print(f"[train] step_by_step={cfg.get('step_by_step', False)}")
    print(f"[train] grad_clip={'JOINT (actor+critic together)' if cfg.get('joint_grad_clip', False) else 'separate (actor | critic)'}")
    if cfg["agent"] == "pt":
        # Print the settings that decide whether the PT mechanism runs at all. Every silent
        # misconfiguration this project has hit (inert permanent, constant alpha_P, a config file
        # that never arrived on the box) would have been visible in one line of the log.
        _rho = float(cfg.get("rho", cfg.get("transfer_rate", 0.5)))
        _rm = float(cfg.get("rm_power", cfg.get("alpha_p_rm_power", 0.6)))
        _decay_rho = float(cfg.get("decay_rho", _rho))
        print(f"[train] PT: rho={_rho} k={cfg.get('k')} "
              f"lr_perm={cfg.get('lr_perm')} lr_perm_actor={cfg.get('lr_perm_actor')} "
              f"opt={cfg.get('perm_optimizer')} rm_power={_rm} "
              f"kl_prior_coef={cfg.get('kl_prior_coef')} "
              f"critic_hidden={cfg.get('critic_hidden_sizes', cfg.get('hidden_sizes'))}")
        # Both of these silently change what the run MEANS, so say so in words, not just values.
        if _rm <= 0.0:
            print("[train] PT:   ^ CONSTANT alpha_P (Robbins-Monro OFF) — Theorem 5's premise "
                  "is not satisfied")
        if _decay_rho != _rho:
            print(f"[train] PT:   ^ decay_rho={_decay_rho} != rho={_rho} — the absorb and the "
                  "decay are DECOUPLED, so the transfer is not composition-preserving and the "
                  "acting value jumps at every consolidation. Watch for divergence.")
    if drift_mode:
        _sched = str(drift_kwargs["schedule"])
        print(f"[train] env_mode={env_mode}  targets={drift_kwargs['drift_targets']}  "
              f"schedule={_sched}")
        if cartpole_mode:
            # The whole reason this benchmark was chosen: r in [0,1] over exactly
            # max_episode_steps steps with no early termination, so the ceiling is known rather
            # than guessed. Print it, so every log says what "good" would be.
            print(f"[train] cartpole task={cfg.get('cartpole_task', 'swingup')}  "
                  f"return ceiling = {cfg.get('max_episode_steps', 1000)} "
                  f"(reward in [0,1], no early termination)")
        if dmc_mode:
            # Same guarantee across the whole family, which is what lets returns be averaged over
            # it without a fudge factor. Naming the environment in the log matters as much as the
            # ceiling: six environments share one code path and one results tree.
            _spec = DMC_SPECS[dmc_env_name]
            print(f"[train] dm_control env={dmc_env_name}  obs={_spec.obs_dim} act={_spec.act_dim}"
                  f"  return ceiling = {cfg.get('max_episode_steps', 1000)} "
                  f"(reward in [0,1], no early termination)")
        if _sched == "step":
            # Piecewise-constant physics: the multiplier is a function of the task index, so the
            # only thing worth printing is the sequence itself and that it actually varies.
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
        # ---- LR annealing (CleanRL: linear decay to 0 over training) ----
        if anneal_lr:
            frac = max(1.0 - update_idx / num_updates, 0.0)
            agent.anneal_lr(frac)

        # ---- Task switching ----
        if boundaries_enabled:
            next_switch = (task_idx + 1) * switch_interval
            if global_step >= next_switch:
                # Snapshot the FINISHING task's converged values first (Theorem 7's v_i). Must
                # happen before on_task_switch, which consolidates and decays and so moves V.
                if probe_states is not None:
                    retention_probe.snapshot(_task_label(task_idx), _v_full, probe_states)
                # Same ordering argument for R[i, :]: it is the policy AT THE END of task i, so it
                # must be measured before the switch and before consolidation.
                if transfer is not None:
                    transfer.add_row(task_idx, *_transfer_row())

                task_idx += 1
                # What crosses the boundary differs by benchmark: `directional` passes the reward
                # DIRECTION, `drift` passes the TASK INDEX, which selects that task's physics.
                # The reward function is fixed in drift mode — the non-stationarity is all in the
                # transition kernel.
                switch_arg = task_idx if drift_mode else tasks[task_idx % len(tasks)]
                env.unwrapped.call("set_task", switch_arg)  # propagates to every sub-env

                # Update the one-hot task label in the observation. This is a wrapper on the OUTER
                # (vector) env, not on the sub-envs, so `call` above does not reach it — it has to
                # be set here or the label would stay stuck on task 0 for the whole run while the
                # reward silently flipped underneath it, which is worse than having no label at all.
                # _set_task_id_obs walks the wrapper stack and returns whether it found the wrapper.
                if cfg.get("task_id_obs", False):
                    if not _set_task_id_obs(env, task_idx):
                        raise RuntimeError(
                            "task_id_obs is on but no TaskIDObservation wrapper was found — the "
                            "observation would carry a task label that never changes")

                # RESET THE ENVIRONMENT AT THE BOUNDARY, so a task starts from a defined state.
                #
                # Without this the cheetah carries its pose, velocity and body angle straight
                # across the switch: a boundary that lands mid-stride, mid-flight or upside-down
                # starts the new task from an arbitrary state, and how good that state happens to
                # be is pure luck that differs per seed and per arm. It also blurs what "adaptation
                # after a boundary" means, which is the quantity this whole study measures.
                #
                # It matters MORE for the physics benchmark than for the reward flip: changing
                # mass or damping while the body is airborne is not a well-defined transition at
                # all. The drift clock deliberately survives the reset (the physics are a property
                # of the world, not of the episode), and the in-flight episode is discarded —
                # RecordEpisodeStatistics simply reports no return for it.
                # DEFAULT FALSE, deliberately. Turning it on mid-study would give the runs launched
                # afterwards a different benchmark from the ones already finished — the silent
                # within-study inconsistency this project keeps getting burned by. The Phase 2a
                # study now completing ran WITHOUT it; set `reset_on_task_switch: true` in the
                # config for the next study, where it is the intended design.
                if cfg.get("reset_on_task_switch", False):
                    obs, _ = env.reset()
                    done = np.zeros(num_envs, dtype=np.float32)

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

                # What the DECAY alone buys, measured with no gradient step in between. Run
                # AFTER the switch so it is measured on the physics the agent must now adapt to.
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

        # ---- Collect rollout ----
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

        # Track velocity. HalfCheetah reports x_velocity in info; cartpole has no such quantity,
        # so the list is empty there and np.mean would return nan with a warning every update.
        if agent._velocities:
            velocity_curve.append((global_step, float(np.mean(agent._velocities))))

        # Track episodic returns (EMA, mirrors baseline's avg_return = 0.99 * avg_return + 0.01 * epi_return)
        ema = cfg.get("eval_ema", 0.99)
        for ep_ret in episode_returns:
            avg_return = ema * avg_return + (1 - ema) * ep_ret

        returns_curve.append((global_step, avg_return))

        # ---- PPO update (batch PPO over the flattened rollout) ----
        metrics = agent.update(obs, done, update_idx)
        update_idx += 1

        # PT only: capture any consolidation loss traces produced by this update, tagging each
        # with the step it happened at so the curves can be grouped by task phase when plotted.
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

        # ---- Zero-momentum offline evaluation & checkpointing ----
        eval_interval = cfg.get("eval_interval_updates")
        if eval_interval is None:
            eval_interval = 50
        if eval_env is not None and (update_idx % eval_interval == 0 or update_idx == 1):
            if drift_mode and drift_schedule == "step":
                _set_env_task(eval_env, task_idx)        # evaluate on THIS task's physics
            elif drift_mode:
                _sync_drift_clock(eval_env, global_step)  # evaluate on the CURRENT physics
            else:
                _set_env_task(eval_env, tasks[task_idx % len(tasks)])  # match current train task
                # ...and the task LABEL too, or eval runs the policy with a label that says task 0
                # while the reward is task 1 — a mismatch that would show up as a fake collapse.
                if cfg.get("task_id_obs", False):
                    _set_task_id_obs(eval_env, task_idx)
            _sync_obs_stats(env, eval_env)  # evaluate with the training normalizer's stats
            with _isolated_rng():          # the eval must not perturb the training RNG stream
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
            corr = _actor_perm_trans_corr(agent, agent.buffer.obs.reshape(-1, obs_dim))
            if corr is not None:
                scalars["diag/actor_perm_trans_corr"] = corr
            for metric_name in (
                    "kl_prior", "clip_fraction", "value_perm_l2", "value_trans_l2",
                    "policy_perm_l2", "policy_trans_l2", "log_std_mean", "log_std_min",
                    "grad_norm_trans_actor", "grad_norm_perm_actor",
                    "grad_norm_trans_critic", "grad_norm_perm_critic"):
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
            # Consolidation-regression diagnostics (PT, separate-trunk variant only).
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
                # Record where on the drift schedule we are, so return curves can be read against
                # the physics that produced them.
                #
                # `_drift_multiplier` only knows the CLOCK-driven schedules. Under "step" — which
                # is Phase 2a's default on both physics benchmarks — the multiplier is a function
                # of the task index, and asking the clock formula for it returned the "linear"
                # branch: a ramp from 1.0 to 2.25 over a run whose physics were actually cycling
                # through [1.0, 1.6, 0.6, ...]. Diagnostic-only, but it is a logged quantity that
                # said something untrue, so it is read from the task label here instead.
                scalars["drift/multiplier"] = (
                    float(_task_label(task_idx)) if drift_schedule == "step"
                    else _drift_multiplier(cfg, global_step))
            logger.log_scalars(scalars, step=global_step)

        # Sample probe states lazily (first rollout provides good coverage)
        if probe_states is None:
            flat_obs = agent.buffer.obs.reshape(-1, obs_dim)
            n_probe = min(cfg.get("probe_states", 256), flat_obs.shape[0])
            probe_states = flat_obs[:n_probe].copy()
            # Control baselines for the retention metric — see RetentionProbe's docstring.
            # Without these, an INERT permanent scores better than an adapted one on a
            # sign-flip task pair and reads as a false confirmation of Theorem 7.
            retention_probe.set_baseline("perm_init", _v_perm(probe_states))
            retention_probe.set_baseline("zero", np.zeros(len(probe_states), dtype=np.float32))

        # Is the permanent component doing ANYTHING? `perm_frac` is its share of |V|; `perm_drift`
        # is how far it has moved from its initialisation. A permanent with drift ~0 means the
        # dual-timescale mechanism is not running, whatever the returns say.
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

        # Jumpstart window (Thm 6/8): return in the short window right after each switch.
        jrec = jumpstart_tracker.update(global_step, avg_return)
        if jrec is not None:
            logger.log_scalars({
                "boundary/jumpstart_first": jrec["first"],
                "boundary/jumpstart_mean": jrec["mean"],
                "boundary/jumpstart_end": jrec["end"],
                "boundary/jumpstart_gain": jrec["gain"],
            }, step=jrec["step"])

        # Retention (Thm 7): squared error against the converged values of the INACTIVE task(s).
        # Reported for the permanent component and for the full acting value; the paper predicts
        # the permanent degrades less. Silent until the first task has finished.
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

    # --- Finalize ---
    if transfer is not None:
        # The last task never hits a boundary, so its row is measured here, at the end of training.
        # If the run went round the cycle more than once there is no row left to fill; the matrix
        # is then incomplete and bwt()/fwt() return None rather than a number built from a
        # row that belongs to a different pass.
        if task_idx < n_tasks:
            transfer.add_row(task_idx, *_transfer_row())
        summary = transfer.summary()
        # Two independent reads on evaluation noise: the within-cell standard error, and the
        # disagreement between columns that are the SAME task (the sequence revisits, so those
        # should agree). If a method difference is not comfortably larger than these, the
        # transfer figure is noise.
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

    # Persist ALL logged scalars (jumpstart, retention, consolidation diagnostics, ...) regardless
    # of which logging backends were enabled — sweeps normally run --no-tb --no-wandb.
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
