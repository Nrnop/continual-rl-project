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

from .agents import AGENTS
from .envs.directional_half_cheetah import DirectionalHalfCheetah
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
    """Merge default.yaml ← agent-specific YAML ← CLI overrides."""
    cfg = _load_yaml(os.path.join(_CFG_DIR, "default.yaml"))

    agent_yaml = os.path.join(_CFG_DIR, f"ppo_{cli_args.agent}.yaml")
    if os.path.exists(agent_yaml):
        cfg.update(_load_yaml(agent_yaml))

    # CLI overrides (only non-None values)
    for key, val in vars(cli_args).items():
        if val is not None:
            cfg[key] = val

    # Ensure agent is set
    cfg.setdefault("agent", cli_args.agent)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Continuous-control PT-PPO training")
    p.add_argument("--agent", type=str, default="pt", choices=list(AGENTS.keys()))
    p.add_argument("--seed", type=int, default=0)

    # Env
    p.add_argument("--env-id", type=str, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--switch", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--max-episode-steps", type=int, default=None)

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
            obs, r, term, trunc, _ = eval_env.step(action_np)
            ep_ret += r
            done = term or trunc
        eval_returns.append(ep_ret)
    return float(np.mean(eval_returns))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    cli = parse_args()
    cfg = build_config(cli)

    seed = cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    # --- Environment ---
    env = DirectionalHalfCheetah(
        env_id=cfg.get("env_id", "HalfCheetah-v5"),
        direction=cfg["tasks"][0],
        max_episode_steps=cfg.get("max_episode_steps", 1000),
    )
    obs = seed_env(env, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

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

    # --- Training state ---
    avg_return = 0.0
    global_step = 0
    update_idx = 0
    task_idx = 0
    done = 0.0

    returns_curve = []
    all_episode_returns = []
    eval_returns_curve = []

    eval_env = None
    if not cfg.get("no_eval", False):
        eval_env = DirectionalHalfCheetah(
            env_id=cfg.get("env_id", "HalfCheetah-v5"),
            direction=cfg["tasks"][0],
            max_episode_steps=cfg.get("max_episode_steps", 1000),
        )
        seed_env(eval_env, seed + 10000)

    print(f"[train] agent={cfg['agent']}  seed={seed}  device={device}")
    print(f"[train] total_steps={total_steps}  switch={switch_interval}  n_steps={n_steps}")
    t0 = time.time()

    while global_step < total_steps:
        # ---- Task switching ----
        next_switch = (task_idx + 1) * switch_interval
        if global_step >= next_switch:
            task_idx += 1
            direction = tasks[task_idx % len(tasks)]
            env.set_task(direction)
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
        global_step += n_steps

        # Track episodic returns (EMA, mirrors baseline's avg_return = 0.99 * avg_return + 0.01 * epi_return)
        ema = cfg.get("eval_ema", 0.99)
        for ep_ret in episode_returns:
            avg_return = ema * avg_return + (1 - ema) * ep_ret

        returns_curve.append(avg_return)

        # ---- PPO update ----
        metrics = agent.update(obs, done, update_idx)
        update_idx += 1

        # ---- Zero-momentum offline evaluation & checkpointing ----
        eval_interval = cfg.get("eval_interval_updates")
        if eval_interval is None:
            eval_interval = 50
        if eval_env is not None and (update_idx % eval_interval == 0 or update_idx == 1):
            eval_env.set_task(env.direction)
            clean_eval_ret = _run_offline_eval(agent, eval_env, n_episodes=5)
            eval_returns_curve.append((update_idx, clean_eval_ret))
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
            logger.log_scalars(scalars, step=global_step)

        # Sample probe states lazily (first rollout provides good coverage)
        if probe_states is None:
            n_probe = min(cfg.get("probe_states", 256), n_steps)
            probe_states = agent.buffer.obs[:n_probe].copy()

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
    mean_drop = boundary_tracker.mean_drop()
    if mean_drop is not None:
        logger.log_scalar("boundary/mean_drop", mean_drop, global_step)
        print(f"[train] mean boundary drop: {mean_drop:.2f}")

    logger.close()
    elapsed = time.time() - t0
    print(f"[train] Done. {global_step} steps in {elapsed:.0f}s ({global_step/elapsed:.0f} sps). "
          f"Returns saved to {fname}")


if __name__ == "__main__":
    main()
