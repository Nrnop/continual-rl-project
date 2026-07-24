"""DirectionalHalfCheetah: a continual-RL wrapper around MuJoCo HalfCheetah.

The continual non-stationarity mirrors the baseline's reward-sign flip (e.g. the
[(1.0, -1.0), (-1.0, 1.0)] goal-reward pairs in control/tabular/CL_envs.py), but here the "task"
is the *running direction*:

    task = +1  ->  reward agent for running FORWARD  (+x velocity)
    task = -1  ->  reward agent for running BACKWARD (-x velocity)

We recompute the reward from the env's `info` dict rather than the env's own reward, so the same
code works across HalfCheetah-v4 and v5 (both expose `x_velocity` and `reward_ctrl`). The control
cost is kept task-invariant (shared physics), only the forward term flips sign.

The wrapper does NOT switch tasks on its own; the training loop calls `set_task(direction)` at the
fixed `--switch` boundary, exactly like the baseline cycles its env list.
"""
import gymnasium as gym
import numpy as np


def make_base_env(env_id="HalfCheetah-v5", max_episode_steps=1000, render_mode=None):
    """Create the underlying gymnasium HalfCheetah, falling back v5 -> v4."""
    for candidate in (env_id, "HalfCheetah-v4", "HalfCheetah-v3"):
        try:
            return gym.make(candidate, max_episode_steps=max_episode_steps, render_mode=render_mode)
        except Exception:
            continue
    # Last resort: let gym raise its own informative error on the requested id.
    return gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)


class DirectionalHalfCheetah(gym.Wrapper):
    """HalfCheetah with a flippable running direction.

    Parameters
    ----------
    env_id : str
        gymnasium id to instantiate (v5 preferred; auto-falls back to v4/v3).
    direction : int
        Initial task, +1 (forward) or -1 (backward).
    forward_reward_weight : float
        Scales the directional velocity term. HalfCheetah-v4/v5 default is 1.0.
    """

    def __init__(
        self,
        env_id="HalfCheetah-v5",
        direction=1,
        forward_reward_weight=1.0,
        max_episode_steps=1000,
        render_mode=None,
    ):
        env = make_base_env(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
        super().__init__(env)
        self.direction = int(np.sign(direction)) or 1
        self.forward_reward_weight = float(forward_reward_weight)

    def set_task(self, direction):
        """Flip the running direction. direction in {+1, -1}."""
        self.direction = int(np.sign(direction)) or 1
        return self.direction

    def _directional_reward(self, info):
        """Reconstruct reward = direction * fwd_velocity_term + ctrl_cost (shared)."""
        x_velocity = info.get("x_velocity")
        reward_ctrl = info.get("reward_ctrl")
        if x_velocity is None:
            # Older wrappers may expose the forward term as reward_run instead.
            run_term = info.get("reward_run", 0.0)
        else:
            run_term = self.forward_reward_weight * x_velocity
        if reward_ctrl is None:
            reward_ctrl = 0.0
        return self.direction * run_term + reward_ctrl

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        reward = self._directional_reward(info)
        info["direction"] = self.direction
        info["directional_reward"] = reward
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
