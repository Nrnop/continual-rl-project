"""DirectionalPointMass: a discrete-task-switch continual benchmark, CPU-cheap but non-trivial.

Same 1-D point-mass dynamics as `simple_drift.DriftingPointMass`, but the non-stationarity is a
discrete TARGET flip at fixed step boundaries (mirrors `DirectionalHalfCheetah`'s reward-sign
flip), not a continuous drift. The task is position-tracking (reach and HOLD `target =
direction * target_magnitude`), not pure velocity-racing: with a weak actuator and drag, this
needs a real braking/stabilizing controller, so relearning after a switch has a genuine cost
(unlike a bang-bang velocity race, which vanilla PPO solves in a couple of updates regardless of
forgetting). Lets `train.py`'s existing switch/consolidation/retention machinery (generic over any
env exposing `set_task`) run on a 2-D-observation env instead of MuJoCo.
"""
import functools

import gymnasium as gym
import numpy as np


class DirectionalPointMass(gym.Env):
    """1-D point mass tracking `target = direction * target_magnitude`; `set_task` flips it."""

    metadata = {"render_modes": []}

    def __init__(self, direction=1.0, target_magnitude=2.0, dt=0.1, drag=0.35, action_scale=1.0,
                 position_limit=3.0, velocity_limit=3.0, ctrl_cost_weight=0.01,
                 max_episode_steps=150):
        super().__init__()
        self.direction = float(direction)
        self.target_magnitude = float(target_magnitude)
        self.dt = float(dt)
        self.drag = float(drag)
        self.action_scale = float(action_scale)
        self.position_limit = float(position_limit)
        self.velocity_limit = float(velocity_limit)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.max_episode_steps = int(max_episode_steps)
        self.observation_space = gym.spaces.Box(
            low=np.array([-self.position_limit, -self.velocity_limit], dtype=np.float32),
            high=np.array([self.position_limit, self.velocity_limit], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.episode_step = 0
        self.position = 0.0
        self.velocity = 0.0

    @property
    def target(self):
        return self.direction * self.target_magnitude

    def set_task(self, direction):
        """Flip the rewarded target's sign; any float, +1/-1 give the symmetric case."""
        self.direction = float(direction)
        return self.direction

    def _observation(self):
        return np.asarray([self.position, self.velocity], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        self.position = float(self.np_random.uniform(-0.5, 0.5))
        self.velocity = float(self.np_random.uniform(-0.1, 0.1))
        return self._observation(), {"direction": self.direction, "target": self.target}

    def step(self, action):
        control = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        acceleration = self.action_scale * control - self.drag * self.velocity
        self.velocity = float(np.clip(
            self.velocity + self.dt * acceleration, -self.velocity_limit, self.velocity_limit,
        ))
        self.position = float(np.clip(
            self.position + self.dt * self.velocity, -self.position_limit, self.position_limit,
        ))
        reward_ctrl = -self.ctrl_cost_weight * control * control
        reward = 1.0 - 0.5 * (self.position - self.target) ** 2 + reward_ctrl
        self.episode_step += 1
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        info = {
            "direction": self.direction,
            "target": self.target,
            "x_velocity": self.velocity,
            "reward_ctrl": reward_ctrl,
            "directional_reward": reward,
        }
        return self._observation(), float(reward), terminated, truncated, info


def _make_single(**kwargs):
    return DirectionalPointMass(**kwargs)


def _wrap_env(env, normalize_obs, normalize_reward, gamma, clip_obs, clip_reward):
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -clip_obs, clip_obs), env.observation_space
        )
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.ClipReward(env, -clip_reward, clip_reward)
    return env


def make_directional_point_env(direction=1.0, target_magnitude=2.0, dt=0.1, drag=0.35,
                               action_scale=1.0, position_limit=3.0, velocity_limit=3.0,
                               ctrl_cost_weight=0.01, max_episode_steps=150, normalize_obs=False,
                               normalize_reward=False, gamma=0.99, clip_obs=10.0, clip_reward=10.0):
    env = DirectionalPointMass(
        direction=direction, target_magnitude=target_magnitude, dt=dt, drag=drag,
        action_scale=action_scale, position_limit=position_limit, velocity_limit=velocity_limit,
        ctrl_cost_weight=ctrl_cost_weight, max_episode_steps=max_episode_steps,
    )
    return _wrap_env(env, normalize_obs, normalize_reward, gamma, clip_obs, clip_reward)


def make_directional_point_vector_env(direction=1.0, target_magnitude=2.0, dt=0.1, drag=0.35,
                                      action_scale=1.0, position_limit=3.0, velocity_limit=3.0,
                                      ctrl_cost_weight=0.01, max_episode_steps=150, num_envs=1,
                                      normalize_obs=False, normalize_reward=False, gamma=0.99,
                                      clip_obs=10.0, clip_reward=10.0, asynchronous=False):
    kwargs = dict(
        direction=direction, target_magnitude=target_magnitude, dt=dt, drag=drag,
        action_scale=action_scale, position_limit=position_limit, velocity_limit=velocity_limit,
        ctrl_cost_weight=ctrl_cost_weight, max_episode_steps=max_episode_steps,
    )
    fns = [functools.partial(_make_single, **kwargs) for _ in range(int(num_envs))]
    if asynchronous and num_envs > 1:
        base = gym.vector.AsyncVectorEnv(fns)
    else:
        base = gym.vector.SyncVectorEnv(fns)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(base)
    if normalize_obs:
        envs = gym.wrappers.vector.NormalizeObservation(envs)
        envs = gym.wrappers.vector.TransformObservation(
            envs, functools.partial(np.clip, a_min=-clip_obs, a_max=clip_obs)
        )
    if normalize_reward:
        envs = gym.wrappers.vector.NormalizeReward(envs, gamma=gamma)
        envs = gym.wrappers.vector.ClipReward(envs, -clip_reward, clip_reward)
    return envs
