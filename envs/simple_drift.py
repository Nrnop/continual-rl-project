"""A tiny continuous-control benchmark with smooth, boundary-free dynamics drift.

The agent controls a one-dimensional point mass. The target stays at the origin while the drag
coefficient follows a bounded sinusoid, so the task is stationary in reward but non-stationary in
transition dynamics. The state is only position and velocity; the changing drag is not exposed.
"""
import functools

import gymnasium as gym
import numpy as np


class DriftingPointMass(gym.Env):
    """One-dimensional stabilization with a smoothly drifting drag coefficient."""

    metadata = {"render_modes": []}

    def __init__(self, target=0.0, dt=0.1, base_drag=0.35, drift_amplitude=0.75,
                 drift_period=4000, drift_schedule="sin", drift_phase=0.0,
                 action_scale=1.0, position_limit=3.0, velocity_limit=3.0,
                 clock_scale=1, max_episode_steps=200, target_amplitude=0.0):
        super().__init__()
        if drift_schedule not in ("sin", "linear"):
            raise ValueError("drift_schedule must be 'sin' or 'linear'")
        if drift_period <= 0 or dt <= 0 or base_drag < 0:
            raise ValueError("drift_period and dt must be positive; base_drag cannot be negative")
        self.base_target = float(target)
        # DRIFTING THE TARGET, NOT ONLY THE DYNAMICS.
        #
        # With target_amplitude = 0 (the default, and the original behaviour) only `drag` drifts
        # while the goal stays at the origin. Measured: post-learning return then swings just
        # 3.3-3.5% and every agent sits at 96-99% of the 200-per-episode ceiling, because holding
        # position at a fixed origin is easy at any drag. A benchmark that saturates cannot
        # distinguish continual-learning methods -- FULL_PT.md §20.
        #
        # A nonzero amplitude drifts the GOAL smoothly on the same clock, so the optimal policy
        # itself changes continuously. That is the true smooth analogue of DirectionalPointMass's
        # target flip, and (as there) the target is NOT part of the observation, so the agent must
        # track it from reward alone.
        self.target_amplitude = float(target_amplitude)
        self.dt = float(dt)
        self.base_drag = float(base_drag)
        self.drift_amplitude = float(drift_amplitude)
        self.drift_period = int(drift_period)
        self.drift_schedule = drift_schedule
        self.drift_phase = float(drift_phase)
        self.action_scale = float(action_scale)
        self.position_limit = float(position_limit)
        self.velocity_limit = float(velocity_limit)
        self.clock_scale = int(clock_scale)
        self.max_episode_steps = int(max_episode_steps)
        self.observation_space = gym.spaces.Box(
            low=np.array([-self.position_limit, -self.velocity_limit], dtype=np.float32),
            high=np.array([self.position_limit, self.velocity_limit], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.t = 0
        self.episode_step = 0
        self.position = 0.0
        self.velocity = 0.0
        self._apply(self.multiplier())

    def multiplier(self, t=None):
        """Return the smooth drift multiplier at global virtual step ``t``."""
        t = self.t if t is None else int(t)
        if self.drift_schedule == "sin":
            signal = np.sin(2.0 * np.pi * t / self.drift_period + self.drift_phase)
        else:
            signal = t / self.drift_period
        return float(1.0 + self.drift_amplitude * signal)

    def lipschitz_constant(self):
        """Maximum multiplier change per global step for this schedule."""
        if self.drift_schedule == "sin":
            return abs(self.drift_amplitude) * 2.0 * np.pi / self.drift_period
        return abs(self.drift_amplitude) / self.drift_period

    def _apply(self, multiplier):
        self.drag = max(0.0, self.base_drag * float(multiplier))

    def _observation(self):
        return np.asarray([self.position, self.velocity], dtype=np.float32)

    @property
    def target(self):
        """Goal position at the current drift-clock step.

        Rides the SAME clock as the dynamics drift, so one `drift_period` moves both. With
        `target_amplitude = 0` this is the constant `base_target` and the env is bit-identical to
        its original behaviour.
        """
        if self.target_amplitude == 0.0:
            return self.base_target
        if self.drift_schedule == "sin":
            signal = np.sin(2.0 * np.pi * self.t / self.drift_period + self.drift_phase)
        else:
            signal = self.t / self.drift_period
        return float(self.base_target + self.target_amplitude * signal)

    def current_params(self):
        return {
            "drift_multiplier": self.multiplier(),
            "drift_drag": float(self.drag),
            "drift_target": float(self.target),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        self.position = float(self.np_random.uniform(-0.5, 0.5))
        self.velocity = float(self.np_random.uniform(-0.1, 0.1))
        return self._observation(), self.current_params()

    def step(self, action):
        control = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        control = float(np.clip(control, -1.0, 1.0))
        acceleration = self.action_scale * control - self.drag * self.velocity
        self.velocity = float(np.clip(
            self.velocity + self.dt * acceleration,
            -self.velocity_limit,
            self.velocity_limit,
        ))
        self.position = float(np.clip(
            self.position + self.dt * self.velocity,
            -self.position_limit,
            self.position_limit,
        ))
        reward_ctrl = -0.01 * control * control
        reward = 1.0 - 0.5 * (self.position - self.target) ** 2 + reward_ctrl
        self.episode_step += 1
        self.t += self.clock_scale
        self._apply(self.multiplier())
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        info = self.current_params()
        info.update({
            "x_velocity": self.velocity,
            "reward_ctrl": reward_ctrl,
            "directional_reward": reward,
        })
        return self._observation(), float(reward), terminated, truncated, info


def _make_single_point_drift(**kwargs):
    return DriftingPointMass(**kwargs)


def _wrap_point_env(env, normalize_obs, normalize_reward, gamma, clip_obs, clip_reward):
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -clip_obs, clip_obs), env.observation_space
        )
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.ClipReward(env, -clip_reward, clip_reward)
    return env


def make_point_drift_env(target=0.0, target_amplitude=0.0, dt=0.1, base_drag=0.35, drift_amplitude=0.75,
                         drift_period=4000, drift_schedule="sin", drift_phase=0.0,
                         action_scale=1.0, position_limit=3.0, velocity_limit=3.0,
                         max_episode_steps=200, normalize_obs=False, normalize_reward=False,
                         gamma=0.99, clip_obs=10.0, clip_reward=10.0):
    env = DriftingPointMass(
        target=target, target_amplitude=target_amplitude, dt=dt, base_drag=base_drag, drift_amplitude=drift_amplitude,
        drift_period=drift_period, drift_schedule=drift_schedule, drift_phase=drift_phase,
        action_scale=action_scale, position_limit=position_limit,
        velocity_limit=velocity_limit, max_episode_steps=max_episode_steps,
    )
    return _wrap_point_env(env, normalize_obs, normalize_reward, gamma, clip_obs, clip_reward)


def make_point_drift_vector_env(target=0.0, target_amplitude=0.0, dt=0.1, base_drag=0.35, drift_amplitude=0.75,
                                drift_period=4000, drift_schedule="sin", drift_phase=0.0,
                                action_scale=1.0, position_limit=3.0, velocity_limit=3.0,
                                max_episode_steps=200, num_envs=1, normalize_obs=False,
                                normalize_reward=False, gamma=0.99, clip_obs=10.0,
                                clip_reward=10.0, asynchronous=False):
    kwargs = dict(
        target=target, target_amplitude=target_amplitude, dt=dt, base_drag=base_drag, drift_amplitude=drift_amplitude,
        drift_period=drift_period, drift_schedule=drift_schedule, drift_phase=drift_phase,
        action_scale=action_scale, position_limit=position_limit,
        velocity_limit=velocity_limit, max_episode_steps=max_episode_steps,
        clock_scale=num_envs,
    )
    fns = [functools.partial(_make_single_point_drift, **kwargs) for _ in range(int(num_envs))]
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
