"""DirectionalHalfCheetah: HalfCheetah whose task is the running direction (reward-sign flip).

Task +1 rewards +x velocity, task -1 rewards -x velocity; the control cost is unchanged. The reward
is rebuilt from `info` so it works on HalfCheetah-v4 and v5. The training loop calls `set_task` at
each boundary. Also provides `make_base_env` (used by the drift env) and the TaskIDObservation
wrappers (used by the other envs too).
"""
import functools

import gymnasium as gym
import numpy as np


def make_base_env(env_id="HalfCheetah-v5", max_episode_steps=1000, render_mode=None):
    """Create the underlying gymnasium HalfCheetah, falling back v5 -> v4."""
    if isinstance(env_id, gym.Env):
        return env_id
    for candidate in (env_id, "HalfCheetah-v4", "HalfCheetah-v3"):
        try:
            return gym.make(candidate, max_episode_steps=max_episode_steps, render_mode=render_mode)
        except Exception:
            continue
    # Let gym raise its own error on the requested id.
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
        self.direction = float(direction)
        self.forward_reward_weight = float(forward_reward_weight)

    def set_task(self, direction):
        """Set the forward-velocity coefficient. Any float; +1 / -1 give the symmetric flip.

        Floats allow asymmetric task sets. Under a symmetric +-1 flip E_tau[r_tau] = ctrl, so the
        permanent's fixed point E_tau[v_tau] (Theorem 5) has no velocity term to store;
        `tasks: [1.0, -0.5]` gives E_tau[r_tau] = 0.25*w*v_x + ctrl.
        """
        self.direction = float(direction)
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


def make_directional_env(
    env_id="HalfCheetah-v5",
    direction=1,
    max_episode_steps=1000,
    render_mode=None,
    normalize_obs=False,
    normalize_reward=False,
    gamma=0.99,
    clip_obs=10.0,
    clip_reward=10.0,
    task_id_obs=False,
    n_task_ids=2,
):
    """DirectionalHalfCheetah optionally wrapped with CleanRL-style normalization.

    Running-mean/std observation and discounted-return reward normalizers, clipped to +/-10, as
    in CleanRL's ppo_continuous_action. The un-normalized reward stays in
    info["directional_reward"] for episodic-return logging.

    Wrapper order (inner -> outer): Directional -> NormalizeObservation ->
    clip obs -> NormalizeReward -> clip reward.
    """
    env = DirectionalHalfCheetah(
        env_id=env_id,
        direction=direction,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda o: np.clip(o, -clip_obs, clip_obs),
            env.observation_space,
        )
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.ClipReward(env, -clip_reward, clip_reward)
    # Outermost, so the label is never normalized.
    if task_id_obs:
        env = TaskIDObservationSingle(env, n_task_ids)
    return env


class TaskIDObservationSingle(gym.ObservationWrapper):
    """Single-env version of `TaskIDObservation`, for the eval and probe envs.

    Eval and probe envs must match the training env's observation layout.
    """

    def __init__(self, env, n_tasks):
        super().__init__(env)
        self.n_tasks = int(n_tasks)
        self._task_id = 0
        low, high = env.observation_space.low, env.observation_space.high
        self.observation_space = gym.spaces.Box(
            np.concatenate([low, np.zeros(self.n_tasks, dtype=low.dtype)]),
            np.concatenate([high, np.ones(self.n_tasks, dtype=high.dtype)]),
            dtype=env.observation_space.dtype)

    def set_task_id(self, i):
        self._task_id = int(i) % self.n_tasks
        return self._task_id

    def observation(self, obs):
        onehot = np.zeros(self.n_tasks, dtype=obs.dtype)
        onehot[self._task_id] = 1.0
        return np.concatenate([obs, onehot])


class TaskIDObservation(gym.vector.VectorObservationWrapper):
    """Append a one-hot task identifier to every observation.

    On the reward-flip benchmark the two tasks need opposite actions from identical observations,
    so the policy needs a task label (Anand & Precup's chain states carry a similar feature).
    Keep this wrapper outside NormalizeObservation so the one-hot stays exactly 0/1. The training
    loop calls `set_task_id(i)` at each boundary.
    """

    def __init__(self, env, n_tasks):
        super().__init__(env)
        self.n_tasks = int(n_tasks)
        self._task_id = 0
        low = self.env.single_observation_space.low
        high = self.env.single_observation_space.high
        pad_lo = np.zeros(self.n_tasks, dtype=low.dtype)
        pad_hi = np.ones(self.n_tasks, dtype=high.dtype)
        self.single_observation_space = gym.spaces.Box(
            np.concatenate([low, pad_lo]), np.concatenate([high, pad_hi]),
            dtype=self.env.single_observation_space.dtype)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.env.num_envs)

    def set_task_id(self, i):
        """Set the task label; wraps modulo n_tasks like the task list."""
        self._task_id = int(i) % self.n_tasks
        return self._task_id

    def observations(self, obs):
        onehot = np.zeros((obs.shape[0], self.n_tasks), dtype=obs.dtype)
        onehot[:, self._task_id] = 1.0
        return np.concatenate([obs, onehot], axis=1)


def _make_single_directional(env_id, direction, max_episode_steps, render_mode=None):
    """Module-level factory so functools.partial stays picklable for AsyncVectorEnv on Windows."""
    return DirectionalHalfCheetah(
        env_id=env_id, direction=direction,
        max_episode_steps=max_episode_steps, render_mode=render_mode,
    )


def make_vector_env(
    env_id="HalfCheetah-v5",
    num_envs=1,
    direction=1,
    max_episode_steps=1000,
    gamma=0.99,
    normalize_obs=False,
    normalize_reward=False,
    clip_obs=10.0,
    clip_reward=10.0,
    asynchronous=True,
    task_id_obs=False,
    n_task_ids=2,
):
    """Vectorized DirectionalHalfCheetah with CleanRL-style normalization at the vector level.

    Wrapper order (inner -> outer): [Async|Sync]VectorEnv -> RecordEpisodeStatistics
    (true return, before normalization) -> NormalizeObservation -> clip obs -> NormalizeReward
    -> clip reward -> TaskIDObservation. Switch tasks with
    `env.unwrapped.call("set_task", direction)`, which reaches every sub-env.
    """
    fns = [
        functools.partial(_make_single_directional, env_id, direction, max_episode_steps)
        for _ in range(num_envs)
    ]
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
    # Last, so the one-hot is never normalized.
    if task_id_obs:
        envs = TaskIDObservation(envs, n_task_ids)
    return envs
