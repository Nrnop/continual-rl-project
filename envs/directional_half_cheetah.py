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
        self.direction = float(direction)
        self.forward_reward_weight = float(forward_reward_weight)

    def set_task(self, direction):
        """Set the forward-velocity coefficient. Any float; +1 / -1 give the symmetric flip.

        This used to coerce to `int(np.sign(direction)) or 1`, which made ASYMMETRIC task sets
        impossible to express. That mattered more than it looks: Theorem 5 puts the permanent value
        function's fixed point at E_tau[v_tau], and under a symmetric +-1 flip

            r_{+1} = +w*v_x + ctrl ,  r_{-1} = -w*v_x + ctrl   =>   E_tau[r_tau] = ctrl

        the entire task-discriminative term cancels, so the permanent component has essentially
        nothing to store and the method is being tested in the one regime where its own theory says
        it has no room to work. The paper's benchmarks are asymmetric by construction (JBW
        alternates -1 / +2; MinAtar samples three different games).

        With floats allowed, `tasks: [1.0, -0.5]` gives E_tau[r_tau] = 0.25*w*v_x + ctrl — a
        non-degenerate permanent target — while keeping the same physics and the same reversal
        structure. See configs/pt_paper_asym.yaml.
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

    CleanRL's ppo_continuous_action reaches its HalfCheetah benchmark only with a
    running-mean/std observation normalizer and a discounted-return reward
    normalizer, both clipped to +/-10. The true (un-normalized) reward is still
    exposed via info["directional_reward"] so episodic-return logging stays honest.

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
    # Outermost, so the label is never normalized — same reasoning as the vector version.
    if task_id_obs:
        env = TaskIDObservationSingle(env, n_task_ids)
    return env


class TaskIDObservationSingle(gym.ObservationWrapper):
    """Single-env twin of `TaskIDObservation`, for the eval and probe envs.

    These envs must produce observations with the SAME layout as the training env or the policy
    cannot be run on them at all — the shapes simply will not multiply. Keeping them in step is not
    optional bookkeeping; a probe env one dimension short silently measures a different agent.
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

    WHY THIS EXISTS. On the reward-flip benchmark the two tasks demand OPPOSITE actions from
    identical observations, so without a task label the policy is being asked to be two different
    functions of the same input. That is not a memory problem, it is a contradiction, and no
    continual-learning method can solve it. Anand & Precup's own control experiment gives the agent
    exactly this signal: their chain states carry a feature encoding "which end of the chain
    contains the reward", and their transient value function reads it -- that reward-correlated
    feature is *why* their transient adapts within five episodes of a task change.

    OUTERMOST ON PURPOSE. This wrapper sits OUTSIDE NormalizeObservation. The running normalizer
    would rescale the one-hot by a mean/std estimated across tasks, shrinking a constant-per-phase
    signal toward zero and drifting it every time the task changes -- destroying the very thing it
    is meant to convey. Appending after normalization keeps it exactly 0/1.

    The training loop owns the label: it calls `set_task_id(i)` at each boundary, the same place it
    calls `set_task` on the inner envs.
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
        """Called by the training loop at a boundary. Wraps like the task list does."""
        self._task_id = int(i) % self.n_tasks
        return self._task_id

    def observations(self, obs):
        onehot = np.zeros((obs.shape[0], self.n_tasks), dtype=obs.dtype)
        onehot[:, self._task_id] = 1.0
        return np.concatenate([obs, onehot], axis=1)


def _make_single_directional(env_id, direction, max_episode_steps, render_mode=None):
    """Module-level single-env factory so functools.partial is picklable for AsyncVectorEnv
    (Windows spawn requires picklable env_fns — a local closure would fail)."""
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
    """Vectorized DirectionalHalfCheetah with CleanRL-style normalization applied at the
    vector level (in the main process). Returns a gymnasium VectorEnv.

    Wrapper order (inner -> outer): [Async|Sync]VectorEnv -> RecordEpisodeStatistics
    (records the TRUE directional return, before normalization) -> NormalizeObservation ->
    clip obs -> NormalizeReward -> clip reward.

    Task switching: call `env.unwrapped.call("set_task", direction)` (reaches every sub-env,
    including across processes with AsyncVectorEnv).
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
    # LAST, so the one-hot is never touched by NormalizeObservation — see TaskIDObservation.
    if task_id_obs:
        envs = TaskIDObservation(envs, n_task_ids)
    return envs
