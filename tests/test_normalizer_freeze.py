import gymnasium as gym
import numpy as np

from src_continuous_control.train import (
    _freeze_normalizers,
    _normalizer_freeze_due,
    _normalizer_wrappers,
)


class TinyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.full(2, self.step_count, dtype=np.float32)
        return obs, 1.0 + self.step_count, self.step_count >= 10, False, {}


def _single_wrapped_env():
    env = gym.wrappers.NormalizeObservation(TinyEnv())
    return gym.wrappers.NormalizeReward(env, gamma=0.99)


def _stats(env):
    wrappers = dict(_normalizer_wrappers(env))
    obs_rms = wrappers["observation"].obs_rms
    reward_rms = wrappers["reward"].return_rms
    return (
        obs_rms.mean.copy(), obs_rms.var.copy(), obs_rms.count,
        reward_rms.mean.copy(), reward_rms.var.copy(), reward_rms.count,
    )


def _assert_stats_equal(actual, expected):
    for actual_value, expected_value in zip(actual, expected):
        if np.isscalar(expected_value):
            assert actual_value == expected_value
        else:
            assert np.array_equal(actual_value, expected_value)


def test_freeze_single_env_observation_and_reward_normalizers():
    env = _single_wrapped_env()
    env.reset(seed=0)
    for _ in range(3):
        env.step(0)
    before = _stats(env)

    snapshot = _freeze_normalizers(env)
    assert {kind for kind, _ in _normalizer_wrappers(env)} == {"observation", "reward"}
    assert all(wrapper.update_running_mean is False
               for _, wrapper in _normalizer_wrappers(env))
    assert "observation_mean_l2" in snapshot
    assert "reward_var_l2" in snapshot

    for _ in range(3):
        env.step(1)
    _assert_stats_equal(_stats(env), before)


def test_freeze_vector_env_observation_and_reward_normalizers():
    base = gym.vector.SyncVectorEnv([TinyEnv, TinyEnv])
    env = gym.wrappers.vector.NormalizeObservation(base)
    env = gym.wrappers.vector.NormalizeReward(env, gamma=0.99)
    env.reset(seed=0)
    env.step(np.zeros(2, dtype=np.int64))
    before = _stats(env)

    _freeze_normalizers(env)
    assert len(_normalizer_wrappers(env)) == 2
    for _ in range(2):
        env.step(np.ones(2, dtype=np.int64))
    _assert_stats_equal(_stats(env), before)


def test_normalizer_freeze_threshold_boundaries():
    assert not _normalizer_freeze_due(0, None)
    assert not _normalizer_freeze_due(0, -1)
    assert not _normalizer_freeze_due(9, 10)
    assert _normalizer_freeze_due(10, 10)
    assert _normalizer_freeze_due(11, 10)
