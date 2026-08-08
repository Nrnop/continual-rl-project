from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from src_continuous_control.envs.simple_drift import (
    DriftingPointMass,
    make_point_drift_vector_env,
)
from src_continuous_control.train import build_config


def test_point_drift_is_smooth_cyclic_and_reward_preserving():
    env = DriftingPointMass(drift_amplitude=0.6, drift_period=200, max_episode_steps=10)
    multipliers = np.asarray([env.multiplier(t) for t in range(401)])
    assert np.max(np.abs(np.diff(multipliers))) <= env.lipschitz_constant() + 1e-7
    assert multipliers.max() == pytest.approx(1.6, abs=1e-3)
    assert multipliers.min() == pytest.approx(0.4, abs=1e-3)
    assert env.multiplier(0) == pytest.approx(env.multiplier(200), abs=1e-7)

    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.asarray([0.25], dtype=np.float32))
    assert info["directional_reward"] == pytest.approx(reward)
    assert "drift_multiplier" in info
    assert "drift_drag" in info


def test_point_drift_clock_survives_reset_and_vector_steps_are_global():
    env = DriftingPointMass(max_episode_steps=2, clock_scale=3)
    env.reset(seed=0)
    env.step(np.zeros(1, dtype=np.float32))
    env.step(np.zeros(1, dtype=np.float32))
    assert env.t == 6
    env.reset()
    assert env.t == 6

    envs = make_point_drift_vector_env(num_envs=3, asynchronous=False, max_episode_steps=20)
    envs.reset(seed=0)
    envs.step(np.zeros((3, 1), dtype=np.float32))
    assert all(child.t == 3 for child in envs.unwrapped.envs)
    envs.close()


def test_point_drift_vector_env_has_ppo_compatible_spaces():
    envs = make_point_drift_vector_env(num_envs=2, asynchronous=False)
    assert envs.single_observation_space.shape == (2,)
    assert envs.single_action_space.shape == (1,)
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    envs.close()


def test_cli_agent_wins_over_overlay_agent_field():
    cli = SimpleNamespace(agent="pt_full", config="simple_drift")
    cfg = build_config(cli)
    assert cfg["agent"] == "pt_full"
    assert cfg["env_mode"] == "point_drift"
