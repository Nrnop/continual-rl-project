import os
import pickle
import numpy as np
import pytest
import torch
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.utils.logger import Logger
from src_continuous_control.plots.plot_compare import _load_returns


def test_collect_rollout_velocity_extraction(mock_vector_env, mock_cfg):
    num_envs = 2
    n_steps = 10
    envs = mock_vector_env(num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    device = torch.device("cpu")
    cfg = dict(mock_cfg)
    cfg["n_steps"] = n_steps
    cfg["num_envs"] = num_envs
    agent = PPOVanilla(obs_dim, act_dim, cfg, device)

    obs, _ = envs.reset(seed=0)
    done = np.zeros(num_envs, dtype=np.float32)
    last_obs, last_done, ep_returns = agent.collect_rollout(envs, obs, done)

    # One mean x_velocity is recorded per step of the rollout.
    assert len(agent._velocities) == n_steps
    # MockCheetahEnv returns a fixed x_velocity = 1.5 for every env.
    assert np.isclose(np.mean(agent._velocities), 1.5)
    # The rollout buffer is filled to (n_steps, num_envs, obs_dim).
    assert agent.buffer.obs.shape == (n_steps, num_envs, obs_dim)
    assert last_obs.shape == (num_envs, obs_dim)


def test_returns_and_velocity_curves_serialization_shape(tmp_path):
    logger = Logger(exp_name="vanilla_ppo", seed=0, backend="none", results_dir=str(tmp_path))
    curve_data = [(2048, -15.2), (4096, -10.5), (6144, -8.0)]
    logger.save_returns(curve_data, suffix="test_curve")

    file_path = tmp_path / "vanilla_ppo_seed_0_test_curve.pkl"
    assert file_path.exists()

    with open(file_path, "rb") as f:
        loaded = pickle.load(f)
    arr = np.asarray(loaded, dtype=np.float32)

    assert arr.ndim == 2
    assert arr.shape == (3, 2)
    assert np.all(arr[:, 0] == [2048, 4096, 6144])
    assert np.allclose(arr[:, 1], [-15.2, -10.5, -8.0])
    assert np.isfinite(arr).all()


def test_plot_compare_load_returns_backward_compatibility(tmp_path):
    # Save a 2D format file (new) and a 1D format file (legacy)
    new_data = [[2048.0, -15.0], [4096.0, -10.0]]
    old_data = [-15.0, -10.0]

    with open(tmp_path / "vanilla_ppo_seed_0_returns.pkl", "wb") as f:
        pickle.dump(new_data, f)
    with open(tmp_path / "pt_ppo_seed_0_returns.pkl", "wb") as f:
        pickle.dump(old_data, f)

    # Load 2D data
    y_van, x_van = _load_returns(str(tmp_path), "vanilla", [0], n_steps=2048)
    assert len(y_van) == 1 and len(x_van) == 1
    assert np.allclose(x_van[0], [2048.0, 4096.0])
    assert np.allclose(y_van[0], [-15.0, -10.0])

    # Load 1D data (legacy compatibility: infers step as idx * n_steps)
    y_pt, x_pt = _load_returns(str(tmp_path), "pt", [0], n_steps=2048)
    assert len(y_pt) == 1 and len(x_pt) == 1
    assert np.allclose(x_pt[0], [0.0, 2048.0])
    assert np.allclose(y_pt[0], [-15.0, -10.0])
