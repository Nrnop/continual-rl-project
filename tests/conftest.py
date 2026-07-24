import gymnasium as gym
import numpy as np
import pytest
import torch
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_ewc import PPOEWC
from src_continuous_control.envs.directional_half_cheetah import DirectionalHalfCheetah


class MockCheetahEnv(gym.Env):
    """Fast, deterministic CPU mock of HalfCheetah for unit testing."""
    def __init__(self, obs_dim=17, act_dim=6):
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, **kwargs):
        self.step_count = 0
        return np.ones(self.observation_space.shape, dtype=np.float32) * 0.1, {}

    def step(self, action):
        self.step_count += 1
        obs = np.ones(self.observation_space.shape, dtype=np.float32) * 0.1 * self.step_count
        reward_ctrl = -0.1 * float(np.sum(np.square(action)))
        x_velocity = 1.5  # Fixed simulated speed
        reward = x_velocity + reward_ctrl
        terminated = self.step_count >= 50
        truncated = False
        info = {"x_velocity": x_velocity, "reward_ctrl": reward_ctrl}
        return obs, reward, terminated, truncated, info


@pytest.fixture
def mock_base_env():
    return MockCheetahEnv()


@pytest.fixture
def mock_env(mock_base_env):
    return DirectionalHalfCheetah(env_id=mock_base_env, direction=1)


@pytest.fixture
def mock_vector_env():
    """Factory -> a SyncVectorEnv of DirectionalHalfCheetah(mock) with episode-stat recording.

    collect_rollout expects a gymnasium VectorEnv whose infos expose per-env `x_velocity`
    (aggregated to shape (num_envs,)) and `episode`/`_episode` on completion.
    """
    def _make(num_envs=2):
        def factory():
            return DirectionalHalfCheetah(env_id=MockCheetahEnv(), direction=1)
        envs = gym.vector.SyncVectorEnv([factory for _ in range(num_envs)])
        return gym.wrappers.vector.RecordEpisodeStatistics(envs)
    return _make


@pytest.fixture
def mock_cfg():
    return {
        "agent": "vanilla", "n_steps": 32, "step_by_step": True,
        "lr_actor": 3e-4, "lr_critic": 3e-4, "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_coef": 0.2, "epochs": 10, "minibatch_size": 32, "ent_coef": 0.01, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "hidden_sizes": [64, 64],
        "ewc_lambda": 100.0, "ewc_samples": 32,
        "lr_perm": 1e-5, "lr_trans": 3e-4, "k": 10
    }


@pytest.fixture(params=[PPOVanilla, PPOPT, PPOEWC])
def any_agent_cls(request):
    return request.param


@pytest.fixture
def any_agent(any_agent_cls, mock_cfg):
    obs_dim = 17
    act_dim = 6
    device = torch.device("cpu")
    cfg = dict(mock_cfg)
    if any_agent_cls == PPOVanilla:
        cfg["agent"] = "vanilla"
    elif any_agent_cls == PPOPT:
        cfg["agent"] = "pt"
    elif any_agent_cls == PPOEWC:
        cfg["agent"] = "ewc"
    return any_agent_cls(obs_dim, act_dim, cfg, device)
