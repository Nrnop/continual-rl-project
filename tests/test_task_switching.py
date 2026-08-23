import gymnasium as gym
import numpy as np
import pytest
import torch
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_ewc import PPOEWC
from src_continuous_control.envs.directional_half_cheetah import DirectionalHalfCheetah
from src_continuous_control.tests.conftest import MockCheetahEnv


def test_directional_half_cheetah_reward_flip_and_info_sync():
    base_env = MockCheetahEnv()
    env = DirectionalHalfCheetah(env_id=base_env, direction=1, forward_reward_weight=1.0)

    action = np.zeros(6, dtype=np.float32)
    obs, reward_fwd, terminated, truncated, info = env.step(action)
    assert info["direction"] == 1
    # MockCheetahEnv returns x_velocity = 1.5, reward_ctrl = 0.0 for zero action
    assert np.isclose(reward_fwd, 1.5)

    # Flip task to backward (-1)
    env.set_task(-1)
    obs, reward_bwd, terminated, truncated, info = env.step(action)
    assert info["direction"] == -1
    assert np.isclose(reward_bwd, -1.5)

    # Check reset preserves direction
    env.reset()
    assert env.direction == -1


def test_task_boundary_trigger_at_614400():
    switch = 614400
    # Simulate boundary detection logic in train.py:
    # if global_step > 0 and global_step % switch_interval == 0: boundary_tracker.on_boundary(global_step)
    steps_to_test = [0, 2048, 614399, 614400, 614401, 1228800]
    triggered = []
    for step in steps_to_test:
        if step > 0 and step % switch == 0:
            triggered.append(step)
    assert triggered == [614400, 1228800]


def test_agent_on_task_switch_hooks(mock_cfg):
    obs_dim = 17
    act_dim = 6
    device = torch.device("cpu")

    # Vanilla and PT should run on_task_switch safely without raising errors
    vanilla = PPOVanilla(obs_dim, act_dim, mock_cfg, device)
    # `pt` regularizes with KL-to-permanent, not entropy, and refuses ent_coef != 0.
    pt_cfg = dict(mock_cfg, ent_coef=0.0, kl_prior_coef=0.01, rho=0.5)
    pt = PPOPT(obs_dim, act_dim, pt_cfg, device)
    vanilla.on_task_switch(step=614400)
    pt.on_task_switch(step=614400)

    # EWC should compute Fisher over buffer samples and increment past_tasks
    ewc = PPOEWC(obs_dim, act_dim, mock_cfg, device)
    for _ in range(32):
        ewc.buffer.add(
            obs=np.zeros(17, dtype=np.float32),
            action=np.zeros(6, dtype=np.float32),
            logprob=0.0, reward=1.0, done=0.0, v_perm=0.5, v_trans=0.0
        )
    assert len(ewc.past_tasks) == 0
    assert ewc.fisher == {}, "no Fisher information exists before the first boundary"
    ewc.on_task_switch(step=614400)
    assert len(ewc.past_tasks) == 1

    # Online EWC keeps ONE running Fisher and ONE anchor, not a list of per-task pairs.
    assert ewc.fisher and ewc.anchor
    for k, f_vec in ewc.fisher.items():
        assert (f_vec >= 0.0).all(), f"Negative Fisher value found in {k}"

    # A second boundary must ACCUMULATE into the same estimate, decayed by gamma, rather than
    # appending another anchor — `ewc_gamma` was a dead config key until 2026-08-12, and the
    # penalty grew without bound as tasks accumulated.
    before = {k: v.clone() for k, v in ewc.fisher.items()}
    ewc.on_task_switch(step=1228800)
    assert len(ewc.fisher) == len(before), "the running Fisher must not grow a second copy"
    grew = any(float(ewc.fisher[k].sum()) > float(before[k].sum()) for k in before)
    assert grew, "the second boundary did not accumulate into the running Fisher"
