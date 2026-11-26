import copy
import gymnasium as gym
import numpy as np
import pytest
import torch
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_ewc import PPOEWC


def test_online_step_update_gradient_flow_and_weight_changes(any_agent):
    """One online step updates actor and critic weights and produces no NaNs."""
    agent = any_agent
    old_state = copy.deepcopy(agent.state_dict())

    obs = np.ones(17, dtype=np.float32) * 0.5
    action = np.ones(6, dtype=np.float32) * 0.1
    next_obs = np.ones(17, dtype=np.float32) * 0.6
    reward = 1.5
    done_flag = 0.0

    # Use the logprob of this action under the current policy so the ratio starts at 1, inside
    # the clip. Outside it torch.clamp has zero gradient, and since `pt` has no entropy bonus the
    # actor would not move at all.
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, device=agent.device).unsqueeze(0)
        action_t = torch.as_tensor(action, device=agent.device).unsqueeze(0)
        logprob_t, entropy = agent.actor.evaluate_actions(obs_t, action_t)
        old_logprob = logprob_t.item()

    al, cl, ent, kl, ext = agent._online_step_update(obs, action, old_logprob, reward, next_obs, done_flag)

    new_state = agent.state_dict()

    actor_changed = False
    for k in old_state["actor"]:
        if not torch.allclose(old_state["actor"][k], new_state["actor"][k]):
            actor_changed = True
            break
    assert actor_changed, "Actor weights did not update after _online_step_update"

    if isinstance(agent, PPOPT):
        # An online step trains only the transient critic; the permanent changes only at
        # consolidation.
        perm_changed = any(not torch.allclose(old_state["critic"][k], new_state["critic"][k]) for k in old_state["critic"] if k.startswith("perm."))
        trans_changed = any(not torch.allclose(old_state["critic"][k], new_state["critic"][k]) for k in old_state["critic"] if k.startswith("trans."))
        assert trans_changed, "PT transient critic weights did not update"
        assert not perm_changed, "PT permanent critic must NOT change on an online step (only at consolidation)"
        for k, v in new_state["critic"].items():
            assert not torch.isnan(v).any(), f"NaN found in critic parameter {k}"
    else:
        critic_changed = any(not torch.allclose(old_state["critic"][k], new_state["critic"][k]) for k in old_state["critic"])
        assert critic_changed, f"Critic weights did not update for {type(agent).__name__}"
        for k, v in new_state["critic"].items():
            assert not torch.isnan(v).any(), f"NaN found in critic parameter {k}"

    for k, v in new_state["actor"].items():
        assert not torch.isnan(v).any(), f"NaN found in actor parameter {k}"


def test_online_step_update_td_target_and_done_masking(mock_cfg):
    """With done_flag=1 the online update runs and returns finite losses."""
    obs_dim = 17
    act_dim = 6
    device = torch.device("cpu")
    agent = PPOVanilla(obs_dim, act_dim, mock_cfg, device)

    obs = np.zeros(17, dtype=np.float32)
    next_obs = np.ones(17, dtype=np.float32) * 10.0
    action = np.zeros(6, dtype=np.float32)

    # With done_flag = 1.0 the TD target is the reward alone: gamma * (1 - 1) * V(next_obs) = 0.
    al, cl, ent, kl, ext = agent._online_step_update(obs_np=obs, action_np=action, old_logprob_val=0.0, reward=5.0, next_obs_np=next_obs, done_flag=1.0)
    assert np.isfinite(al) and np.isfinite(cl)


def test_online_step_update_ewc_extra_loss_injection(mock_cfg):
    """EWC's penalty is positive once weights leave the anchor, and the online step reports it."""
    obs_dim = 17
    act_dim = 6
    device = torch.device("cpu")
    agent = PPOEWC(obs_dim, act_dim, mock_cfg, device)

    # No penalty before the first task switch.
    assert agent._extra_loss().item() == 0.0

    # Fill the buffer so the task switch has samples for the Fisher.
    for _ in range(32):
        agent.buffer.add(
            obs=np.zeros(17, dtype=np.float32),
            action=np.zeros(6, dtype=np.float32),
            logprob=0.0, reward=1.0, done=0.0, v_perm=0.5, v_trans=0.0
        )
    agent.on_task_switch(step=2000)
    assert len(agent.past_tasks) == 1

    # Move the weights off the anchor so the penalty is non-zero.
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(0.1)

    extra_loss = agent._extra_loss()
    assert extra_loss.item() > 0.0, "EWC extra loss should be > 0 when parameters deviate from anchor"

    # The online step includes the penalty as its extra loss.
    al, cl, ent, kl, ext = agent._online_step_update(
        obs_np=np.zeros(17, dtype=np.float32),
        action_np=np.zeros(6, dtype=np.float32),
        old_logprob_val=0.0, reward=1.0, next_obs_np=np.zeros(17, dtype=np.float32), done_flag=0.0
    )
    assert ext > 0.0
