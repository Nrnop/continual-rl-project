import copy
import gymnasium as gym
import numpy as np
import pytest
import torch
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_ewc import PPOEWC


def test_online_step_update_gradient_flow_and_weight_changes(any_agent):
    agent = any_agent
    # Clone initial state dict
    old_state = copy.deepcopy(agent.state_dict())

    obs = np.ones(17, dtype=np.float32) * 0.5
    action = np.ones(6, dtype=np.float32) * 0.1
    next_obs = np.ones(17, dtype=np.float32) * 0.6
    reward = 1.5
    done_flag = 0.0

    # The old logprob must be that of THIS action under the current policy, so the ratio starts at
    # 1 and the clipped surrogate is inside its trust region. Using the logprob of some other
    # sampled action can put the ratio outside the clip, where `torch.clamp` has zero gradient —
    # and `pt` has no entropy bonus to move the actor anyway, so the step becomes a silent no-op.
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, device=agent.device).unsqueeze(0)
        action_t = torch.as_tensor(action, device=agent.device).unsqueeze(0)
        logprob_t, entropy = agent.actor.evaluate_actions(obs_t, action_t)
        old_logprob = logprob_t.item()

    al, cl, ent, kl, ext = agent._online_step_update(obs, action, old_logprob, reward, next_obs, done_flag)

    new_state = agent.state_dict()

    # Check actor weights changed
    actor_changed = False
    for k in old_state["actor"]:
        if not torch.allclose(old_state["actor"][k], new_state["actor"][k]):
            actor_changed = True
            break
    assert actor_changed, "Actor weights did not update after _online_step_update"

    # Check critic weights changed and no NaNs
    if isinstance(agent, PPOPT):
        # After the PT correction, a single online step trains ONLY the transient critic;
        # the permanent head is frozen between consolidations. So trans must change, perm must NOT.
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
    obs_dim = 17
    act_dim = 6
    device = torch.device("cpu")
    agent = PPOVanilla(obs_dim, act_dim, mock_cfg, device)

    obs = np.zeros(17, dtype=np.float32)
    next_obs = np.ones(17, dtype=np.float32) * 10.0
    action = np.zeros(6, dtype=np.float32)

    # When done_flag = 1.0, target_val = reward + 0.0 (gamma * (1 - 1) * V(next_obs))
    # We verify it runs without error and returns valid losses
    al, cl, ent, kl, ext = agent._online_step_update(obs_np=obs, action_np=action, old_logprob_val=0.0, reward=5.0, next_obs_np=next_obs, done_flag=1.0)
    assert np.isfinite(al) and np.isfinite(cl)


def test_online_step_update_ewc_extra_loss_injection(mock_cfg):
    obs_dim = 17
    act_dim = 6
    device = torch.device("cpu")
    agent = PPOEWC(obs_dim, act_dim, mock_cfg, device)

    # Initially no past tasks
    assert agent._extra_loss().item() == 0.0

    # Fill buffer to compute Fisher and simulate a task switch
    for _ in range(32):
        agent.buffer.add(
            obs=np.zeros(17, dtype=np.float32),
            action=np.zeros(6, dtype=np.float32),
            logprob=0.0, reward=1.0, done=0.0, v_perm=0.5, v_trans=0.0
        )
    agent.on_task_switch(step=2000)
    assert len(agent.past_tasks) == 1

    # Now _extra_loss should be > 0 (or >= 0 if params haven't moved yet)
    # Let's perturb weights slightly so anchor distance > 0
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(0.1)

    extra_loss = agent._extra_loss()
    assert extra_loss.item() > 0.0, "EWC extra loss should be > 0 when parameters deviate from anchor"

    # Verify online step update backpropagates this penalty safely
    al, cl, ent, kl, ext = agent._online_step_update(
        obs_np=np.zeros(17, dtype=np.float32),
        action_np=np.zeros(6, dtype=np.float32),
        old_logprob_val=0.0, reward=1.0, next_obs_np=np.zeros(17, dtype=np.float32), done_flag=0.0
    )
    assert ext > 0.0
