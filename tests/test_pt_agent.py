import copy

import numpy as np
import pytest
import torch

from src_continuous_control.agents.ppo_pt import PPOPT


OBS_DIM = 5
ACT_DIM = 2


def _cfg(**overrides):
    cfg = {
        "n_steps": 4,
        "num_envs": 1,
        "lr_actor": 3e-4,
        "lr_trans": 3e-4,
        "lr_perm": 1e-3,
        "lr_perm_actor": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "epochs": 1,
        "minibatch_size": 4,
        "ent_coef": 0.0,
        "max_grad_norm": 10.0,
        "hidden_sizes": [8, 8],
        "critic_hidden_sizes": [8, 8],
        "trans_hidden_sizes": [4, 4],
        "k": 10,
        "consolidation_epochs": 2,
        "rho": 0.25,
        "rm_power": 0.5,
        "rm_power_actor": 0.75,
        "kl_prior_coef": 0.01,
        "target_kl": None,
        "normalize_advantage": False,
    }
    cfg.update(overrides)
    return cfg


def _agent(**overrides):
    torch.manual_seed(0)
    return PPOPT(OBS_DIM, ACT_DIM, _cfg(**overrides), torch.device("cpu"))


def _same_state(left, right):
    return all(torch.equal(left[name], right[name]) for name in left)


def _fill_rollout(agent):
    torch.manual_seed(1)
    np.random.seed(1)
    obs = np.linspace(-1.0, 1.0, agent.buffer.batch_size * OBS_DIM,
                      dtype=np.float32).reshape(agent.buffer.batch_size, OBS_DIM)
    actions = np.tile(np.array([[1.5, -1.25]], dtype=np.float32),
                      (agent.buffer.batch_size, 1))
    obs_t = torch.as_tensor(obs)
    with torch.no_grad():
        logprobs, _ = agent.actor.evaluate_actions(obs_t, torch.as_tensor(actions))
        v_perm, v_trans = agent.critic(obs_t)
    for index in range(agent.buffer.batch_size):
        agent.buffer.add(
            obs[index:index + 1],
            actions[index:index + 1],
            np.asarray([logprobs[index].item()], dtype=np.float32),
            np.asarray([1.0 + index], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([v_perm[index].item()], dtype=np.float32),
            np.asarray([v_trans[index].item()], dtype=np.float32),
        )
    return obs


def test_nonzero_entropy_coefficient_fails_at_construction():
    with pytest.raises(ValueError, match="ent_coef"):
        _agent(ent_coef=1e-3)


def test_online_update_moves_only_transient_parameters():
    agent = _agent()
    _fill_rollout(agent)
    perm_actor_before = copy.deepcopy(agent.actor.perm_mean.state_dict())
    perm_critic_before = copy.deepcopy(agent.critic.perm.state_dict())
    trans_actor_before = copy.deepcopy(agent.actor.trans_mean.state_dict())
    trans_critic_before = copy.deepcopy(agent.critic.trans.state_dict())
    log_std_before = agent.actor.log_std.detach().clone()

    metrics = agent.update(np.zeros((1, OBS_DIM), dtype=np.float32), np.zeros(1), 0)

    assert {"actor_loss", "critic_loss", "kl_prior", "entropy", "approx_kl",
            "clip_fraction"}.issubset(metrics)
    assert {"value_perm_l2", "value_trans_l2", "policy_perm_l2", "policy_trans_l2",
            "log_std_mean", "grad_norm_perm_actor", "grad_norm_perm_critic"}.issubset(metrics)
    assert metrics["grad_norm_perm_actor"] == 0.0
    assert metrics["grad_norm_perm_critic"] == 0.0
    assert _same_state(perm_actor_before, agent.actor.perm_mean.state_dict())
    assert _same_state(perm_critic_before, agent.critic.perm.state_dict())
    assert not _same_state(trans_actor_before, agent.actor.trans_mean.state_dict())
    assert not _same_state(trans_critic_before, agent.critic.trans.state_dict())
    assert torch.equal(log_std_before, agent.actor.log_std.detach())  # frozen, Constraint C4
    assert not agent.perm_actor_optim.state
    assert not agent.perm_optim.state


def _populate_optimizer_state(agent, states):
    states_t = torch.as_tensor(states)
    actor_loss = agent.actor.trans_mean(states_t).sum()
    agent.actor_optim.zero_grad()
    actor_loss.backward()
    agent.actor_optim.step()

    critic_loss = agent.critic.trans(states_t).sum()
    agent.trans_optim.zero_grad()
    critic_loss.backward()
    agent.trans_optim.step()


def test_consolidation_uses_entry_snapshot_and_flushes_only_transient_state():
    agent = _agent(k=10, rho=0.25, consolidation_epochs=2)
    states = np.linspace(-0.8, 0.8, 12 * OBS_DIM, dtype=np.float32).reshape(12, OBS_DIM)
    with torch.no_grad():
        torch.nn.init.normal_(agent.actor.trans_mean[-1].weight)
        torch.nn.init.normal_(agent.actor.trans_mean[-1].bias)
        torch.nn.init.normal_(agent.critic.trans[-1].weight)
        torch.nn.init.normal_(agent.critic.trans[-1].bias)
        states_t = torch.as_tensor(states)
    _populate_optimizer_state(agent, states)
    log_std_before = agent.actor.log_std.detach().clone()
    with torch.no_grad():
        actor_trans_before = agent.actor.trans_mean(states_t).clone()
        critic_trans_before = agent.critic.trans(states_t).clone()
    assert agent.actor.log_std not in agent.actor_optim.state  # frozen, never optimized
    assert any(p in agent.actor_optim.state for p in agent.actor.trans_mean.parameters())
    assert any(p in agent.trans_optim.state for p in agent.critic.trans.parameters())

    agent.consolidation_buffer.add_batch(states)
    record = agent._consolidate()

    assert record["buffer_size"] == len(states)
    assert record["alpha_p_used"] == pytest.approx(1e-3)
    assert record["alpha_p_actor_used"] == pytest.approx(1e-3)
    assert agent._n_consolidations == 1
    assert len(agent.consolidation_buffer) == 0
    assert agent._updates_since_consolidation == 0
    assert agent.perm_optim.param_groups[0]["lr"] == pytest.approx(1e-3 / 2**0.5)
    assert agent.perm_actor_optim.param_groups[0]["lr"] == pytest.approx(1e-3 / 2**0.75)
    assert torch.equal(log_std_before, agent.actor.log_std.detach())
    assert torch.allclose(agent.actor.trans_mean(states_t), 0.75 * actor_trans_before)
    assert torch.allclose(agent.critic.trans(states_t), 0.75 * critic_trans_before,
                           atol=1e-6, rtol=1e-5)
    assert agent.actor.log_std not in agent.actor_optim.state
    assert all(p not in agent.actor_optim.state for p in agent.actor.trans_mean.parameters())
    assert all(p not in agent.trans_optim.state for p in agent.critic.trans.parameters())
    assert agent._pre_consolidation_snapshot is not None
    assert not hasattr(agent.consolidation_buffer, "old_v_perm")

    agent.consolidation_buffer.add_batch(states)
    second = agent._consolidate()
    assert second["alpha_p_used"] == pytest.approx(1e-3 / 2**0.5)
    assert second["alpha_p_actor_used"] == pytest.approx(1e-3 / 2**0.75)
    assert agent._n_consolidations == 2

    fresh_states = np.ones((3, OBS_DIM), dtype=np.float32)
    drift = agent.measure_post_consolidation_drift(fresh_states)
    assert set(drift) == {"delta_v", "delta_pi"}
    assert agent.measure_post_consolidation_drift(fresh_states) is None


def test_post_consolidation_drift_does_not_consume_rng():
    agent = _agent(k=10)
    states = np.zeros((4, OBS_DIM), dtype=np.float32)
    agent.consolidation_buffer.add_batch(states)
    agent._consolidate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    agent.measure_post_consolidation_drift(np.ones((3, OBS_DIM), dtype=np.float32))

    numpy_after = np.random.get_state()
    torch_after = torch.get_rng_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch_after)


def test_rho_one_preserves_composed_value_and_policy_on_overfit_toy():
    agent = _agent(
        lr_actor=0.05,
        lr_trans=0.05,
        lr_perm=0.05,
        lr_perm_actor=0.05,
        rho=1.0,
        rm_power=0.0,
        rm_power_actor=0.0,
        perm_zero_init=True,
        consolidation_epochs=200,
    )
    states = torch.zeros(4, OBS_DIM)
    with torch.no_grad():
        agent.critic.trans[-1].weight.zero_()
        agent.critic.trans[-1].bias.fill_(2.0)
        agent.actor.trans_mean[-1].weight.zero_()
        agent.actor.trans_mean[-1].bias.fill_(0.5)
        value_before = agent.critic.value(states).clone()
        policy_before = agent.actor.act_deterministic(states).clone()

    agent.consolidation_buffer.add_batch(states.numpy())
    agent._consolidate()

    with torch.no_grad():
        value_after = agent.critic.value(states)
        policy_after = agent.actor.act_deterministic(states)
    assert torch.max(torch.abs(value_after - value_before)) < 1e-3
    assert torch.max(torch.abs(policy_after - policy_before)) < 1e-3


def test_switch_clears_pending_states_without_consolidation():
    agent = _agent(k=10)
    states = np.zeros((3, OBS_DIM), dtype=np.float32)
    agent.consolidation_buffer.add_batch(states)
    agent._updates_since_consolidation = 3
    agent.on_task_switch(100)
    assert len(agent.consolidation_buffer) == 0
    assert agent._updates_since_consolidation == 0
    assert agent._n_consolidations == 1
