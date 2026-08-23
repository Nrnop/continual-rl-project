import numpy as np
import torch

from src_continuous_control.agents.ppo_pt import PPOPT


def make_cfg():
    return {
        "lr_actor": 1e-3,
        "n_steps": 2,
        "num_envs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "epochs": 1,
        "minibatch_size": 2,
        "ent_coef": 0.0,
        "max_grad_norm": 0.5,
        "normalize_advantage": False,
        "rho": 0.5,
        "kl_prior_coef": 0.01,
        "k": 1,
        "consolidation_epochs": 1,
        "critic_hidden_sizes": [8, 8],
        "hidden_sizes": [8, 8],
        "actor_trans_hidden_sizes": [4, 4],
        "critic_trans_hidden_sizes": [4, 4],
        "lr_trans": 1e-3,
        "lr_perm": 1e-3,
        "perm_optimizer": "adam",
        "rm_power": 0.5,
        "consolidation_buffer_size": 20,
    }


def test_consolidation_flush_and_stats():
    cfg = make_cfg()
    obs_dim = 3
    act_dim = 2
    device = "cpu"
    agent = PPOPT(obs_dim, act_dim, cfg, device)

    # Add synthetic states to the consolidation buffer
    states = np.random.randn(10, obs_dim).astype(np.float32)
    agent.consolidation_buffer.add_batch(states)

    # Create optimizer state entries for transient optimizers by doing a dummy step
    t_states = torch.as_tensor(states[:4], dtype=torch.float32)
    loss_t = agent.critic.trans(t_states).mean()
    agent.trans_optim.zero_grad()
    loss_t.backward()
    agent.trans_optim.step()

    loss_at = agent.actor.trans_mean(t_states).mean()
    agent.actor_optim.zero_grad()
    loss_at.backward()
    agent.actor_optim.step()

    trans_params = list(agent.critic.trans.parameters())
    act_trans_params = list(agent.actor.trans_mean.parameters())
    assert any(p in agent.trans_optim.state for p in trans_params)
    assert any(p in agent.actor_optim.state for p in act_trans_params)

    # Run consolidation and inspect post-conditions
    record = agent._consolidate()
    assert record is not None
    assert len(agent.consolidation_buffer) == 0
    assert agent._updates_since_consolidation == 0

    # Transient optimizers must have their parameter states flushed
    assert not any(p in agent.trans_optim.state for p in trans_params)
    assert not any(p in agent.actor_optim.state for p in act_trans_params)

    # Consolidation metadata
    assert agent._n_consolidations >= 1
    assert "absorbed_frac" in record and "actor_absorbed_frac" in record
