"""Tests for PPOPT consolidation and the decay_match options."""
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
    """Consolidation empties the buffer, flushes transient optimizer state, and logs stats."""
    cfg = make_cfg()
    obs_dim = 3
    act_dim = 2
    device = "cpu"
    agent = PPOPT(obs_dim, act_dim, cfg, device)

    states = np.random.randn(10, obs_dim).astype(np.float32)
    agent.consolidation_buffer.add_batch(states)

    # A dummy step creates optimizer state for the transient parameters.
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

    record = agent._consolidate()
    assert record is not None
    assert len(agent.consolidation_buffer) == 0
    assert agent._updates_since_consolidation == 0

    # Consolidation flushes the transient optimizer state.
    assert not any(p in agent.trans_optim.state for p in trans_params)
    assert not any(p in agent.actor_optim.state for p in act_trans_params)

    assert agent._n_consolidations >= 1
    assert "absorbed_frac" in record and "actor_absorbed_frac" in record


# decay_match="realised": decay the transient by what the permanent actually absorbed.

def _primed_agent(**overrides):
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = make_cfg()
    cfg.update(overrides)
    agent = PPOPT(3, 2, cfg, "cpu")
    # Transients start at zero (Theorem 1); give them a non-zero output to transfer.
    with torch.no_grad():
        for layer in (agent.critic.trans[-1], agent.actor.trans_mean[-1]):
            layer.weight.normal_(0.0, 0.5)
            layer.bias.normal_(0.0, 0.5)
    agent.consolidation_buffer.add_batch(np.random.randn(16, 3).astype(np.float32))
    return agent


def test_matched_decay_factors_recover_known_absorption():
    """_matched_decay_factors recovers a known per-unit absorption fraction."""
    torch.manual_seed(0)
    before = torch.randn(50, 2)
    trans = torch.randn(50, 2)
    rho = 0.5
    # Unit 0 absorbed 30% of rho*f_T, unit 1 absorbed all of it.
    after = before + torch.stack([0.3 * rho * trans[:, 0], rho * trans[:, 1]], dim=1)
    factors, c = PPOPT._matched_decay_factors(before, after, trans, rho)
    assert torch.allclose(c, torch.tensor([0.3, 1.0]), atol=1e-5)
    assert torch.allclose(factors, torch.tensor([1 - 0.3 * rho, 1 - rho]), atol=1e-5)


def test_matched_decay_factors_clip_and_zero_transient():
    """Absorption fractions are clipped to [0, 1] and are 0 where the transient is zero."""
    before = torch.zeros(10, 3)
    trans = torch.ones(10, 3)
    trans[:, 2] = 0.0  # nothing to transfer on unit 2
    after = torch.stack([-0.5 * trans[:, 0], 3.0 * trans[:, 1], torch.ones(10)], dim=1)
    factors, c = PPOPT._matched_decay_factors(before, after, trans, 0.5)
    assert torch.equal(c, torch.tensor([0.0, 1.0, 0.0]))  # negative -> 0, overshoot -> 1, none -> 0
    assert torch.equal(factors, torch.tensor([1.0, 0.5, 1.0]))


def test_realised_decay_preserves_composition_along_transient():
    """Realised decay keeps the composition along f_T unchanged per unit; fixed decay does not."""
    for mode, should_vanish in (("realised", True), ("fixed", False)):
        agent = _primed_agent(decay_match=mode)
        states = torch.as_tensor(agent.consolidation_buffer.as_array())
        with torch.no_grad():
            vp0, vt0 = agent.critic(states)
            mp0, mt0 = agent.actor.perm_forward(states), agent.actor.trans_mean(states)
        record = agent._consolidate()
        with torch.no_grad():
            vp1, vt1 = agent.critic(states)
            mp1, mt1 = agent.actor.perm_forward(states), agent.actor.trans_mean(states)
        d_critic = (((vp1 + vt1) - (vp0 + vt0)) * vt0).sum() / (vt0 * vt0).sum()
        d_actor = (((mp1 + mt1) - (mp0 + mt0)) * mt0).sum(0) / (mt0 * mt0).sum(0)
        if should_vanish:
            c_crit, c_act = record["critic_matched_c"], record["actor_matched_c"]
            assert 0.0 < c_crit[0] < 1.0 and np.all((c_act > 0.0) & (c_act < 1.0)), \
                "test needs unclipped c to be meaningful"
            assert abs(float(d_critic)) < 1e-4
            assert torch.allclose(d_actor, torch.zeros(2), atol=1e-4)
            assert torch.allclose(mt1, mt0 * torch.as_tensor(record["actor_decay_factor"]), atol=1e-6)
        else:
            assert record["actor_matched_c"] is None and record["actor_decay_factor"] is None
            assert torch.allclose(mt1, 0.5 * mt0, atol=1e-6)  # the fixed (1 - rho)
            assert torch.all(d_actor < -1e-3)  # under-absorption leaks out of the composition


def test_realised_and_explicit_decay_rho_conflict():
    """decay_match="realised" with decay_rho, or an unknown decay_match, raises ValueError."""
    import pytest
    with pytest.raises(ValueError):
        PPOPT(3, 2, {**make_cfg(), "decay_match": "realised", "decay_rho": 0.3}, "cpu")
    with pytest.raises(ValueError):
        PPOPT(3, 2, {**make_cfg(), "decay_match": "bogus"}, "cpu")


def test_fixed_is_the_default_and_bit_identical():
    """The default decay_match gives results identical to decay_match="fixed"."""
    a = _primed_agent()
    b = _primed_agent(decay_match="fixed")
    a._consolidate()
    b._consolidate()
    for pa, pb in zip(list(a.actor.parameters()) + list(a.critic.parameters()),
                      list(b.actor.parameters()) + list(b.critic.parameters())):
        assert torch.equal(pa, pb)
