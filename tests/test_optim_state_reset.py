"""Decaying the transient must be able to clear its optimiser state as well as its weights.

Without this, Adam keeps exp_avg / exp_avg_sq for parameters that were just scaled (to zero, when
decay=0), so the next update displaces them again using momentum from a network that no longer
exists — undoing the consistency consolidation just established. See FINDINGS.md 5.6.
"""
import numpy as np
import torch

from src_continuous_control.agents.ppo_pt import PPOPT


def _cfg(**kw):
    c = dict(hidden_sizes=[32, 32], lr_actor=3e-4, adam_eps=1e-5, num_envs=2, n_steps=8,
             gamma=0.99, gae_lambda=0.95, clip_coef=0.2, epochs=1, minibatch_size=8,
             ent_coef=0.0, max_grad_norm=0.5, target_kl=None, normalize_advantage=True,
             lr_trans=3e-3, lr_perm=1e-3, perm_optimizer="adam", decay=0.0, k=1,
             consolidation_epochs=1, consolidation_buffer_size=64, on_switch="consolidate")
    c.update(kw)
    return c


def _build_momentum(agent, steps=6):
    """Take a few real optimiser steps so Adam accumulates state for the transient."""
    for _ in range(steps):
        obs = torch.randn(16, 17)
        v_perm, v_trans = agent.critic(obs)
        loss = ((v_perm.detach() + v_trans - torch.randn(16)) ** 2).mean()
        agent.trans_optim.zero_grad()
        loss.backward()
        agent.trans_optim.step()


def _has_state(agent):
    return any(agent.trans_optim.state.get(p) for p in agent.critic.trans.parameters())


def test_default_keeps_optimiser_state_unchanged_behaviour():
    agent = PPOPT(17, 6, _cfg(), torch.device("cpu"))
    assert agent.reset_trans_optim_on_decay is False
    _build_momentum(agent)
    assert _has_state(agent)
    agent._decay_transient(0.0)
    assert _has_state(agent), "default behaviour must leave the optimiser state in place"


def test_reset_clears_transient_optimiser_state():
    agent = PPOPT(17, 6, _cfg(reset_trans_optim_on_decay=True), torch.device("cpu"))
    _build_momentum(agent)
    assert _has_state(agent)
    agent._decay_transient(0.0)
    assert not _has_state(agent), "optimiser state for the transient should have been cleared"


def test_reset_stops_the_weights_springing_back_from_stale_momentum():
    """The point of the fix: after a hard reset the next step must not undo it."""
    moved = {}
    for reset in (False, True):
        torch.manual_seed(0)
        agent = PPOPT(17, 6, _cfg(reset_trans_optim_on_decay=reset), torch.device("cpu"))
        _build_momentum(agent)
        agent._decay_transient(0.0)                      # weights -> exactly 0
        after_decay = torch.cat([p.detach().flatten().clone()
                                 for p in agent.critic.trans.parameters()])
        assert torch.allclose(after_decay, torch.zeros_like(after_decay))
        # one more optimiser step with a ZERO gradient: any movement is momentum alone
        agent.trans_optim.zero_grad()
        for p in agent.critic.trans.parameters():
            p.grad = torch.zeros_like(p)
        agent.trans_optim.step()
        moved[reset] = torch.cat([p.detach().flatten()
                                  for p in agent.critic.trans.parameters()]).abs().max().item()
    assert moved[False] > 0, "expected stale momentum to move the zeroed weights"
    assert moved[True] < moved[False], (
        f"resetting the state should reduce the springback (no-reset {moved[False]:.2e} "
        f"vs reset {moved[True]:.2e})")


def test_shared_trunk_reset_preserves_trunk_state():
    """The shared-trunk optimiser also holds the trunk; only the head's state may be dropped."""
    agent = PPOPT(17, 6, _cfg(critic_arch="shared_trunk", decay=0.5,
                              reset_trans_optim_on_decay=True), torch.device("cpu"))
    for _ in range(6):
        obs = torch.randn(16, 17)
        v_perm, v_trans = agent.critic(obs)
        loss = ((v_perm.detach() + v_trans - torch.randn(16)) ** 2).mean()
        agent.trans_optim.zero_grad(); loss.backward(); agent.trans_optim.step()
    assert any(agent.trans_optim.state.get(p) for p in agent.critic.trunk.parameters())
    agent._decay_transient(0.5)
    assert any(agent.trans_optim.state.get(p) for p in agent.critic.trunk.parameters()), \
        "trunk optimiser state must survive a transient-head reset"
