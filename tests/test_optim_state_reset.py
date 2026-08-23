"""Decaying the transient must clear its optimiser state as well as its weights.

Without this, Adam keeps exp_avg / exp_avg_sq for parameters that were just scaled (to zero, when
rho=1), so the next update displaces them again using momentum from a network that no longer
exists — undoing the consolidation just established. See FINDINGS.md 5.6.

PORTED IN PHASE 2 (T2b). The old critic-only agent made this optional behind
`reset_trans_optim_on_decay`, defaulting off so earlier runs reproduced. The surviving agent makes
it UNCONDITIONAL (Constraint C2) and does it for the actor's transient as well as the critic's, so
the tests assert the realised behaviour rather than a config key — failure mode #1 in CLAUDE.md.
"""
import copy

import numpy as np
import torch

from src_continuous_control.agents.ppo_pt import PPOPT


def _cfg(**kw):
    c = dict(hidden_sizes=[32, 32], lr_actor=3e-4, adam_eps=1e-5, num_envs=2, n_steps=8,
             gamma=0.99, gae_lambda=0.95, clip_coef=0.2, epochs=1, minibatch_size=8,
             ent_coef=0.0, max_grad_norm=0.5, target_kl=None, normalize_advantage=True,
             lr_trans=3e-3, lr_perm=1e-3, lr_perm_actor=1e-3, perm_optimizer="adam",
             rho=1.0, k=1, kl_prior_coef=0.01, consolidation_epochs=1,
             consolidation_buffer_size=64)
    c.update(kw)
    return c


def _agent(**kw):
    torch.manual_seed(0)
    return PPOPT(17, 6, _cfg(**kw), torch.device("cpu"))


def _build_momentum(agent, steps=6):
    """Take a few real optimiser steps so Adam accumulates state for both transients."""
    for _ in range(steps):
        obs = torch.randn(16, 17)
        v_perm, v_trans = agent.critic(obs)
        loss = ((v_perm.detach() + v_trans - torch.randn(16)) ** 2).mean()
        agent.trans_optim.zero_grad(); loss.backward(); agent.trans_optim.step()

        mu_loss = ((agent.actor.trans_mean(obs) - torch.randn(16, 6)) ** 2).mean()
        agent.actor_optim.zero_grad(); mu_loss.backward(); agent.actor_optim.step()


def _has_state(optimizer, module):
    return any(optimizer.state.get(p) for p in module.parameters())


def _consolidate_on(agent, n=32):
    states = np.random.randn(n, 17).astype(np.float32)
    agent.consolidation_buffer.add_batch(states)
    return agent._consolidate()


def test_consolidation_flushes_the_transient_optimiser_state_of_both_components():
    """Unconditional in this agent — there is no flag to get wrong."""
    agent = _agent()
    _build_momentum(agent)
    assert _has_state(agent.trans_optim, agent.critic.trans)
    assert _has_state(agent.actor_optim, agent.actor.trans_mean)

    _consolidate_on(agent)

    assert not _has_state(agent.trans_optim, agent.critic.trans), \
        "the critic's transient optimiser state survived a consolidation"
    assert not _has_state(agent.actor_optim, agent.actor.trans_mean), \
        "the actor's transient optimiser state survived a consolidation"


def test_flush_stops_the_weights_springing_back_from_stale_momentum():
    """The point of the flush: after rho=1 zeroes the output layer, the next step must not undo it.

    `restored=True` puts the pre-consolidation Adam moments back by hand, which is exactly what the
    old default (no flush) did — so the two arms differ only in whether the state was cleared.
    """
    moved = {}
    for restored in (True, False):
        agent = _agent(rho=1.0)                          # decay factor 1-rho = 0
        _build_momentum(agent)
        saved = {p: copy.deepcopy(agent.trans_optim.state[p])
                 for p in agent.critic.trans.parameters() if p in agent.trans_optim.state}
        _consolidate_on(agent)

        out = agent.critic.trans[-1]
        assert torch.allclose(out.weight, torch.zeros_like(out.weight))
        assert torch.allclose(out.bias, torch.zeros_like(out.bias))
        if restored:
            for p, state in saved.items():
                agent.trans_optim.state[p] = state

        # one more optimiser step with a ZERO gradient: any movement is momentum alone
        agent.trans_optim.zero_grad()
        for p in agent.critic.trans.parameters():
            p.grad = torch.zeros_like(p)
        agent.trans_optim.step()
        moved[restored] = max(float(out.weight.detach().abs().max()),
                              float(out.bias.detach().abs().max()))

    assert moved[True] > 0, "expected stale momentum to move the zeroed weights"
    assert moved[False] == 0.0, (
        f"the flush must leave the zeroed transient at zero (moved {moved[False]:.2e} "
        f"vs {moved[True]:.2e} with the state restored)")


def test_consolidation_leaves_the_permanent_optimiser_state_alone():
    """theta_P lives in its own optimisers, which must survive the transient flush."""
    agent = _agent()
    _build_momentum(agent)
    _consolidate_on(agent)
    assert _has_state(agent.perm_optim, agent.critic.perm), \
        "the critic's permanent optimiser state must survive a consolidation"
    assert _has_state(agent.perm_actor_optim, agent.actor.perm_mean), \
        "the actor's permanent optimiser state must survive a consolidation"


def test_the_flush_never_touches_log_std():
    """log_std is frozen (Constraint C4): never optimised, so never in any optimiser's state.

    A flush that reached it would be evidence the exploration schedule is being trained on one arm
    and not the others — failure mode #3 in CLAUDE.md, and the confound that invalidated a 24-run
    sweep in Phase 1.
    """
    agent = _agent()
    _build_momentum(agent)
    before = agent.actor.log_std.detach().clone()
    _consolidate_on(agent)
    assert agent.actor.log_std not in agent.actor_optim.state
    assert agent.actor.log_std.requires_grad is False
    assert torch.equal(before, agent.actor.log_std.detach())
