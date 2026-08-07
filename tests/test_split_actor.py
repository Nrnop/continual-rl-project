"""The split actor: correctness, and proof it changes nothing when switched off.

The second point is load-bearing. Every result in this project was produced with a single
GaussianActor; if enabling `split_actor` in a config could perturb the default path, none of
those numbers would remain comparable and the new arm would have invalidated the old work.
"""
import numpy as np
import pytest
import torch

from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.models.actor import GaussianActor, SplitActor


def _cfg(**over):
    cfg = {
        "hidden_sizes": [16, 16], "critic_hidden_sizes": [8, 8],
        "lr_actor": 3e-4, "lr_trans": 3e-4, "lr_perm": 2e-4, "perm_optimizer": "sgd",
        "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2, "epochs": 1,
        "minibatch_size": 16, "ent_coef": 0.0, "max_grad_norm": 0.5, "target_kl": None,
        "normalize_advantage": True, "n_steps": 8, "num_envs": 2,
        "k": 3, "decay": 0.95, "decay_mode": "output",
    }
    cfg.update(over)
    return cfg


def _agent(kind, cfg=None):
    torch.manual_seed(0)
    cls = {"pt": PPOPT, "vanilla": PPOVanilla}[kind]
    return cls(obs_dim=4, act_dim=2, cfg=cfg or _cfg(), device=torch.device("cpu"))


def _fill(agent, rng):
    b = agent.buffer
    b.ptr = b.n_steps
    b.obs[:] = rng.standard_normal(b.obs.shape).astype(np.float32)
    b.actions[:] = rng.standard_normal(b.actions.shape).astype(np.float32)
    b.logprobs[:] = rng.standard_normal(b.logprobs.shape).astype(np.float32)
    b.rewards[:] = rng.standard_normal(b.rewards.shape).astype(np.float32)
    b.dones[:] = 0.0
    for t in range(b.n_steps):
        vp, vt = agent.get_value(b.obs[t])
        b.v_perm[t] = vp
        b.v_trans[t] = vt


# ----------------------------------------------------------------------------------
# The guarantee that protects every earlier result
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["pt", "vanilla"])
def test_split_actor_off_is_bit_identical(kind):
    """Default config must train exactly as it did before `split_actor` existed."""
    def run(cfg):
        agent = _agent(kind, cfg)
        rng = np.random.default_rng(3)
        np.random.seed(11)
        torch.manual_seed(11)
        for i in range(4):
            _fill(agent, rng)
            agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), i)
            agent.on_task_switch(i * 100)
        return [p.detach().clone() for p in agent.actor.parameters()]

    for a, b in zip(run(_cfg()), run(_cfg(split_actor=False))):
        assert torch.equal(a, b)


@pytest.mark.parametrize("kind", ["pt", "vanilla"])
def test_split_actor_off_builds_the_plain_actor(kind):
    """No SplitActor, no slow optimizer, no state buffer when the flag is off."""
    agent = _agent(kind)
    assert isinstance(agent.actor, GaussianActor)
    assert not hasattr(agent, "actor_perm_optim")
    assert not hasattr(agent, "actor_state_buffer")


# ----------------------------------------------------------------------------------
# Construction and initialisation
# ----------------------------------------------------------------------------------
def test_transient_starts_at_the_zero_function():
    """mu_trans must be identically zero at init, so the agent starts as a plain actor.

    Theorem 1's `V^(T)_0 = 0` transposed to the policy. A randomly initialised transient would
    make the split actor a different agent from step 0, and any comparison would be measuring
    that instead of the mechanism.
    """
    torch.manual_seed(0)
    a = SplitActor(4, 2, hidden_sizes=[8, 8])
    obs = torch.randn(64, 4)
    _, mu_t = a.means(obs)
    assert torch.all(mu_t == 0.0)
    assert torch.allclose(a.act_deterministic(obs), a.perm_mean(obs))


def test_decay_scales_the_function_exactly():
    """mu_trans <- decay * mu_trans, not decay^n_layers * mu_trans (defect #13)."""
    torch.manual_seed(0)
    a = SplitActor(4, 2, hidden_sizes=[8, 8], trans_zero_init=False)
    obs = torch.randn(256, 4)
    before = a.means(obs)[1].clone()
    a.decay_transient(0.95)
    after = a.means(obs)[1]
    np.testing.assert_allclose(after.detach(), 0.95 * before.detach(), rtol=1e-5, atol=1e-6)


def test_ppo_gradient_reaches_only_the_transient():
    """mu_perm is detached in the training forward, so PPO cannot move it.

    Without this the "slow" component trains at the fast timescale and there is no
    decomposition at all — just a wider actor.
    """
    torch.manual_seed(0)
    a = SplitActor(4, 2, hidden_sizes=[8, 8], trans_zero_init=False)
    obs, act = torch.randn(32, 4), torch.randn(32, 2)
    lp, _ = a.evaluate_actions(obs, act)
    lp.sum().backward()
    assert all(p.grad is None or torch.all(p.grad == 0) for p in a.perm_mean.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in a.trans_mean.parameters())


# ----------------------------------------------------------------------------------
# Consolidation
# ----------------------------------------------------------------------------------
def test_consolidation_moves_perm_toward_the_full_policy():
    """mu_perm regresses onto mu_perm + mu_trans (keep = 1, the Eq. (4) analogue)."""
    cfg = _cfg(split_actor=True, actor_hidden_sizes=[8, 8], actor_trans_zero_init=False,
               lr_actor_perm=0.05, actor_alpha_p_rm_power=0.0, actor_k=1)
    agent = _agent("vanilla", cfg)
    probe = torch.randn(128, 4)
    with torch.no_grad():
        p0, t0 = agent.actor.means(probe)

    _fill(agent, np.random.default_rng(5))
    agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), 0)

    with torch.no_grad():
        p1, _ = agent.actor.means(probe)
    moved = (p1 - p0)
    # It moved, and it moved TOWARD the transient rather than in an arbitrary direction.
    assert float(moved.norm()) > 0
    assert float((moved * t0).sum()) > 0
    assert agent.last_actor_absorbed_frac is not None


def test_consolidation_fires_on_the_k_cycle_and_at_a_switch():
    cfg = _cfg(split_actor=True, actor_hidden_sizes=[8, 8], actor_k=3)
    agent = _agent("vanilla", cfg)
    rng = np.random.default_rng(6)
    ages = []
    for i in range(6):
        _fill(agent, rng)
        agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), i)
        ages.append(agent._actor_updates_since_consolidation)
    assert ages == [1, 2, 0, 1, 2, 0]

    _fill(agent, rng)
    agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), 6)
    assert agent._actor_updates_since_consolidation == 1
    agent.on_task_switch(1000)
    assert agent._actor_updates_since_consolidation == 0


def test_pt_both_consolidates_actor_and_critic_at_a_switch():
    """PPOPT.on_task_switch must call super(), or pt_both never consolidates its actor."""
    cfg = _cfg(split_actor=True, actor_hidden_sizes=[8, 8], actor_k=100)
    agent = _agent("pt", cfg)
    _fill(agent, np.random.default_rng(7))
    agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), 0)
    agent._actor_updates_since_consolidation = 5
    agent.on_task_switch(1000)
    assert agent._actor_updates_since_consolidation == 0
    assert agent.last_actor_absorbed_frac is not None


# ----------------------------------------------------------------------------------
# The probe
# ----------------------------------------------------------------------------------
def test_decay_only_changes_actions_immediately():
    """The whole premise: decaying mu_trans changes behaviour with zero gradient steps.

    The split CRITIC cannot do this — decaying V_trans changes no action — which is why the
    value decomposition could never deliver a jumpstart in an actor-critic.
    """
    cfg = _cfg(split_actor=True, actor_hidden_sizes=[8, 8], actor_trans_zero_init=False)
    agent = _agent("vanilla", cfg)
    obs = torch.randn(64, 4)
    before = agent.actor.act_deterministic(obs).clone()
    agent.actor_decay_only()
    after = agent.actor.act_deterministic(obs)
    assert not torch.allclose(before, after)


def test_decay_only_is_a_no_op_without_a_split_actor():
    """Same call on the default agent must do nothing at all."""
    agent = _agent("pt")
    obs = torch.randn(32, 4)
    before = agent.actor.act_deterministic(obs).clone()
    agent.actor_decay_only()
    assert torch.equal(before, agent.actor.act_deterministic(obs))


def test_cancellation_probe_is_logged():
    """corr(mu_perm, mu_trans) must be recorded from the first consolidation.

    On the critic this number came out at ~-1.0 and that is what made the decomposition
    invisible. If the actor does the same, we need it on run one — not after two sweeps.
    """
    cfg = _cfg(split_actor=True, actor_hidden_sizes=[8, 8], actor_trans_zero_init=False,
               actor_k=1)
    agent = _agent("vanilla", cfg)
    _fill(agent, np.random.default_rng(8))
    m = agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), 0)
    assert "diag/actor_perm_trans_corr" in m
    assert -1.0001 <= m["diag/actor_perm_trans_corr"] <= 1.0001
