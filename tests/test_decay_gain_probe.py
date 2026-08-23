"""probe/decay_gain must measure the decay without changing anything.

With the Phase 1 shrinkage control dropped, this probe is the only thing that can separate "the
decomposition works" from "the decay works": at a boundary it evaluates, decays mu_T, evaluates
again, with NO gradient step in between. That makes its correctness a precondition of the whole
comparison — and its defining property is that it is invisible to the run it measures.
"""
import numpy as np
import pytest
import torch

from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.train import (_actor_perm_trans_corr, _decay_gain_probe,
                                          _run_deterministic_eval)

OBS_DIM, ACT_DIM = 5, 2


class CountingEnv:
    """Deterministic toy env whose return depends only on the actions the policy takes."""

    def __init__(self, episode_len=4):
        self.episode_len = episode_len
        self.resets = 0
        self._t = 0

    def reset(self, **kwargs):
        self.resets += 1
        self._t = 0
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        reward = float(np.sum(np.abs(action)))       # bigger |mu| -> bigger return
        return (np.zeros(OBS_DIM, dtype=np.float32), reward,
                False, self._t >= self.episode_len, {"directional_reward": reward})


def _cfg(**kw):
    c = dict(hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8], lr_actor=3e-4, num_envs=1,
             n_steps=4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, epochs=1, minibatch_size=4,
             actor_trans_hidden_sizes=[8, 8], critic_trans_hidden_sizes=[8, 8],
             ent_coef=0.0, max_grad_norm=0.5, normalize_advantage=False, lr_trans=3e-4,
             lr_perm=1e-3, rho=0.5, k=10, kl_prior_coef=0.01, consolidation_epochs=1,
             consolidation_buffer_size=32)
    c.update(kw)
    return c


def _agent(**kw):
    torch.manual_seed(0)
    return PPOPT(OBS_DIM, ACT_DIM, _cfg(**kw), torch.device("cpu"))


def _give_transient_a_magnitude(agent, scale=1.0):
    with torch.no_grad():
        agent.actor.trans_mean[-1].bias.fill_(scale)


def test_probe_restores_mu_t_exactly():
    """The measurement must leave the policy bit-identical — it runs mid-training."""
    agent = _agent()
    _give_transient_a_magnitude(agent)
    before = {n: p.detach().clone() for n, p in agent.actor.named_parameters()}

    rec = _decay_gain_probe(agent, CountingEnv(), n_episodes=2, max_steps=4)

    assert rec is not None
    for name, param in agent.actor.named_parameters():
        assert torch.equal(param.detach(), before[name]), f"the probe moved {name}"


def test_probe_does_not_consume_the_training_rng():
    """A probe that shifted the RNG would change every subsequent training action."""
    agent = _agent()
    _give_transient_a_magnitude(agent)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()

    _decay_gain_probe(agent, CountingEnv(), n_episodes=2, max_steps=4)

    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


def test_probe_measures_the_decay_and_nothing_else():
    """Return must change by exactly what scaling mu_T by (1-rho) does to the actions.

    The toy env pays |action|, and with a deterministic policy mu = mu_P + mu_T, so decaying mu_T
    is the only thing that can move the number.
    """
    agent = _agent(rho=0.5)
    with torch.no_grad():                    # mu_P = 0, mu_T = 1 -> mu = 1 per dimension
        agent.actor.perm_mean[-1].weight.zero_()
        agent.actor.perm_mean[-1].bias.zero_()
        agent.actor.trans_mean[-1].weight.zero_()
        agent.actor.trans_mean[-1].bias.fill_(1.0)

    env = CountingEnv(episode_len=4)
    rec = _decay_gain_probe(agent, env, n_episodes=1, max_steps=4)
    # 4 steps x 2 action dims x |mu|: 8.0 before, 4.0 after halving mu_T.
    assert rec["before"] == pytest.approx(8.0)
    assert rec["after"] == pytest.approx(4.0)
    assert rec["gain"] == pytest.approx(-4.0)


def test_probe_is_none_for_agents_without_a_split_actor():
    """Vanilla and EWC have nothing to decay; the probe must decline, not invent a number."""
    van = PPOVanilla(OBS_DIM, ACT_DIM, _cfg(ent_coef=0.0), torch.device("cpu"))
    assert _decay_gain_probe(van, CountingEnv(), n_episodes=1, max_steps=2) is None


def test_deterministic_eval_uses_the_policy_mean():
    """No exploration noise: two evaluations of an unchanged policy must agree exactly."""
    agent = _agent()
    _give_transient_a_magnitude(agent)
    env = CountingEnv()
    first = _run_deterministic_eval(agent, env, n_episodes=2, max_steps=4)
    second = _run_deterministic_eval(agent, env, n_episodes=2, max_steps=4)
    assert first == pytest.approx(second)


def test_perm_trans_correlation_detects_cancelling_components():
    """Near -1 means the components cancel and the ablation figure will read flat."""
    agent = _agent()
    states = np.random.randn(64, OBS_DIM).astype(np.float32)

    # Undefined while mu_T is still exactly the zero function (its init).
    assert _actor_perm_trans_corr(agent, states) is None

    # Make mu_T the exact negative of mu_P: the composed policy is 0 and corr is -1.
    with torch.no_grad():
        agent.actor.trans_mean.load_state_dict(agent.actor.perm_mean.state_dict())
        agent.actor.trans_mean[-1].weight.mul_(-1.0)
        agent.actor.trans_mean[-1].bias.mul_(-1.0)
    corr = _actor_perm_trans_corr(agent, states)
    assert corr is not None and corr == pytest.approx(-1.0, abs=1e-4)
