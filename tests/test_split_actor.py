"""Tests for SplitGaussianActor: zero init, output-layer decay, and gradient isolation.

Also covers the KL-to-prior term and the SplitCritic width options.
"""
import torch
from torch.distributions import Normal

from src_continuous_control.models.actor import SplitGaussianActor
from src_continuous_control.models.critic import SplitCritic

OBS_DIM = 8
ACT_DIM = 3


def test_zero_init_exactness():
    """The transient mean is zero at init, so the composed mean equals mu_P and KL is 0."""
    torch.manual_seed(0)
    actor = SplitGaussianActor(OBS_DIM, ACT_DIM)
    obs = torch.randn(16, OBS_DIM)

    assert torch.all(actor.trans_mean(obs) == 0)
    assert torch.equal(actor.act_deterministic(obs), actor.perm_mean(obs))
    assert torch.all(actor.kl_to_prior(obs) == 0)


def test_composed_distribution_matches_perm_at_init():
    """At init, log-probs of the composed policy equal those of N(mu_P, sigma)."""
    torch.manual_seed(1)
    actor = SplitGaussianActor(OBS_DIM, ACT_DIM)
    obs = torch.randn(16, OBS_DIM)
    actions = torch.randn(16, ACT_DIM)

    logprobs, _ = actor.evaluate_actions(obs, actions)
    expected = Normal(actor.perm_mean(obs), torch.exp(actor.log_std)).log_prob(actions).sum(-1)
    assert torch.allclose(logprobs, expected, atol=1e-6)


def test_decay_transient_output_exactness():
    """decay_transient(f) scales mu_T by f and leaves the hidden layers unchanged."""
    torch.manual_seed(2)
    actor = SplitGaussianActor(OBS_DIM, ACT_DIM)
    out_layer = actor.trans_mean[-1]
    torch.nn.init.normal_(out_layer.weight, 0.0, 1.0)
    torch.nn.init.normal_(out_layer.bias, 0.0, 1.0)

    obs = torch.randn(16, OBS_DIM)
    interior_before = [p.clone() for p in actor.trans_mean[:-1].parameters()]
    mu_t_before = actor.trans_mean(obs)

    actor.decay_transient(0.3)

    mu_t_after = actor.trans_mean(obs)
    assert torch.allclose(mu_t_after, 0.3 * mu_t_before, atol=1e-6, rtol=1e-5)
    for before, after in zip(interior_before, actor.trans_mean[:-1].parameters()):
        assert torch.equal(before, after)


def test_log_std_untouched_by_decay():
    """Decay changes neither log_std nor the permanent network."""
    torch.manual_seed(3)
    actor = SplitGaussianActor(OBS_DIM, ACT_DIM)
    log_std_before = actor.log_std.data.clone()
    perm_before = [p.clone() for p in actor.perm_mean.parameters()]

    actor.decay_transient(0.5)

    assert torch.equal(actor.log_std.data, log_std_before)
    for before, after in zip(perm_before, actor.perm_mean.parameters()):
        assert torch.equal(before, after)


def test_architecture_isolation():
    """Permanent and transient share no parameters, in the actor and in the critic."""
    torch.manual_seed(4)
    actor = SplitGaussianActor(OBS_DIM, ACT_DIM)
    perm_ids = {id(p) for p in actor.perm_mean.parameters()}
    trans_ids = {id(p) for p in actor.trans_mean.parameters()}
    assert perm_ids.isdisjoint(trans_ids)

    critic = SplitCritic(OBS_DIM)
    perm_ids = {id(p) for p in critic.perm.parameters()}
    trans_ids = {id(p) for p in critic.trans.parameters()}
    assert perm_ids.isdisjoint(trans_ids)


def test_kl_detached_sigma():
    """The KL-to-prior gradient reaches only the transient, not log_std or mu_P."""
    torch.manual_seed(5)
    actor = SplitGaussianActor(OBS_DIM, ACT_DIM)
    out_layer = actor.trans_mean[-1]
    torch.nn.init.normal_(out_layer.weight, 0.0, 1.0)
    torch.nn.init.normal_(out_layer.bias, 0.0, 1.0)

    obs = torch.randn(16, OBS_DIM)
    actor.kl_to_prior(obs).mean().backward()

    assert actor.log_std.grad is None or torch.all(actor.log_std.grad == 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in actor.trans_mean.parameters())
    assert all(p.grad is None for p in actor.perm_mean.parameters())


def test_evaluate_actions_detach_perm():
    """detach_perm=True blocks the gradient to mu_P; detach_perm=False lets it through."""
    torch.manual_seed(6)
    actor = SplitGaussianActor(OBS_DIM, ACT_DIM)
    obs = torch.randn(16, OBS_DIM)
    actions = torch.randn(16, ACT_DIM)

    logprobs, _ = actor.evaluate_actions(obs, actions, detach_perm=True)
    logprobs.sum().backward()
    assert all(p.grad is None or torch.all(p.grad == 0) for p in actor.perm_mean.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in actor.trans_mean.parameters())
    assert actor.log_std.grad is None  # log_std is frozen (requires_grad=False)

    actor.zero_grad()
    logprobs, _ = actor.evaluate_actions(obs, actions, detach_perm=False)
    logprobs.sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in actor.perm_mean.parameters())


def test_split_critic_trans_hidden_sizes():
    """trans_hidden_sizes sets the transient width and keeps its zero init."""
    torch.manual_seed(7)
    default = SplitCritic(OBS_DIM, hidden_sizes=(32, 32))
    explicit = SplitCritic(OBS_DIM, hidden_sizes=(32, 32), trans_hidden_sizes=(32, 32))
    assert {(k, tuple(v.shape)) for k, v in default.state_dict().items()} == \
        {(k, tuple(v.shape)) for k, v in explicit.state_dict().items()}

    narrow = SplitCritic(OBS_DIM, hidden_sizes=(32, 32), trans_hidden_sizes=(64, 64))
    assert dict(narrow.perm.state_dict())["0.weight"].shape == (32, OBS_DIM)
    assert dict(narrow.trans.state_dict())["0.weight"].shape == (64, OBS_DIM)

    obs = torch.randn(16, OBS_DIM)
    with torch.no_grad():
        _, v_trans = narrow(obs)
    assert torch.all(v_trans == 0)


def test_split_critic_backward_compat_zero_init():
    """The default SplitCritic transient outputs zero at init."""
    torch.manual_seed(8)
    critic = SplitCritic(OBS_DIM)
    obs = torch.randn(16, OBS_DIM)
    with torch.no_grad():
        _, v_trans = critic(obs)
    assert torch.all(v_trans == 0)
