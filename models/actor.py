"""Gaussian MLP policies for continuous actions (PPO-style).

The module provides a monolithic Gaussian policy and a permanent-transient policy with separate
mean networks and shared state-independent log_std.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim, hidden_sizes, out_dim, activation=nn.Tanh, out_gain=0.01):
    """Build an MLP with orthogonal init (PPO convention: small gain on the output layer)."""
    layers = []
    last = in_dim
    for h in hidden_sizes:
        linear = nn.Linear(last, h)
        nn.init.orthogonal_(linear.weight, np.sqrt(2))
        nn.init.constant_(linear.bias, 0.0)
        layers += [linear, activation()]
        last = h
    out = nn.Linear(last, out_dim)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.constant_(out.bias, 0.0)
    layers.append(out)
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), log_std_init=0.0,
                 freeze_log_std=False):
        super().__init__()
        self.mean_net = mlp(obs_dim, list(hidden_sizes), act_dim, out_gain=0.01)
        # `freeze_log_std` exists ONLY to make a fair comparison against pt_full possible.
        # SplitGaussianActor owns log_std in its PERMANENT set and freezes it (Constraint C4),
        # while this actor learns it — so by default the two arms run different exploration
        # schedules and any return difference confounds the PT mechanism with sigma. Measured on
        # DirectionalPointMass: vanilla anneals to sigma ~0.58-0.76 while pt_full stays at 1.0.
        # Default False keeps every earlier run bit-identical.
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init,
                                    requires_grad=not freeze_log_std)

    def _distribution(self, obs):
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, obs):
        """Sample an action. Returns (action, logprob) as tensors."""
        dist = self._distribution(obs)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        return action, logprob

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.mean_net(obs)

    def evaluate_actions(self, obs, actions):
        """For the PPO update: returns (logprobs, entropy) of given actions under current policy."""
        dist = self._distribution(obs)
        logprobs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprobs, entropy

    def kl_to_zero_prior(self, obs):
        """KL(pi || N(0, sigma^2 I)) = sum_i mu(s)_i^2 / (2 sigma_i^2).

        The single-network analogue of SplitGaussianActor.kl_to_prior, with the permanent
        replaced by the ZERO policy. It exists to test the Stage-1/2 conclusion directly: the
        measured benefit of pt_full came from its INERT arm, whose permanent is the zero function
        (RMS |mu_P| = 0.002), so its KL anchor is exactly this penalty. If vanilla plus this term
        plus a frozen sigma reproduces the inert arm, then the whole PT apparatus reduces to one
        regularizer and should be reported as such.

        Deliberately identical in form to kl_to_prior so the two arms differ only in the prior.
        """
        mu = self.mean_net(obs)
        var = torch.exp(self.log_std) ** 2
        return (mu ** 2 / (2 * var)).sum(-1)



class SplitGaussianActor(nn.Module):
    """Dual-timescale Gaussian policy: pi_PT = Normal(perm_mean(s) + trans_mean(s), exp(log_std)).

    Two fully separate MLPs (no shared parameters/modules — same isolation as `SplitCritic`).
    `trans_mean`'s output layer is zeroed after construction so mu_T(s) == 0 at init and pi_PT is
    exactly pi_P (KL-to-prior == 0); interior layers keep the normal orthogonal init so gradients
    still flow. `log_std` is owned by the PERMANENT set (Constraint C4): `requires_grad=False`
    freezes it for the life of the run (the "held on a global schedule" option, with the trivial
    constant schedule), so no online transient step or autograd path can ever move it.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), trans_hidden_sizes=(64, 64),
                 log_std_init=0.0):
        super().__init__()
        self.perm_mean = mlp(obs_dim, list(hidden_sizes), act_dim, out_gain=0.01)
        self.trans_mean = mlp(obs_dim, list(trans_hidden_sizes), act_dim, out_gain=0.01)
        nn.init.constant_(self.trans_mean[-1].weight, 0.0)
        nn.init.constant_(self.trans_mean[-1].bias, 0.0)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init, requires_grad=False)

    def _composed_mean(self, obs, detach_perm=False):
        perm = self.perm_mean(obs)
        if detach_perm:
            perm = perm.detach()
        return perm + self.trans_mean(obs)

    def _distribution(self, obs, detach_perm=False):
        mean = self._composed_mean(obs, detach_perm=detach_perm)
        return Normal(mean, torch.exp(self.log_std))

    @torch.no_grad()
    def act(self, obs):
        """Sample an action. Returns (action, logprob) as tensors."""
        dist = self._distribution(obs)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        return action, logprob

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self._composed_mean(obs)

    def evaluate_actions(self, obs, actions, detach_perm=True):
        """For the PPO update: returns (logprobs, entropy) of given actions under current policy.

        detach_perm=True (default, the online PPO path): perm_mean's output is detached before
        summation, so gradients reach only trans_mean and log_std. detach_perm=False is for the
        later consolidation phase, where perm_mean is meant to receive gradient too.
        """
        dist = self._distribution(obs, detach_perm=detach_perm)
        logprobs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprobs, entropy

    def kl_to_prior(self, obs):
        """KL(pi_PT || pi_P) in closed form for a shared, fixed covariance:

            sum_i mu_T(s)_i^2 / (2 sigma_i^2)

        log_std already has requires_grad=False (frozen, C4); mu_T is NOT detached, since this
        term is what regularizes the transient toward the permanent. perm_mean does not appear
        here at all.
        """
        mu_trans = self.trans_mean(obs)
        var = torch.exp(self.log_std) ** 2
        return (mu_trans ** 2 / (2 * var)).sum(-1)

    @torch.no_grad()
    def decay_transient(self, factor):
        """Scale mu_T(s) by `factor` exactly, for any obs, by scaling only its (affine) output
        layer. Never touches log_std or perm_mean."""
        self.trans_mean[-1].weight.data.mul_(factor)
        self.trans_mean[-1].bias.data.mul_(factor)
