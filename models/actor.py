"""Gaussian MLP policies for continuous actions.

A single-network Gaussian policy and a permanent-transient policy with two separate mean networks
and one shared, state-independent log_std.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim, hidden_sizes, out_dim, activation=nn.Tanh, out_gain=0.01, layer_norm=False):
    """Build an MLP with orthogonal init and a small gain on the output layer (PPO convention).

    Every actor and critic is built here, so all arms share one architecture. `layer_norm` (off by
    default) adds nn.LayerNorm before the activation of each hidden layer, a standard mitigation
    for loss of plasticity under a changing task. It is never applied to the output layer: the
    transient decay (Alg. 2 line 9) scales the final Linear, W*h + b -> decay*(W*h + b), which is
    exact only while the output layer is affine. The zero-init of the transient output layer
    (Theorem 1) is unaffected by normalisation upstream of it.
    """
    layers = []
    last = in_dim
    for h in hidden_sizes:
        linear = nn.Linear(last, h)
        nn.init.orthogonal_(linear.weight, np.sqrt(2))
        nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        last = h
    out = nn.Linear(last, out_dim)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.constant_(out.bias, 0.0)
    layers.append(out)
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), log_std_init=0.0,
                 freeze_log_std=False, layer_norm=False):
        super().__init__()
        self.mean_net = mlp(obs_dim, list(hidden_sizes), act_dim, out_gain=0.01,
                            layer_norm=layer_norm)
        # freeze_log_std lets this actor hold sigma fixed like SplitGaussianActor does, so a
        # return difference between arms is not confounded with a different exploration schedule.
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
        """Return (logprobs, entropy) of `actions` under the current policy."""
        dist = self._distribution(obs)
        logprobs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprobs, entropy

    def kl_to_zero_prior(self, obs):
        """KL(pi || N(0, sigma^2 I)) = sum_i mu(s)_i^2 / (2 sigma_i^2).

        Same form as SplitGaussianActor.kl_to_prior with the permanent replaced by the zero
        policy. Used as a control: a single network with the KL anchor of a PT agent whose
        permanent is inert.
        """
        mu = self.mean_net(obs)
        var = torch.exp(self.log_std) ** 2
        return (mu ** 2 / (2 * var)).sum(-1)



class SplitGaussianActor(nn.Module):
    """Split Gaussian policy: pi_PT = Normal(perm_mean(s) + trans_mean(s), exp(log_std)).

    Two fully separate MLPs with no shared modules, as in `SplitCritic`. The output layer of
    `trans_mean` starts at zero, so mu_T(s) = 0 and pi_PT = pi_P at init (Theorem 1); its hidden
    layers keep the orthogonal init so gradients still flow. `log_std` is not split. It belongs to
    the permanent and is frozen by default, so no transient step can move it.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), trans_hidden_sizes=(64, 64),
                 log_std_init=0.0, freeze_log_std=True, perm_obs_dim=None, layer_norm=False):
        super().__init__()
        # perm_obs_dim < obs_dim feeds the tail of the observation (a task identifier) to the
        # transient only, as in Anand & Precup's control experiment. None feeds both nets the
        # whole observation.
        self.perm_obs_dim = int(perm_obs_dim) if perm_obs_dim is not None else int(obs_dim)
        if not 0 < self.perm_obs_dim <= obs_dim:
            raise ValueError(f"perm_obs_dim {self.perm_obs_dim} outside (0, {obs_dim}]")
        self.perm_mean = mlp(self.perm_obs_dim, list(hidden_sizes), act_dim, out_gain=0.01,
                             layer_norm=layer_norm)
        self.trans_mean = mlp(obs_dim, list(trans_hidden_sizes), act_dim, out_gain=0.01,
                              layer_norm=layer_norm)
        nn.init.constant_(self.trans_mean[-1].weight, 0.0)
        nn.init.constant_(self.trans_mean[-1].bias, 0.0)
        # log_std is held constant by default. The same config key governs every arm, so vanilla
        # and ewc are frozen too and sigma does not differ between arms. Setting it False
        # everywhere gives standard PPO's learned sigma.
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init,
                                    requires_grad=not freeze_log_std)

    def perm_forward(self, obs):
        """Return mu_P(s), with the observation cut to its first `perm_obs_dim` entries.

        Always call this instead of `self.perm_mean(obs)`, or the permanent may see the task
        identifier it is meant not to see.
        """
        return self.perm_mean(obs[..., :self.perm_obs_dim])

    def _composed_mean(self, obs, detach_perm=False):
        perm = self.perm_forward(obs)
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
        """Return (logprobs, entropy) of `actions` under the current policy.

        detach_perm=True (the PPO update) detaches mu_P, so the gradient reaches only the
        transient and log_std. detach_perm=False lets the gradient reach perm_mean as well.
        """
        dist = self._distribution(obs, detach_perm=detach_perm)
        logprobs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprobs, entropy

    def kl_to_prior(self, obs):
        """KL(pi_PT || pi_P) = sum_i mu_T(s)_i^2 / (2 sigma_i^2), for a shared fixed covariance.

        mu_T is not detached: this term pulls the transient toward the permanent.
        """
        mu_trans = self.trans_mean(obs)
        var = torch.exp(self.log_std) ** 2
        return (mu_trans ** 2 / (2 * var)).sum(-1)

    @torch.no_grad()
    def decay_transient(self, factor):
        """Scale mu_T(s) by `factor` exactly by scaling its affine output layer only."""
        self.trans_mean[-1].weight.data.mul_(factor)
        self.trans_mean[-1].bias.data.mul_(factor)
