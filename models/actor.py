"""Gaussian MLP policy for continuous actions (PPO-style).

State-dependent mean, state-independent log_std (the standard, stable PPO-on-MuJoCo choice). The
policy is shared by both agents; only the critic differs between vanilla and PT.
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
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), log_std_init=0.0):
        super().__init__()
        self.mean_net = mlp(obs_dim, list(hidden_sizes), act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

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

class SplitActor(nn.Module):
    """Dual-timescale Gaussian Actor: action_mean = perm_mean + trans_mean."""
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), log_std_init=0.0):
        super().__init__()
        self.perm_mean = mlp(obs_dim, list(hidden_sizes), act_dim, out_gain=0.01)
        self.trans_mean = mlp(obs_dim, list(hidden_sizes), act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def _distribution(self, obs):
        mean = self.perm_mean(obs) + self.trans_mean(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, obs):
        dist = self._distribution(obs)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        return action, logprob

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.perm_mean(obs) + self.trans_mean(obs)

    def evaluate_actions(self, obs, actions):
        dist = self._distribution(obs)
        logprobs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprobs, entropy

    @torch.no_grad()
    def decay_transient(self, decay):
        for p in self.trans_mean.parameters():
            p.data.mul_(decay)
