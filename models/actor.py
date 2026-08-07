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
    """Dual-timescale Gaussian actor: action_mean = mu_perm + mu_trans.

    THE POINT OF THIS CLASS. On the critic, PT provably does nothing here: the permanent and
    transient contributions to the advantage come out anti-correlated at ~-1.0 and cancel before
    anything reaches the policy (TRANSMISSION_RESULTS.md §4). That cancellation is structural —
    V_trans is fit to R - V_perm, so V_trans = R - V_perm identically and the sum is the same
    however the split is made. A value decomposition is therefore invisible at the point of use.

    A policy decomposition is not. `mu_perm + mu_trans` IS the action, so decaying the transient
    changes behaviour immediately, with no gradient steps — which is exactly how PT works in DQN
    (argmax over Q_perm + Q_trans) and exactly what an actor-critic's value split cannot do.
    Whether that survives contact with PPO is the experiment.

    Every design choice below mirrors SplitCritic deliberately, so any difference between the two
    is the actor/critic distinction and not an incidental change:

      - `mu_trans` output layer is ZEROED at init, so the agent starts as a plain GaussianActor.
        (Theorem 1's `V^(T)_0 = 0`, transposed to the policy.)
      - `mu_perm` is DETACHED in the training forward, so PPO's gradient reaches only the
        transient. The permanent moves during consolidation and nowhere else, exactly as theta_P
        does. Without this the "slow" component is trained at the fast timescale and there is no
        decomposition, only a wider network.
      - decay scales the OUTPUT LAYER ONLY. Scaling every parameter of an N-layer MLP by `decay`
        shrinks its output by roughly decay^N (defect #13 on the critic side: 0.75 left ~0.42 of
        V_trans), which makes lambda uncontrollable and unrelated to what Alg. 2 line 9 asks for.
      - `log_std` is NOT split. The paper decomposes a value function, not an exploration
        schedule, and splitting the noise would confound any result with an exploration change.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), log_std_init=0.0,
                 trans_zero_init=True):
        super().__init__()
        self.perm_mean = mlp(obs_dim, list(hidden_sizes), act_dim, out_gain=0.01)
        self.trans_mean = mlp(obs_dim, list(hidden_sizes), act_dim, out_gain=0.01)
        if trans_zero_init:
            # Zero the final affine layer so mu_trans is the zero FUNCTION, not merely small.
            last = [m for m in self.trans_mean.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def means(self, obs):
        """(mu_perm, mu_trans) separately — for consolidation and for the cancellation probe."""
        return self.perm_mean(obs), self.trans_mean(obs)

    def _distribution(self, obs):
        # perm detached: PPO trains the transient only (see class docstring).
        mean = self.perm_mean(obs).detach() + self.trans_mean(obs)
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
        """mu_trans <- decay * mu_trans, exactly, by scaling only the final affine layer."""
        last = [m for m in self.trans_mean.modules() if isinstance(m, nn.Linear)][-1]
        last.weight.data.mul_(decay)
        last.bias.data.mul_(decay)
