"""Critics: a single-trunk VanillaCritic and a dual-timescale SplitCritic.

SplitCritic uses TWO SEPARATE trunks (V_perm and V_trans) so each can carry its own optimizer and
learning rate and be decayed independently — mirroring the separate P_Net / T_Net in the paper's
control/minatar_crl/PT_DQN_half.py (rather than the shared-trunk two-heads used for minigrid).

The acting/bootstrapping value is always V = V_perm + V_trans.
"""
import torch
import torch.nn as nn

from .actor import mlp


class VanillaCritic(nn.Module):
    """Single undivided state-value V(s) (the baseline)."""

    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, list(hidden_sizes), 1, out_gain=1.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class SplitCritic(nn.Module):
    """Dual-timescale state-value: V(s) = V_perm(s; theta_P) + V_trans(s; theta_T)."""

    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.perm = mlp(obs_dim, list(hidden_sizes), 1, out_gain=1.0)
        self.trans = mlp(obs_dim, list(hidden_sizes), 1, out_gain=1.0)

    def forward(self, obs):
        """Returns (v_perm, v_trans), each shape (batch,)."""
        return self.perm(obs).squeeze(-1), self.trans(obs).squeeze(-1)

    def value(self, obs):
        v_perm, v_trans = self.forward(obs)
        return v_perm + v_trans

    @torch.no_grad()
    def decay_transient(self, decay):
        """theta_T <- decay * theta_T  (decay=0 ~= reset). Mirrors `params.data *= args.decay`."""
        for p in self.trans.parameters():
            p.data.mul_(decay)
