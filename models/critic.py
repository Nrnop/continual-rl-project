"""Critics: VanillaCritic (one network) and SplitCritic (two separate networks).

The value used for acting and bootstrapping is V = V_perm + V_trans. `SplitCritic` mirrors
`P_Net` / `T_Net` in the reference code (control/minatar_crl/PT_DQN_half.py): two independent
networks whose outputs are added, with no shared trunk, so the two components keep separate
timescales. Because the trunks are separate, consolidation is a regression of the permanent onto
`old_V_perm + V_trans` over visited states (see `agents/ppo_pt.py::_consolidate`).
"""
import torch
import torch.nn as nn

from .actor import mlp


class VanillaCritic(nn.Module):
    """Single undivided state-value V(s) (the baseline)."""

    def __init__(self, obs_dim, hidden_sizes=(256, 256), zero_init=False, layer_norm=False):
        super().__init__()
        self.net = mlp(obs_dim, list(hidden_sizes), 1, out_gain=1.0, layer_norm=layer_norm)
        # zero_init: V(s) = 0 at t = 0, matching a PT agent with perm_zero_init. This meets
        # Theorem 1's second condition, V^(TD)_0 = V^(P): the TD baseline starts from the same
        # function as the permanent.
        if zero_init:
            nn.init.constant_(self.net[-1].weight, 0.0)
            nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class SplitCritic(nn.Module):
    """Split state value: V(s) = V_perm(s; theta_P) + V_trans(s; theta_T).

    Two fully separate MLPs, as P_Net / T_Net in the reference. Initialisation follows Theorem 1
    ("If V^(TD)_0 = V^(P), V^(T)_0 = 0 [...] then for all t, V^(PT)_t = V^(TD)_t"): the permanent
    gets the ordinary value-head init and the transient starts at the zero function (output layer
    zeroed), so at t = 0 the acting value equals the permanent alone.
    """

    def __init__(self, obs_dim, hidden_sizes=(256, 256), perm_zero_init=False,
                 trans_zero_init=True, perm_init_std=None, trans_hidden_sizes=None,
                 perm_obs_dim=None, layer_norm=False):
        super().__init__()
        # trans_hidden_sizes=None reuses hidden_sizes.
        if trans_hidden_sizes is None:
            trans_hidden_sizes = hidden_sizes
        # perm_obs_dim < obs_dim hides the observation's tail from the permanent, as in
        # SplitGaussianActor. Split the critic the same way as the actor, or it produces advantages
        # from information the policy cannot see.
        self.perm_obs_dim = int(perm_obs_dim) if perm_obs_dim is not None else int(obs_dim)
        if not 0 < self.perm_obs_dim <= obs_dim:
            raise ValueError(f"perm_obs_dim {self.perm_obs_dim} outside (0, {obs_dim}]")
        self.perm = mlp(self.perm_obs_dim, list(hidden_sizes), 1, out_gain=1.0,
                        layer_norm=layer_norm)
        self.trans = mlp(obs_dim, list(trans_hidden_sizes), 1, out_gain=1.0,
                         layer_norm=layer_norm)

        # trans_zero_init: V^(T)_0 = 0 (Theorem 1). The paper leaves deep initialisation open, and
        # the reference code initialises both nets randomly; set False to reproduce that.
        if trans_zero_init:
            nn.init.constant_(self.trans[-1].weight, 0.0)
            nn.init.constant_(self.trans[-1].bias, 0.0)

        # perm_zero_init: V^(P)_0 = 0 as well. With a small alpha_P the permanent barely moves, so
        # a random init stays in V_perm for the whole run and the transient must learn to cancel
        # it, which does not generalise to newly visited states. With V_perm = 0 the critic loss
        # (V_perm.detach() + V_trans - R)^2 reduces exactly to vanilla's (V - R)^2.
        if perm_zero_init:
            nn.init.constant_(self.perm[-1].weight, 0.0)
            nn.init.constant_(self.perm[-1].bias, 0.0)
        elif perm_init_std is not None:
            # Reproduces the deep reference init, small but non-zero
            # (prediction_semi_crl/minigrid/model.py):
            #     def weight_init(m):
            #         if isinstance(m, nn.Linear):
            #             nn.init.normal_(m.weight, 0, 0.01); m.bias.data.fill_(0.0)
            # Only the reference's tabular and linear versions start from zeros.
            for m in self.perm:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, perm_init_std)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        """Return (v_perm, v_trans), each of shape (batch,)."""
        return self.perm_forward(obs).squeeze(-1), self.trans(obs).squeeze(-1)

    def perm_forward(self, obs):
        """Return V_P(s) on the first `perm_obs_dim` obs entries. Use instead of `self.perm`."""
        return self.perm(obs[..., :self.perm_obs_dim])

    def value(self, obs):
        v_perm, v_trans = self.forward(obs)
        return v_perm + v_trans

    @torch.no_grad()
    def decay_transient(self, decay, mode="params"):
        """Decay the transient value function (Alg. 2 line 9).

        mode="params": multiply every transient parameter by `decay`, as the reference does
            (`for params in T_Net.parameters(): params.data *= args.decay`). For a nonlinear net
            this does not scale the output by `decay` (on [43,43], decay=0.75 leaves about 0.42 of
            V_trans). Only decay=0 is exact.
        mode="output": scale only the final affine layer, W*h + b -> decay*(W*h + b), so
            V_trans <- decay * V_trans exactly and the hidden features are left intact.
        """
        if mode == "output":
            self.trans[-1].weight.data.mul_(decay)
            self.trans[-1].bias.data.mul_(decay)
        else:
            for p in self.trans.parameters():
                p.data.mul_(decay)
