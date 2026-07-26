"""Critics: VanillaCritic, SplitCritic (two trunks), and SharedTrunkSplitCritic (two linear heads).

The acting/bootstrapping value is always V = V_perm + V_trans.

Two PT variants, both present in the reference implementation:

- `SplitCritic` — TWO SEPARATE trunks, mirroring P_Net / T_Net in the paper's
  control/minatar_crl/PT_DQN_half.py. Consolidation must make V_perm *learn* old_V_perm + V_trans by
  regression, i.e. represent the SUM OF TWO MLPs with a single MLP. That function class is not closed
  under addition, so the operation is lossy by construction (measured ~16-35% value error per
  consolidation even with a perfectly converged regression, and ~98% at production settings where the
  slow SGD barely moves). Repeated ~150x per run it destroys the value function.

- `SharedTrunkSplitCritic` — ONE shared feature trunk with two LINEAR heads (the shared-trunk
  two-head variant the reference uses for minigrid). Because
      V = w_P·phi(s) + w_T·phi(s) = (w_P + w_T)·phi(s)
  the sum of the two heads is itself linear in phi, so consolidation is EXACT weight arithmetic
  (w_P += (1-decay)·w_T ; w_T *= decay) with zero regression and zero value drift, for ANY decay.
  Decaying a linear head also scales its output exactly, unlike scaling the parameters of an MLP.
  This is the correct variant under deep function approximation.
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
        """theta_T <- decay * theta_T  (decay=0 ~= reset). Mirrors `params.data *= args.decay`.

        NOTE: for this two-MLP variant, scaling the PARAMETERS by `decay` does NOT scale the OUTPUT
        V_trans by `decay` (nonlinear activations + biases) — only decay=0 is exact. See
        SharedTrunkSplitCritic, where the head is linear and this operation is exact for any decay.
        """
        for p in self.trans.parameters():
            p.data.mul_(decay)


class SharedTrunkSplitCritic(nn.Module):
    """V(s) = w_P·phi(s) + w_T·phi(s), with a shared trunk phi and two LINEAR heads.

    The point of this variant is that consolidation becomes exact. Since both heads are linear in the
    same features, their sum is linear in those features too, so absorbing the transient into the
    permanent is pure weight arithmetic — no regression, no consolidation buffer, no lr_perm:

        w_P <- w_P + (1-decay)·w_T ;   w_T <- decay·w_T
        =>  V_new = (w_P + (1-decay)·w_T + decay·w_T)·phi = (w_P + w_T)·phi = V_old   (EXACT)

    The trunk is shared, so the permanent head's *function* still moves as the features are learned;
    the timescale separation lives in the heads. That is the trade-off of this variant.
    """

    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            lin = nn.Linear(last, h)
            nn.init.orthogonal_(lin.weight, 2 ** 0.5)
            nn.init.constant_(lin.bias, 0.0)
            layers += [lin, nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.perm = nn.Linear(last, 1)
        self.trans = nn.Linear(last, 1)
        # The transient is the active learner -> standard PPO value-head init (orthogonal, gain 1).
        nn.init.orthogonal_(self.trans.weight, 1.0)
        nn.init.constant_(self.trans.bias, 0.0)
        # The permanent starts EMPTY and only ever accumulates via consolidation. Initialising it to a
        # random function would force the transient to spend capacity cancelling an arbitrary offset,
        # so zero it: V = V_perm + V_trans starts as pure transient, exactly the intended semantics.
        nn.init.constant_(self.perm.weight, 0.0)
        nn.init.constant_(self.perm.bias, 0.0)

    def forward(self, obs):
        """Returns (v_perm, v_trans), each shape (batch,)."""
        f = self.trunk(obs)
        return self.perm(f).squeeze(-1), self.trans(f).squeeze(-1)

    def value(self, obs):
        v_perm, v_trans = self.forward(obs)
        return v_perm + v_trans

    @torch.no_grad()
    def consolidate(self, decay):
        """Absorb the transient head into the permanent head and decay it — exactly value-preserving."""
        keep = 1.0 - decay
        self.perm.weight.add_(keep * self.trans.weight)
        self.perm.bias.add_(keep * self.trans.bias)
        self.trans.weight.mul_(decay)
        self.trans.bias.mul_(decay)

    @torch.no_grad()
    def decay_transient(self, decay):
        """w_T <- decay·w_T. Exact output scaling here, because the head is linear."""
        self.trans.weight.mul_(decay)
        self.trans.bias.mul_(decay)
