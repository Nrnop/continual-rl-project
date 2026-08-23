"""Critics: VanillaCritic (single trunk) and SplitCritic (two fully separate trunks).

The acting/bootstrapping value is always V = V_perm + V_trans.

`SplitCritic` mirrors the reference implementation we are porting — `T_Net` / `P_Net` in
control/minatar_crl/PT_DQN_half.py, which are TWO INDEPENDENT networks
(`T_Net = CNN_half(...)`, `P_Net = CNN_half(...)`) whose outputs are added. There is no shared
trunk and no weight sharing of any kind: the whole point of the decomposition is that the two
components live on separate timescales, which a shared trunk would couple.

Consequence, stated plainly because it is the mechanism's cost: with two separate trunks,
consolidation cannot be weight arithmetic. The permanent net has to *learn* `old_V_perm + V_trans`
by regression over a buffer of visited states (see `agents/ppo_pt.py::_consolidate`), and the decay
scales the transient's PARAMETERS, which for a nonlinear net does not scale its OUTPUT by the same
factor. Both are properties of the reference algorithm, not of this port; both are measured in
FINDINGS.md 6.1. Anything that removes them (e.g. sharing a trunk so the transfer becomes exact)
is a different architecture, not this one.
"""
import torch
import torch.nn as nn

from .actor import mlp


class VanillaCritic(nn.Module):
    """Single undivided state-value V(s) (the baseline)."""

    def __init__(self, obs_dim, hidden_sizes=(256, 256), zero_init=False, layer_norm=False):
        super().__init__()
        self.net = mlp(obs_dim, list(hidden_sizes), 1, out_gain=1.0, layer_norm=layer_norm)
        # zero_init: V(s) = 0 at t=0, matching a PT agent run with perm_zero_init.
        #
        # Theorem 1 has TWO conditions: V^(T)_0 = 0 AND V^(TD)_0 = V^(P) — the TD baseline must
        # start from the SAME function as the permanent. We enforced only the first. With
        # perm_zero_init the PT acting value starts at exactly 0 while this critic starts at a
        # random function of magnitude ~0.4, so the two agents have never begun from the same
        # place and the equivalence the theorem asserts has never actually been tested.
        #
        # This is a systematic offset, not a variance source, and it matches the signature of the
        # residual: theorem1_zeroperm vs vanilla was whole-run p=0.008 with NO individual phase
        # significant — a small consistent difference everywhere.
        if zero_init:
            nn.init.constant_(self.net[-1].weight, 0.0)
            nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class SplitCritic(nn.Module):
    """Dual-timescale state-value: V(s) = V_perm(s; theta_P) + V_trans(s; theta_T).

    Two fully separate MLPs, exactly as P_Net / T_Net in the reference.

    INITIALISATION follows Theorem 1 of the paper, which is the condition under which PT is a
    *strict generalisation* of TD learning:

        "If V^(TD)_0 = V^(P), V^(T)_0 = 0 [...] then for all t, V^(PT)_t = V^(TD)_t."

    So the PERMANENT gets the ordinary value-head initialisation (it plays the role of TD's own
    init), and the TRANSIENT starts at the ZERO FUNCTION. Zeroing the transient's output layer makes
    V_trans(s) == 0 exactly for every s, so at t=0 the acting value V = V_perm + V_trans is
    identical to a single vanilla critic's — no extra initialisation noise, and the transient only
    ever represents the residual it has actually learned.

    (Initialising theta_T randomly instead — which this class used to do — makes the acting value
    the sum of two independent random functions, strictly noisier than the baseline it is compared
    against, and breaks the equivalence in Theorem 1.)
    """

    def __init__(self, obs_dim, hidden_sizes=(256, 256), perm_zero_init=False,
                 trans_zero_init=True, perm_init_std=None, trans_hidden_sizes=None,
                 perm_obs_dim=None, layer_norm=False):
        super().__init__()
        # trans_hidden_sizes=None reuses hidden_sizes (backward compatible with every existing caller).
        if trans_hidden_sizes is None:
            trans_hidden_sizes = hidden_sizes
        # perm_obs_dim < obs_dim hides the observation's tail from the permanent — see the same
        # argument on SplitGaussianActor. The critic must be split the same way as the actor: a
        # value function that can see the task while the policy cannot would produce advantages the
        # policy has no way to act on.
        self.perm_obs_dim = int(perm_obs_dim) if perm_obs_dim is not None else int(obs_dim)
        if not 0 < self.perm_obs_dim <= obs_dim:
            raise ValueError(f"perm_obs_dim {self.perm_obs_dim} outside (0, {obs_dim}]")
        self.perm = mlp(self.perm_obs_dim, list(hidden_sizes), 1, out_gain=1.0,
                        layer_norm=layer_norm)
        self.trans = mlp(obs_dim, list(trans_hidden_sizes), 1, out_gain=1.0,
                         layer_norm=layer_norm)

        # trans_zero_init: V^(T)_0 = 0, Theorem 1's condition.
        #
        # NOTE THIS IS A DEVIATION FROM THE REFERENCE CODE, not a restoration of it. The paper does
        # not specify initialisation at all — Algs. 1/2/4 say only "Initialize theta, w", and §3.3
        # says "The initialization and resets are done appropriately based on the function
        # approximation used." The reference implementation random-initialises BOTH nets
        # (`T_Net = CNN_half(...)`, `P_Net = CNN_half(...)`), so its own deep agent does not satisfy
        # V^(T)_0 = 0 either.
        #
        # Zeroing is defensible — it stops the acting value starting as the sum of two independent
        # random functions — but it is a choice made toward a TABULAR theorem, and the paper's
        # positive empirical results come from the code, which does not make it. Set False to run
        # the reference's actual initialisation.
        #
        # Note also that Theorem 1 requires V^(TD)_0 = V^(P) as well: the TD baseline must start
        # from the SAME function as the permanent. Vanilla's critic here is an independent random
        # draw, so the equivalence never strictly applies whatever we do here.
        if trans_zero_init:
            nn.init.constant_(self.trans[-1].weight, 0.0)
            nn.init.constant_(self.trans[-1].bias, 0.0)

        # perm_zero_init: V^(P)_0 = 0 as well.
        #
        # Theorem 1's condition is V^(TD)_0 = V^(P), V^(T)_0 = 0. In the TABULAR setting that
        # licenses any initialisation of V^(P), because a table represents an arbitrary offset
        # exactly and the equivalence with TD is free. UNDER FUNCTION APPROXIMATION IT IS NOT
        # FREE. With alpha_P small by design, theta_P barely moves (measured drift_from_init ~0.3
        # against value magnitudes of O(1)), so V_perm stays dominated by its random init for the
        # whole run. The transient's regression target is then R - V_perm: the value function minus
        # a FIXED, unstructured, high-frequency function it must cancel on every state it visits.
        # It can memorise that cancellation on states it trains on; it cannot on NEW ones — which
        # is precisely the post-switch regime, where the state distribution shifts.
        #
        # Measured consequence (Job D): with the mechanism entirely off, theta_P frozen, and the
        # transient given capacity exactly matching vanilla's critic, PT was still significantly
        # worse than vanilla after a switch (whole-run p=0.001, phase 2 p=0.001, phase 4 p=0.008)
        # while being indistinguishable before one (phase 1 p=0.940).
        #
        # With V_perm == 0 the critic loss (V_perm.detach() + V_trans - R)^2 reduces EXACTLY to
        # vanilla's (V - R)^2, so this is also the only configuration in which the Theorem 1
        # equivalence test is meaningful under deep FA.
        if perm_zero_init:
            nn.init.constant_(self.perm[-1].weight, 0.0)
            nn.init.constant_(self.perm[-1].bias, 0.0)
        elif perm_init_std is not None:
            # The reference's DEEP code uses neither exactly zero nor our orthogonal gain 1.0:
            #
            #   prediction_semi_crl/minigrid/model.py
            #       def weight_init(m):
            #           if isinstance(m, nn.Linear):
            #               nn.init.normal_(m.weight, 0, 0.01); m.bias.data.fill_(0.0)
            #
            # Only the tabular/linear versions use w_1 = np.zeros. That distinction matters here
            # because Job H showed zero-initialising a critic costs ~300 return points on this
            # task — it hits vanilla just as hard — so "exactly zero" is not a free choice. This
            # reproduces the deep reference: small but non-zero.
            for m in self.perm:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, perm_init_std)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        """Returns (v_perm, v_trans), each shape (batch,)."""
        return self.perm_forward(obs).squeeze(-1), self.trans(obs).squeeze(-1)

    def perm_forward(self, obs):
        """V_P(s), with the task-label tail sliced off. Always use this, never `self.perm(obs)` —
        see the matching note on SplitGaussianActor.perm_forward."""
        return self.perm(obs[..., :self.perm_obs_dim])

    def value(self, obs):
        v_perm, v_trans = self.forward(obs)
        return v_perm + v_trans

    @torch.no_grad()
    def decay_transient(self, decay, mode="params"):
        """Decay the transient. Algorithm 2 line 9 is `w <- lambda*w` on the VALUE FUNCTION.

        mode="params" (default, reproduces every run before 2026-08-04, and the reference):
            `for p in trans.parameters(): p.data *= decay` — verbatim the reference's
            `for params in T_Net.parameters(): params.data *= args.decay`.
            For a nonlinear net this does NOT scale the output by `decay`. Measured on [43,43]:
            decay=0.75 leaves ~0.42 of V_trans, not 0.75 — we delete 58% where the algorithm
            says 25%. Only decay=0 is exact.

        mode="output" (EXACT):
            scale only the final Linear layer. Because that layer is affine,
                W*h + b  ->  decay*W*h + decay*b  =  decay*(W*h + b)
            so V_trans <- decay * V_trans EXACTLY, for any decay, which is what Alg. 2 specifies.
            The hidden features are left intact — the transient keeps its learned representation
            and only its magnitude is reduced, which is also the natural reading of "decay the
            transient value function" rather than "decay the transient network's weights".

        Why this matters: the decay is a FIXED COST paid every k updates whether or not there is
        non-stationarity to exploit. Over-decaying by 40% inflates that cost throughout training.
        See FINDINGS.md 6.1(a) and 8.7.
        """
        if mode == "output":
            self.trans[-1].weight.data.mul_(decay)
            self.trans[-1].bias.data.mul_(decay)
        else:
            for p in self.trans.parameters():
                p.data.mul_(decay)
