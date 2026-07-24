"""Dual-timescale Permanent-Transient PPO agent (value-function decomposition).

PT is applied to the CRITIC only (matching Anand & Precup): V(s) = V_perm(s) + V_trans(s).
The policy is the same single GaussianActor as the vanilla/EWC baselines, so the comparison
stays apples-to-apples and any difference is attributable to the split critic.

- V_trans (θ_T): fast head, trained every PPO update to predict the residual
                 (returns - V_perm.detach()) above the frozen permanent baseline.
- V_perm  (θ_P): slow head, NOT trained on returns each step. Every k updates it *consolidates*
                 by absorbing the transient (see _consolidate), then θ_T is decayed.

Consolidation is value-preserving for any decay: θ_P regresses to old_θ_P + (1-decay)·V_trans,
so after θ_T ← decay·θ_T the acting value V = V_perm + V_trans is unchanged (no drift). At a task
boundary we consolidate first (locking the just-learned task value into θ_P) then let θ_T re-adapt.
"""
import numpy as np
import torch

from .ppo_base import PPOBase
from ..models.critic import SplitCritic
from ..utils.buffers import ConsolidationBuffer


class PPOPT(PPOBase):
    """PPO with a dual-timescale split critic (Permanent + Transient)."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        hidden = list(cfg.get("hidden_sizes", [256, 256]))
        self.critic = SplitCritic(obs_dim, hidden_sizes=hidden).to(device)

        adam_eps = cfg.get("adam_eps", 1e-8)

        # Fast optimizer for the transient head (task-specific)
        lr_trans = cfg.get("lr_trans", cfg["lr_actor"])
        self.trans_optim = torch.optim.Adam(
            self.critic.trans.parameters(), lr=lr_trans, eps=adam_eps
        )

        # Slow optimizer for the permanent head (task-invariant)
        lr_perm = cfg.get("lr_perm", 3e-5)
        perm_opt_name = cfg.get("perm_optimizer", "adam").lower()
        if perm_opt_name == "sgd":
            self.perm_optim = torch.optim.SGD(self.critic.perm.parameters(), lr=lr_perm)
        else:
            self.perm_optim = torch.optim.Adam(
                self.critic.perm.parameters(), lr=lr_perm, eps=adam_eps
            )

        # NOTE: PT is critic-only. The actor is the single GaussianActor built by
        # PPOBase.__init__ (lr = lr_actor), identical to vanilla/EWC. We deliberately do NOT
        # split the policy: the paper decomposes the value function, and a split actor with no
        # consolidation just resets the policy at each boundary. Keeping one actor makes the
        # vanilla/PT/EWC comparison clean (only the critic differs).

        # Decay factor for the transient critic head at consolidation / task boundaries.
        self.transient_decay = cfg.get("decay", 0.0)

        # --- Permanent consolidation: the actual PT mechanism (was never wired up) ---
        # theta_P is NOT trained on returns every step. Every k updates it absorbs the
        # transient by regressing V_perm(s) -> old_V_perm(s) + V_trans(s).detach() over a
        # rolling buffer of visited states, then the transient head is decayed. This mirrors
        # train_P_Net in the reference PT_DQN and keeps the permanent critic a slow, stable
        # task-invariant baseline instead of a second full-return regressor.
        self.k = int(cfg.get("k", 10))
        self.consolidation_epochs = int(cfg.get("consolidation_epochs", 1))
        # One consolidation cycle banks k updates x (n_steps * num_envs) states.
        buf_cap = int(cfg.get("consolidation_buffer_size",
                              self.k * cfg["n_steps"] * self.num_envs))
        self.consolidation_buffer = ConsolidationBuffer(buf_cap)
        self._updates_since_consolidation = 0

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------
    def get_value(self, obs_batch_np):
        """Return (V_perm, V_trans) for a batch: (num_envs, obs_dim) -> arrays (num_envs,)."""
        obs_t = torch.as_tensor(obs_batch_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_perm, v_trans = self.critic(obs_t)
        return v_perm.cpu().numpy(), v_trans.cpu().numpy()

    def critic_loss(self, batch, advantages, returns):
        """Transient-only regression during normal PPO updates.

        theta_T (fast): MSE(V_perm.detach() + V_trans, returns) — the transient head
        learns the residual above the *frozen* permanent baseline.

        theta_P is deliberately NOT trained here. Regressing V_perm directly on returns
        every step (the old behaviour) double-counts against the acting value
        V = V_perm + V_trans and, with no target network / no normalization, drove the
        value divergence we saw (critic_loss ~2e5). The permanent critic is updated only
        during consolidation (see post_update / _consolidate).
        """
        v_perm, v_trans = self.critic(batch["obs"])
        v_combined = v_perm.detach() + v_trans
        return 0.5 * ((v_combined - returns) ** 2).mean()

    def post_update(self, update_idx):
        """After each PPO update: bank this rollout's states, consolidate every k updates."""
        # Snapshot visited states with the CURRENT permanent value. theta_P is frozen
        # between consolidations, so this is old_V_perm at visit time. The rollout
        # buffer is (n_steps, num_envs, obs_dim); flatten to a batch of states.
        states = self.buffer.obs.reshape(-1, self.obs_dim)
        with torch.no_grad():
            s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            old_v_perm, _ = self.critic(s_t)
        self.consolidation_buffer.add_batch(states, old_v_perm.cpu().numpy())

        self._updates_since_consolidation += 1
        if self._updates_since_consolidation >= self.k:
            self._consolidate()
            self._updates_since_consolidation = 0

    def _consolidate(self):
        """Absorb the transient value into the permanent critic, then decay the transient.

        For each stored state s: regress V_perm(s) -> old_V_perm(s) + (1-decay)*V_trans(s).detach().
        Decaying V_trans by `decay` afterwards then leaves the acting value EXACTLY preserved for
        any decay:  V_new = P_new + decay*T = old_P + (1-decay)*T + decay*T = old_P + T = V_old.
        (The earlier target old_P + T with a >0 decay double-counted the transient and inflated V
        by decay*T every cycle.)  decay=0 => hard reset, P absorbs all of T.
        """
        if len(self.consolidation_buffer) == 0:
            return
        keep = 1.0 - self.transient_decay
        for _ in range(self.consolidation_epochs):
            for s_mb, old_vp_mb in self.consolidation_buffer.iter_minibatches(
                    self.minibatch_size, self.device):
                v_perm, v_trans = self.critic(s_mb)
                target = old_vp_mb + keep * v_trans.detach()
                loss = 0.5 * ((v_perm - target) ** 2).mean()
                self.perm_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.perm.parameters(), self.max_grad_norm)
                self.perm_optim.step()
        # The transient head has been absorbed into the permanent; decay it back down.
        self.critic.decay_transient(self.transient_decay)
        self.consolidation_buffer.clear()

    def on_task_switch(self, step):
        """At a task boundary, consolidate (absorb T into P) then let T re-adapt to the new task.

        Consolidating BEFORE decaying locks the just-learned task value into the permanent
        baseline, so the transient can be decayed without the acting value V = V_perm + V_trans
        lurching (that boundary lurch is exactly what BoundaryReturnTracker measures). If
        consolidate_on_switch is off, fall back to a bare value-preserving-free transient decay.
        """
        if self.cfg.get("consolidate_on_switch", True):
            self._consolidate()                       # absorbs T into P, then decays T
            self._updates_since_consolidation = 0
        elif self.transient_decay < 1.0:
            self.critic.decay_transient(self.transient_decay)

    # ------------------------------------------------------------------
    # Critic optimizer plumbing
    # ------------------------------------------------------------------
    def _zero_critic_grads(self):
        self.trans_optim.zero_grad()
        self.perm_optim.zero_grad()

    def _step_critic_optims(self):
        self.trans_optim.step()
        self.perm_optim.step()

    def _critic_parameters(self):
        """All critic params (for grad clipping during the main PPO update)."""
        return list(self.critic.trans.parameters()) + list(self.critic.perm.parameters())
