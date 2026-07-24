"""Dual-timescale Permanent-Transient PPO agent.

Uses an online dual-timescale value update to elegantly separate invariant and task-specific values:
- V_perm (θ_P): Trained online to predict the full returns using a slow learning rate (e.g., 1e-5). 
                Acts as a slow moving average, naturally finding the invariant task baseline.
- V_trans (θ_T): Trained online to predict the residual (returns - V_perm.detach()) using a fast LR.
                 Quickly adapts to the current task's specific value function.

At task boundaries, θ_T is wiped clean (hard reset to 0.0) to prevent any negative transfer,
while θ_P is preserved, carrying over the invariant value representations.
"""
import torch

from .ppo_base import PPOBase
from ..models.critic import SplitCritic
from ..models.actor import SplitActor


class PPOPT(PPOBase):
    """PPO with a dual-timescale split critic (Permanent + Transient)."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        hidden = list(cfg.get("hidden_sizes", [256, 256]))
        self.critic = SplitCritic(obs_dim, hidden_sizes=hidden).to(device)

        # Fast optimizer for the transient head (task-specific)
        lr_trans = cfg.get("lr_trans", cfg["lr_actor"])
        self.trans_optim = torch.optim.Adam(self.critic.trans.parameters(), lr=lr_trans)

        # Slow optimizer for the permanent head (task-invariant)
        lr_perm = cfg.get("lr_perm", 3e-5)
        perm_opt_name = cfg.get("perm_optimizer", "adam").lower()
        if perm_opt_name == "sgd":
            self.perm_optim = torch.optim.SGD(self.critic.perm.parameters(), lr=lr_perm)
        else:
            self.perm_optim = torch.optim.Adam(self.critic.perm.parameters(), lr=lr_perm)

        # Replace vanilla actor with SplitActor
        self.actor = SplitActor(obs_dim, act_dim, hidden_sizes=hidden).to(device)
        self.actor_optim = torch.optim.Adam([
            {"params": self.actor.trans_mean.parameters(), "lr": lr_trans},
            {"params": self.actor.perm_mean.parameters(), "lr": lr_perm},
            {"params": [self.actor.log_std], "lr": lr_trans}
        ])

        # Decay factor for the transient head at task boundaries (hard reset to wipe residuals)
        self.transient_decay = cfg.get("decay", 0.0)

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------
    def get_value(self, obs_np):
        """Return (V_perm, V_trans) as python floats."""
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v_perm, v_trans = self.critic(obs_t)
        return v_perm.item(), v_trans.item()

    def critic_loss(self, batch, advantages, returns):
        """Joint online update for both timescales.
        
        θ_P (slow): MSE(V_perm, returns)
        θ_T (fast): MSE(V_perm.detach() + V_trans, returns)
        """
        v_perm, v_trans = self.critic(batch["obs"])
        
        # Transient loss (fast): learns the residual above the permanent estimate
        v_combined = v_perm.detach() + v_trans
        loss_trans = 0.5 * ((v_combined - returns) ** 2).mean()

        # Permanent loss (slow): learns the slow exponential moving average of the returns
        loss_perm = 0.5 * ((v_perm - returns) ** 2).mean()

        # Return sum; optimizers step their respective parameters cleanly
        return loss_trans + loss_perm

    def post_update(self, update_idx):
        """No periodic consolidation needed in the online dual-timescale formulation."""
        pass

    def on_task_switch(self, step):
        """Smoothly handle task boundary without advantage shock."""
        if self.transient_decay < 1.0:
            self.critic.decay_transient(self.transient_decay)
            self.actor.decay_transient(self.transient_decay)

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
