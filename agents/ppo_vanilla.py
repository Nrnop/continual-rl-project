"""Vanilla single-critic PPO baseline.

The comparison point for the PT agent — same actor, same env, same seeds, but a single undivided
V(s) critic. This is the continuous-control analogue of the baseline DQN in PT_DQN_half.py.
"""
import torch

from .ppo_base import PPOBase
from ..models.critic import VanillaCritic


class PPOVanilla(PPOBase):
    """Standard PPO with a single state-value critic."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        hidden = list(cfg.get("hidden_sizes", [256, 256]))
        self.critic = VanillaCritic(obs_dim, hidden_sizes=hidden).to(device)
        lr_critic = cfg.get("lr_critic", cfg["lr_actor"])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------
    def get_value(self, obs_np):
        """Return (V, 0.0) — perm slot carries the full value, trans is zero."""
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.critic(obs_t).item()
        return v, 0.0

    def critic_loss(self, batch, advantages, returns):
        """MSE between predicted V and GAE returns."""
        v_pred = self.critic(batch["obs"])
        return 0.5 * ((v_pred - returns) ** 2).mean()

    def post_update(self, update_idx):
        pass  # no-op for vanilla

    # ------------------------------------------------------------------
    # Critic optimizer plumbing
    # ------------------------------------------------------------------
    def _zero_critic_grads(self):
        self.critic_optim.zero_grad()

    def _step_critic_optims(self):
        self.critic_optim.step()

    def _critic_parameters(self):
        return self.critic.parameters()
