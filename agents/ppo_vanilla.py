"""Vanilla single-critic PPO baseline.

The comparison point for the PT agent — same actor, same env, same seeds, but a single undivided
V(s) critic. This is the continuous-control analogue of the baseline DQN in PT_DQN_half.py.
"""
import numpy as np
import torch

from .ppo_base import PPOBase
from ..models.critic import VanillaCritic


class PPOVanilla(PPOBase):
    """Standard PPO with a single state-value critic."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        # critic_zero_init makes V(s)=0 at t=0, so a PT arm run with perm_zero_init starts from the
        # SAME function — Theorem 1's second condition, V^(TD)_0 = V^(P). See VanillaCritic.
        self.critic = VanillaCritic(
            obs_dim, hidden_sizes=self._critic_hidden(),
            zero_init=bool(cfg.get("critic_zero_init", False)),
        ).to(device)
        lr_critic = cfg.get("lr_critic", cfg["lr_actor"])
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic, eps=cfg.get("adam_eps", 1e-8)
        )

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------
    def get_value(self, obs_batch_np):
        """Return (V, 0) for a batch — perm slot carries the full value, trans is zero.

        obs_batch_np: (num_envs, obs_dim) -> arrays (num_envs,).
        """
        obs_t = torch.as_tensor(obs_batch_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.critic(obs_t).cpu().numpy()
        return v, np.zeros_like(v)

    def critic_loss(self, batch, advantages, returns):
        """MSE between predicted V and GAE returns."""
        v_pred = self.critic(batch["obs"])
        return 0.5 * ((v_pred - returns) ** 2).mean()

    def post_update(self, update_idx):
        # Vanilla has no consolidation cadence, but the base implements `policy_shrink_every`
        # (disabled by default), so this must delegate rather than no-op — otherwise the flag is
        # silently dead for the one agent it exists to test.
        super().post_update(update_idx)

    # ------------------------------------------------------------------
    # Critic optimizer plumbing
    # ------------------------------------------------------------------
    def _zero_critic_grads(self):
        self.critic_optim.zero_grad()

    def _step_critic_optims(self):
        self.critic_optim.step()

    def _critic_parameters(self):
        return self.critic.parameters()
