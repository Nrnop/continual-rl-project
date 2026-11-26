"""Vanilla PPO baseline with a single state-value critic.

Same actor, environment and seeds as the PT agent, with one undivided V(s) critic.
"""
import numpy as np
import torch

from .ppo_base import PPOBase
from ..models.critic import VanillaCritic


class PPOVanilla(PPOBase):
    """Standard PPO with a single state-value critic."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        # critic_zero_init makes V(s) = 0 at t = 0, so a PT arm with perm_zero_init starts from the
        # same function (Theorem 1's second condition, V^(TD)_0 = V^(P)).
        self.critic = VanillaCritic(
            obs_dim, hidden_sizes=self._critic_hidden(),
            zero_init=bool(cfg.get("critic_zero_init", False)),
            layer_norm=bool(cfg.get("layer_norm", False)),
        ).to(device)
        lr_critic = cfg.get("lr_critic", cfg["lr_actor"])
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic, eps=cfg.get("adam_eps", 1e-8)
        )

        # Control: apply the same output-layer decay to a single network, to separate the effect of
        # the decay from the split. Off unless `output_decay_every` > 0.
        self.output_decay_every = int(cfg.get("output_decay_every", 0))
        self.output_decay_factor = float(cfg.get("output_decay_factor", 0.5))
        if not 0.0 <= self.output_decay_factor <= 1.0:
            raise ValueError("output_decay_factor must be between 0 and 1")
        self._updates_since_decay = 0
        self.output_decays = 0

    def get_value(self, obs_batch_np):
        """Return (V, 0): the perm slot carries the full value, trans is zero.

        obs_batch_np: (num_envs, obs_dim) -> two arrays of shape (num_envs,).
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
        """No-op unless the output-decay control is on (`output_decay_every` > 0)."""
        if self.output_decay_every <= 0:
            return
        self._updates_since_decay += 1
        if self._updates_since_decay >= self.output_decay_every:
            self._decay_outputs()

    def on_task_switch(self, step):
        # PT also consolidates (and so decays) at every boundary.
        if self.output_decay_every > 0:
            self._decay_outputs()

    def _decay_outputs(self):
        """Scale the actor-mean and critic output layers, then clear their Adam state.

        Matches what PT does to its transient: decay_transient scales the final Linear's weight
        and bias, and _consolidate then clears the transient optimizer state.
        """
        with torch.no_grad():
            for layer in (self.actor.mean_net[-1], self.critic.net[-1]):
                layer.weight.mul_(self.output_decay_factor)
                layer.bias.mul_(self.output_decay_factor)
        for parameter in self.actor.mean_net.parameters():
            self.actor_optim.state.pop(parameter, None)
        for parameter in self.critic.parameters():
            self.critic_optim.state.pop(parameter, None)
        self._updates_since_decay = 0
        self.output_decays += 1

    # Critic optimizer hooks
    def _zero_critic_grads(self):
        self.critic_optim.zero_grad()

    def _step_critic_optims(self):
        self.critic_optim.step()

    def _critic_parameters(self):
        return self.critic.parameters()
