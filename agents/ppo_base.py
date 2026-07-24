"""Shared PPO core: rollout collection, GAE, and clipped-surrogate update.

Both PPOVanilla and PPOPT inherit from this base. The base drives the collect→compute-GAE→update
loop and delegates critic-specific behaviour to three abstract hooks:

    get_value(obs)    → (v_perm, v_trans)     [vanilla returns (V, 0)]
    critic_loss(…)    → scalar loss            [added to actor loss]
    post_update(idx)  → None                   [PT uses this for consolidation cadence]

Design mirrors the baseline's separation between training logic (the loop in PT_DQN_half.py)
and the network update functions (train_T_Net / train_P_Net), but factored into OOP.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch

from ..models.actor import GaussianActor
from ..utils.buffers import RolloutBuffer


class PPOBase(ABC):
    """Abstract PPO agent with pluggable critic."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.device = device

        # --- actor (shared by both agents) ---
        hidden = list(cfg.get("hidden_sizes", [256, 256]))
        self.actor = GaussianActor(obs_dim, act_dim, hidden_sizes=hidden).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr_actor"])

        # --- rollout buffer ---
        self.buffer = RolloutBuffer(cfg["n_steps"], obs_dim, act_dim, device)

        # hyper-params cached for convenience
        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.clip_coef = cfg["clip_coef"]
        self.epochs = cfg["epochs"]
        self.minibatch_size = cfg["minibatch_size"]
        self.ent_coef = cfg.get("ent_coef", 0.0)
        self.max_grad_norm = cfg["max_grad_norm"]
        self.target_kl = cfg.get("target_kl", None)
        self.normalize_advantage = cfg.get("normalize_advantage", True)

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def get_value(self, obs_np):
        """Return (v_perm, v_trans) as python floats for a SINGLE observation (numpy)."""

    @abstractmethod
    def critic_loss(self, batch, advantages, returns):
        """Compute the critic loss tensor given a minibatch dict, advantages, and returns."""

    @abstractmethod
    def post_update(self, update_idx):
        """Called after each complete PPO update (all epochs). Used by PT for consolidation."""

    def on_task_switch(self, step):
        """Called when the training loop detects a task boundary. Override in PT."""

    def state_dict(self):
        """Return dict of network state dicts for saving checkpoints."""
        state = {"actor": self.actor.state_dict()}
        if hasattr(self, "critic"):
            state["critic"] = self.critic.state_dict()
        return state

    def load_state_dict(self, state):
        """Load network weights from state dict."""
        if "actor" in state and hasattr(self, "actor"):
            self.actor.load_state_dict(state["actor"])
        if "critic" in state and hasattr(self, "critic"):
            self.critic.load_state_dict(state["critic"])

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect_rollout(self, env, obs, done):
        """Fill the RolloutBuffer for n_steps.

        Returns (last_obs, last_done, episode_returns) where episode_returns is a list of
        cumulative returns for each episode that completed during this rollout.
        """
        self.buffer.reset()
        episode_returns = []
        if not hasattr(self, '_epi_return'):
            self._epi_return = 0.0

        for _ in range(self.cfg["n_steps"]):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, logprob = self.actor.act(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            v_perm, v_trans = self.get_value(obs)

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done_flag = float(terminated or truncated)

            self.buffer.add(obs, action_np, logprob.item(), reward, done_flag, v_perm, v_trans)
            self._epi_return += reward

            obs = next_obs
            done = done_flag

            if terminated or truncated:
                episode_returns.append(self._epi_return)
                self._epi_return = 0.0
                obs, _ = env.reset()
                done = 0.0

        return obs, done, episode_returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self, last_obs, last_done, update_idx):
        """Run the full PPO update (GAE + multiple epochs of minibatch SGD).

        Returns a dict of scalar metrics.
        """
        # Bootstrap value for GAE
        v_perm_last, v_trans_last = self.get_value(last_obs)
        last_value = v_perm_last + v_trans_last
        advantages, returns = self.buffer.compute_gae(last_value, last_done,
                                                      self.gamma, self.gae_lambda)

        # Convert to tensors
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        batch = self.buffer.get_tensors()
        n = self.cfg["n_steps"]
        mb = self.minibatch_size

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        approx_kl = 0.0

        for _epoch in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, mb):
                end = start + mb
                mb_idx = idx[start:end]

                mb_obs = batch["obs"][mb_idx]
                mb_actions = batch["actions"][mb_idx]
                mb_old_logprobs = batch["logprobs"][mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                if self.normalize_advantage and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Actor loss (clipped surrogate)
                new_logprobs, entropy = self.actor.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # Critic loss (delegated to subclass)
                c_loss = self.critic_loss(
                    {"obs": mb_obs, "actions": mb_actions,
                     "v_perm": batch["v_perm"][mb_idx], "v_trans": batch["v_trans"][mb_idx]},
                    mb_adv, mb_ret,
                )

                loss = actor_loss + c_loss + self.ent_coef * entropy_loss

                self.actor_optim.zero_grad()
                self._zero_critic_grads()
                loss.backward()
                # Grad clip over all trainable params
                all_params = list(self.actor.parameters()) + list(self._critic_parameters())
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.actor_optim.step()
                self._step_critic_optims()

                total_actor_loss += actor_loss.item()
                total_critic_loss += c_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

            # Optional early stop on KL
            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                break

        self.post_update(update_idx)

        return {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": approx_kl,
        }

    # ------------------------------------------------------------------
    # Helpers that subclasses override to register their critic optimizers
    # ------------------------------------------------------------------
    @abstractmethod
    def _zero_critic_grads(self):
        """Zero-grad all critic optimizers."""

    @abstractmethod
    def _step_critic_optims(self):
        """Step all critic optimizers."""

    @abstractmethod
    def _critic_parameters(self):
        """Return an iterable of all critic parameters (for grad clipping)."""
