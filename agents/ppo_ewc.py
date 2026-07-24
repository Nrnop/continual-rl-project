"""Online Elastic Weight Consolidation (EWC) applied to PPO.

Regularization-based continual-learning baseline.  Inherits the single-critic
PPO architecture from PPOVanilla and augments the actor loss with a quadratic
penalty that discourages movement away from weights that were important for
previous tasks, estimated via a running diagonal Fisher Information matrix.

Mathematical specification
--------------------------
Total actor loss:
    L_total(θ) = L_PPO(θ) + (λ/2) Σ_i  F̃_i (θ_i − θ*_i,old)²

Online Fisher update at each task switch:
    F̃_new  = γ · F̃_old  + F_current

where F_current is the diagonal empirical Fisher accumulated during the
most-recent task block:
    F_i = (1/N) Σ_t (∂ log π(a_t|s_t) / ∂θ_i)²
"""
import copy

import numpy as np
import torch

from .ppo_vanilla import PPOVanilla


class PPOEWC(PPOVanilla):
    """PPO + Online EWC regularization on the actor (policy) network."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        # EWC hyper-parameters
        self.ewc_lambda = cfg.get("ewc_lambda", 50.0)
        
        # Store past tasks: list of dicts with 'fisher' and 'anchor'
        self.past_tasks = []

    # ------------------------------------------------------------------
    # EWC helpers
    # ------------------------------------------------------------------
    def _ewc_penalty(self):
        """Compute the EWC quadratic penalty on actor parameters.

        Returns a scalar tensor attached to the actor's computation graph.
        """
        penalty = torch.tensor(0.0, device=self.device)
        for task in self.past_tasks:
            fisher = task["fisher"]
            anchor = task["anchor"]
            for n, p in self.actor.named_parameters():
                penalty = penalty + (fisher[n] * (p - anchor[n]) ** 2).sum()
        
        # Scale by number of parameters to keep the penalty intensive (mean-like)
        n_params = sum(p.numel() for p in self.actor.parameters())
        penalty = penalty / max(n_params, 1)

        return (self.ewc_lambda / 2.0) * penalty

    # ------------------------------------------------------------------
    # Overridden update  (injects Fisher accumulation + EWC penalty)
    # ------------------------------------------------------------------
    def update(self, last_obs, last_done, update_idx):
        """PPO update with Online EWC penalty on the actor loss.

        Fisher is accumulated once over the full rollout buffer before the
        PPO epoch loop.  The EWC penalty is added to the loss inside each
        minibatch iteration (cheap — just a dot product, no extra backward).
        """
        # Bootstrap value for GAE
        v_perm_last, v_trans_last = self.get_value(last_obs)
        last_value = v_perm_last + v_trans_last
        advantages, returns = self.buffer.compute_gae(
            last_value, last_done, self.gamma, self.gae_lambda
        )

        # Convert to tensors
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        batch = self.buffer.get_tensors()
        n = self.cfg["n_steps"]
        mb = self.minibatch_size

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_ewc_penalty = 0.0
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

                # --- Actor loss (clipped surrogate) ---
                new_logprobs, entropy = self.actor.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # --- Critic loss (delegated to PPOVanilla.critic_loss) ---
                c_loss = self.critic_loss(
                    {"obs": mb_obs, "actions": mb_actions,
                     "v_perm": batch["v_perm"][mb_idx],
                     "v_trans": batch["v_trans"][mb_idx]},
                    mb_adv, mb_ret,
                )

                # --- EWC penalty (active only after the first task switch) ---
                ewc_pen = torch.tensor(0.0, device=self.device)
                if len(self.past_tasks) > 0:
                    ewc_pen = self._ewc_penalty()

                loss = actor_loss + c_loss + self.ent_coef * entropy_loss + ewc_pen

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
                total_ewc_penalty += ewc_pen.item()
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
            "ewc_penalty": total_ewc_penalty / max(n_updates, 1),
        }

    # ------------------------------------------------------------------
    # Task-switch hook
    # ------------------------------------------------------------------
    def on_task_switch(self, step):
        """Estimate Fisher exactly once at the boundary and snapshot anchor weights."""
        batch = self.buffer.get_tensors()
        obs = batch["obs"]
        actions = batch["actions"]
        n_samples = obs.shape[0]

        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.actor.named_parameters()}

        # Compute empirical Fisher by averaging squared individual gradients
        for i in range(n_samples):
            self.actor.zero_grad()
            logprob, _ = self.actor.evaluate_actions(obs[i:i+1], actions[i:i+1])
            logprob.backward()
            with torch.no_grad():
                for n, p in self.actor.named_parameters():
                    if p.grad is not None:
                        fisher[n] += (p.grad.detach() ** 2) / n_samples
        
        self.actor.zero_grad()

        # Snapshot anchor weights
        anchor = {n: p.clone().detach() for n, p in self.actor.named_parameters()}

        self.past_tasks.append({
            "fisher": fisher,
            "anchor": anchor
        })

        print(f"[EWC] on_task_switch at step {step}: "
              f"Fisher computed over {n_samples} samples, anchor saved. "
              f"Total past tasks: {len(self.past_tasks)}.")
