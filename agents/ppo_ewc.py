"""Online Elastic Weight Consolidation (EWC) on PPO's actor.

Vanilla PPO plus a quadratic penalty toward the weights at the last consolidation, weighted by a
running diagonal Fisher:
    L(theta) = L_PPO(theta) + (lambda/2) sum_i F_i (theta_i - theta*_i)^2
    F_new = gamma * F_old + F_current
    F_current,i = (1/N) sum_t (d log pi(a_t|s_t) / d theta_i)^2
"""
import copy

import numpy as np
import torch

from .ppo_vanilla import PPOVanilla


class PPOEWC(PPOVanilla):
    """PPO + Online EWC regularization on the actor (policy) network."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        self.ewc_lambda = cfg.get("ewc_lambda", 50.0)
        # The penalty covers every actor parameter, including log_std, so EWC also anchors the
        # exploration noise. True excludes log_std from both the Fisher and the penalty.
        self.ewc_exclude_log_std = bool(cfg.get("ewc_exclude_log_std", False))

        # Online EWC (Schwarz et al. 2018): one running Fisher, F_new = gamma * F_old + F_current,
        # and a single anchor at the most recent consolidation.
        self.ewc_gamma = float(cfg.get("ewc_gamma", 0.95))
        self.fisher = {}
        self.anchor = {}
        # One entry per consolidation; used to count them.
        self.past_tasks = []

        # ewc_consolidate_every = n consolidates every n PPO updates, for benchmarks without task
        # boundaries, where on_task_switch never fires and EWC would equal vanilla PPO. Set it to
        # pt's k for the same cadence. 0 (default) consolidates at task boundaries only.
        self.ewc_consolidate_every = int(cfg.get("ewc_consolidate_every", 0))
        self._updates_since_ewc_consolidation = 0

    def _ewc_penalty(self):
        """Return (lambda/2) * sum_i F_i (theta_i - theta*_i)^2.

        Not divided by the parameter count, so `ewc_lambda` has its standard meaning.
        """
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.actor.named_parameters():
            if self.ewc_exclude_log_std and n.endswith("log_std"):
                continue
            if n in self.fisher:
                penalty = penalty + (self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
        return (self.ewc_lambda / 2.0) * penalty

    def _extra_loss(self):
        """EWC quadratic penalty for per-step online updates."""
        if self.fisher:
            return self._ewc_penalty()
        return torch.tensor(0.0, device=self.device)

    def update(self, last_obs, last_done, update_idx):
        """PPO update with the EWC penalty added to each minibatch loss."""
        v_perm_last, v_trans_last = self.get_value(last_obs)
        last_value = v_perm_last + v_trans_last
        advantages, returns = self.buffer.compute_gae(
            last_value, last_done, self.gamma, self.gae_lambda
        )

        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        batch = self.buffer.get_tensors()
        n = batch["obs"].shape[0]          # flattened batch = n_steps * num_envs
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

                # Actor loss (clipped surrogate)
                new_logprobs, entropy = self.actor.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # Critic loss (PPOVanilla.critic_loss)
                c_loss = self.critic_loss(
                    {"obs": mb_obs, "actions": mb_actions,
                     "v_perm": batch["v_perm"][mb_idx],
                     "v_trans": batch["v_trans"][mb_idx]},
                    mb_adv, mb_ret,
                )

                # EWC penalty, active after the first consolidation
                ewc_pen = torch.tensor(0.0, device=self.device)
                if self.fisher:
                    ewc_pen = self._ewc_penalty()

                loss = actor_loss + c_loss + self.ent_coef * entropy_loss + ewc_pen

                self.actor_optim.zero_grad()
                self._zero_critic_grads()
                loss.backward()
                self._clip_grads()      # actor and critic clipped separately, see PPOBase
                self.actor_optim.step()
                self._step_critic_optims()

                total_actor_loss += actor_loss.item()
                total_critic_loss += c_loss.item()
                total_entropy += entropy.mean().item()
                total_ewc_penalty += ewc_pen.item()
                n_updates += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

            # Optional early stop on KL.
            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                break

        self.post_update(update_idx)

        return {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": approx_kl,
            "ewc_penalty": total_ewc_penalty / max(n_updates, 1),
            # This class overrides update(), so it must log sigma itself like PPOBase does.
            "log_std_mean": float(self.actor.log_std.detach().mean()),
            "log_std_min": float(self.actor.log_std.detach().min()),
        }

    def post_update(self, update_idx):
        """Consolidate every `ewc_consolidate_every` updates (if > 0) via `on_task_switch`."""
        if self.ewc_consolidate_every <= 0:
            return
        self._updates_since_ewc_consolidation += 1
        if self._updates_since_ewc_consolidation >= self.ewc_consolidate_every:
            self._updates_since_ewc_consolidation = 0
            self.on_task_switch(int(update_idx))

    def on_task_switch(self, step):
        """Accumulate the Fisher into the running estimate and re-anchor at this boundary."""
        batch = self.buffer.get_tensors()
        obs, actions = batch["obs"], batch["actions"]
        n_samples = obs.shape[0]

        current = {n: torch.zeros_like(p, device=self.device)
                   for n, p in self.actor.named_parameters()}
        for i in range(n_samples):
            self.actor.zero_grad()
            logprob, _ = self.actor.evaluate_actions(obs[i:i + 1], actions[i:i + 1])
            logprob.backward()
            with torch.no_grad():
                for n, p in self.actor.named_parameters():
                    if self.ewc_exclude_log_std and n.endswith("log_std"):
                        continue
                    if p.grad is not None:
                        current[n] += (p.grad.detach() ** 2) / n_samples
        self.actor.zero_grad()

        # F_new = gamma * F_old + F_current, then re-anchor on the weights that produced it.
        for n, f in current.items():
            self.fisher[n] = self.ewc_gamma * self.fisher[n] + f if n in self.fisher else f
        self.anchor = {n: p.clone().detach() for n, p in self.actor.named_parameters()}
        self.past_tasks.append({"step": step})

        total = float(sum(f.sum() for f in self.fisher.values()))
        print(f"[EWC] on_task_switch at step {step}: Fisher over {n_samples} samples, "
              f"accumulated with gamma={self.ewc_gamma} (running trace {total:.4e}), "
              f"anchor re-set. Boundaries seen: {len(self.past_tasks)}.")
