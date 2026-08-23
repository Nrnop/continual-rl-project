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

NOTE ON `ewc_lambda` UNITS. Before 2026-08-12 the penalty was divided by the actor's parameter
count, so a configured lambda of 50 acted as 50/5708 = 0.0088. The division is gone; lambda now
carries its standard meaning. To reproduce a pre-2026-08-12 run, divide its lambda by 5708.
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
        # The Fisher penalty runs over actor.named_parameters(), which INCLUDES `log_std`. EWC
        # therefore anchors the exploration schedule as a side effect: measured on HalfCheetah,
        # EWC holds log_std at -1.97 for the whole run while vanilla decays to -2.48 and PT to
        # -2.63, and the return ranking follows that ordering exactly (FULL_PT.md §2).
        # Setting this true excludes log_std from BOTH the Fisher and the penalty, so the arm
        # measures weight protection alone. Default False = the behaviour of every earlier run.
        self.ewc_exclude_log_std = bool(cfg.get("ewc_exclude_log_std", False))

        # ONLINE EWC: ONE running Fisher, decayed by gamma at each boundary — not a growing list.
        #
        # `ewc_gamma` was previously set in ppo_ewc.yaml and exposed on the CLI while NO code read
        # it, so the agent accumulated a separate (Fisher, anchor) pair per boundary and summed
        # their penalties. The policy therefore became monotonically more rigid with every task
        # (measured penalty: 0 -> 0.0054 -> 0.0139 -> 0.0167 -> 0.0185 across the five phases),
        # which is the opposite of what online EWC does and is worst exactly where adaptation
        # matters most. Now:
        #
        #     F_new = gamma * F_old + F_current           (Kirkpatrick et al.; Schwarz et al. 2018)
        #
        # with a single anchor at the most recent boundary.
        self.ewc_gamma = float(cfg.get("ewc_gamma", 0.95))
        self.fisher = {}
        self.anchor = {}
        # Kept so existing code and tests that count consolidations keep working.
        self.past_tasks = []

        # TIMER-BASED CONSOLIDATION — required for the boundary-free (Phase 2b) benchmark.
        #
        # EWC accumulates its Fisher in `on_task_switch`. Under smooth Lipschitz drift there ARE no
        # task switches, so that hook never fires, the Fisher stays empty, the penalty is
        # identically zero, and this agent becomes BIT-IDENTICAL to vanilla PPO. Running it in that
        # state is not a weak baseline, it is no baseline: any "PT beats EWC" claim would only be
        # saying that EWC was switched off.
        #
        # `ewc_consolidate_every = n` accumulates the Fisher and re-anchors every n PPO updates
        # instead. Set it to `pt`'s `k` and the two methods consolidate on exactly the same cadence,
        # which is the fair comparison: same schedule, different mechanism.
        #
        # 0 (the default) keeps the boundary-triggered behaviour of every run made before
        # 2026-08-21, so no existing result changes.
        self.ewc_consolidate_every = int(cfg.get("ewc_consolidate_every", 0))
        self._updates_since_ewc_consolidation = 0

    # ------------------------------------------------------------------
    # EWC helpers
    # ------------------------------------------------------------------
    def _ewc_penalty(self):
        """The standard EWC penalty: (lambda/2) * sum_i F_i (theta_i - theta*_i)^2.

        NO division by the parameter count. The previous implementation divided the sum by
        `n_params` (5,708 on HalfCheetah), which silently redefined `ewc_lambda`: a configured
        lambda of 50 was really an effective 0.0088, so the value was never tuned in the units the
        docstring — or any paper — states. `ewc_lambda` now means what the literature means, and
        `EWC_LAMBDA_LEGACY_SCALE` below records the conversion for reading old runs.
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
                if self.fisher:
                    ewc_pen = self._ewc_penalty()

                loss = actor_loss + c_loss + self.ent_coef * entropy_loss + ewc_pen

                self.actor_optim.zero_grad()
                self._zero_critic_grads()
                loss.backward()
                self._clip_grads()      # actor and critic clipped SEPARATELY — see PPOBase
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
            # EWC overrides update(), so PPOBase's sigma logging never reached this arm — the same
            # gap that left `pt` without log_std_min. HALFCHEETAH_RESULTS.md records the original defect
            # ("sigma logged by pt only") as fixed in PPOBase, but a fix in the base class does not
            # reach a subclass that returns its own metrics dict. Any arm overriding update() must
            # report these itself, or the comparison silently loses the one quantity that has
            # already confounded this project once.
            "log_std_mean": float(self.actor.log_std.detach().mean()),
            "log_std_min": float(self.actor.log_std.detach().min()),
        }

    # ------------------------------------------------------------------
    # Task-switch hook
    # ------------------------------------------------------------------
    def post_update(self, update_idx):
        """Timer-based consolidation, for benchmarks that have no task boundaries.

        Inactive unless `ewc_consolidate_every > 0`, so every boundary-based run behaves exactly as
        before. When active it reuses `on_task_switch` verbatim — same Fisher estimate, same
        gamma-decayed accumulation, same re-anchoring — only the trigger differs.
        """
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
