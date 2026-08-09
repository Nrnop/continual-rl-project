"""PT-PPO with permanent-transient actor and critic decompositions."""
import copy
import time

import numpy as np
import torch

from .ppo_base import PPOBase
from ..models.actor import SplitGaussianActor
from ..models.critic import SplitCritic
from ..utils.buffers import StateConsolidationBuffer


class PPOPTFull(PPOBase):
    """PPO with separate permanent and transient policy means and value heads."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        if self.ent_coef != 0.0:
            raise ValueError("pt_full requires ent_coef=0; use kl_prior_coef for regularization")

        self.rho = float(cfg.get("rho", cfg.get("transfer_rate", 0.5)))
        # Defaults to rho -> the preserving split, i.e. every earlier run is unaffected.
        self.decay_rho = float(cfg.get("decay_rho", self.rho))
        if not 0.0 <= self.decay_rho <= 1.0:
            raise ValueError("decay_rho must be between 0 and 1")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be between 0 and 1")
        self.kl_prior_coef = float(cfg.get("kl_prior_coef", 0.01))
        if self.kl_prior_coef < 0.0:
            raise ValueError("kl_prior_coef must be nonnegative")
        self.k = int(cfg.get("k", 10))
        self.consolidation_epochs = int(cfg.get("consolidation_epochs", 1))
        if self.k <= 0 or self.consolidation_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("k, consolidation_epochs, and minibatch_size must be positive")

        hidden = list(cfg.get("hidden_sizes", [256, 256]))
        trans_hidden = list(cfg.get("trans_hidden_sizes", [64, 64]))
        actor_trans_hidden = list(cfg.get("actor_trans_hidden_sizes", trans_hidden))
        critic_trans_hidden = list(cfg.get("critic_trans_hidden_sizes", trans_hidden))

        self.actor = SplitGaussianActor(
            obs_dim,
            act_dim,
            hidden_sizes=hidden,
            trans_hidden_sizes=actor_trans_hidden,
            log_std_init=cfg.get("log_std_init", 0.0),
        ).to(device)
        adam_eps = cfg.get("adam_eps", 1e-8)
        lr_trans = float(cfg.get("lr_trans", cfg["lr_actor"]))
        lr_perm = float(cfg.get("lr_perm", cfg["lr_actor"]))
        lr_perm_actor = float(cfg.get("lr_perm_actor", lr_perm))
        # log_std is frozen (requires_grad=False, Constraint C4) and deliberately excluded here.
        self.actor_optim = torch.optim.Adam(
            self.actor.trans_mean.parameters(),
            lr=lr_trans,
            eps=adam_eps,
        )
        self.perm_actor_optim = torch.optim.Adam(
            self.actor.perm_mean.parameters(), lr=lr_perm_actor, eps=adam_eps
        )

        self.critic = SplitCritic(
            obs_dim,
            hidden_sizes=self._critic_hidden(),
            trans_hidden_sizes=critic_trans_hidden,
            perm_zero_init=bool(cfg.get("perm_zero_init", False)),
            trans_zero_init=bool(cfg.get("trans_zero_init", True)),
            perm_init_std=cfg.get("perm_init_std", None),
        ).to(device)
        self.trans_optim = torch.optim.Adam(
            self.critic.trans.parameters(), lr=lr_trans, eps=adam_eps
        )
        perm_opt_name = str(cfg.get("perm_optimizer", "adam")).lower()
        if perm_opt_name == "sgd":
            self.perm_optim = torch.optim.SGD(self.critic.perm.parameters(), lr=lr_perm)
        else:
            self.perm_optim = torch.optim.Adam(
                self.critic.perm.parameters(), lr=lr_perm, eps=adam_eps
            )

        self._base_lr_perm = self.perm_optim.param_groups[0]["lr"]
        self._base_lr_perm_actor = self.perm_actor_optim.param_groups[0]["lr"]
        # See _iter_indices: default False keeps every pre-existing result bit-identical.
        self.consolidation_shuffle = bool(cfg.get("consolidation_shuffle", False))
        self.rm_power = float(cfg.get("rm_power", cfg.get("alpha_p_rm_power", 0.6)))
        self.rm_power_actor = float(
            cfg.get("rm_power_actor", cfg.get("alpha_p_rm_power_actor", self.rm_power))
        )
        self._n_consolidations = 0
        self.last_alpha_p = self._base_lr_perm
        self.last_alpha_p_actor = self._base_lr_perm_actor

        capacity = int(cfg.get(
            "consolidation_buffer_size",
            self.k * int(cfg["n_steps"]) * self.num_envs,
        ))
        self.consolidation_buffer = StateConsolidationBuffer(capacity)
        self._updates_since_consolidation = 0
        self._pre_consolidation_snapshot = None

        self.last_absorbed_frac = None
        self.last_absorbed_align = None
        self.last_actor_absorbed_frac = None
        self.last_actor_absorbed_align = None
        self.last_consolidation_loss_curve = None
        self.actor_consolidation_loss_curve = None
        self.consolidation_loss_curves = []
        self.actor_consolidation_loss_curves = []
        self.consolidation_records = []
        self.last_perm_mean_before = None
        self.last_perm_mean_after = None
        self.last_perm_l2_before = None
        self.last_perm_l2_after = None
        self.last_trans_mean_before = None
        self.last_trans_mean_after = None
        self.last_trans_l2_before = None
        self.last_trans_l2_after = None
        self.last_actor_perm_mean_before = None
        self.last_actor_perm_mean_after = None
        self.last_actor_trans_mean_before = None
        self.last_actor_trans_mean_after = None
        self.last_delta_v = None
        self.last_delta_pi = None
        self._print_config_summary(hidden, actor_trans_hidden, critic_trans_hidden)

    def _print_config_summary(self, hidden, actor_trans_hidden, critic_trans_hidden):
        counts = {
            "perm_actor": sum(p.numel() for p in self.actor.perm_mean.parameters()),
            "trans_actor": sum(p.numel() for p in self.actor.trans_mean.parameters()),
            "perm_critic": sum(p.numel() for p in self.critic.perm.parameters()),
            "trans_critic": sum(p.numel() for p in self.critic.trans.parameters()),
        }
        print(
            "[PT-full] "
            f"rho={self.rho:g} kl_prior_coef={self.kl_prior_coef:g} k={self.k} "
            f"rm_power={self.rm_power:g} rm_power_actor={self.rm_power_actor:g} "
            f"lr_trans={self.actor_optim.param_groups[0]['lr']:g} "
            f"lr_perm={self.perm_optim.param_groups[0]['lr']:g} "
            f"lr_perm_actor={self.perm_actor_optim.param_groups[0]['lr']:g} "
            f"hidden={hidden} actor_trans_hidden={actor_trans_hidden} "
            f"critic_trans_hidden={critic_trans_hidden} params={counts}",
            flush=True,
        )

    def get_value(self, obs_batch_np):
        obs_t = torch.as_tensor(obs_batch_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_perm, v_trans = self.critic(obs_t)
        return v_perm.cpu().numpy(), v_trans.cpu().numpy()

    def critic_loss(self, batch, advantages, returns):
        v_perm, v_trans = self.critic(batch["obs"])
        return 0.5 * ((v_perm.detach() + v_trans - returns) ** 2).mean()

    def _online_step_update(self, obs_np, action_np, old_logprob_val,
                            reward, next_obs_np, done_flag):
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        old_lp_t = torch.tensor([old_logprob_val], dtype=torch.float32, device=self.device)
        vp, vt = self.get_value(obs_np[None, :])
        vp_next, vt_next = self.get_value(next_obs_np[None, :])
        value = float(vp[0] + vt[0])
        next_value = float(vp_next[0] + vt_next[0])
        target = reward + self.gamma * (1.0 - done_flag) * next_value
        advantage = target - value
        adv_t = torch.tensor([advantage], dtype=torch.float32, device=self.device)
        ret_t = torch.tensor([target], dtype=torch.float32, device=self.device)

        logprobs, entropy = self.actor.evaluate_actions(obs_t, action_t, detach_perm=True)
        ratio = torch.exp(logprobs - old_lp_t)
        actor_loss = -torch.min(
            ratio * adv_t,
            torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_t,
        ).mean()
        batch = {"obs": obs_t, "actions": action_t}
        critic_loss = self.critic_loss(batch, adv_t, ret_t)
        kl_loss = self.kl_prior_coef * self.actor.kl_to_prior(obs_t).mean()
        loss = actor_loss + critic_loss + kl_loss

        self.actor_optim.zero_grad()
        self._zero_critic_grads()
        self.perm_actor_optim.zero_grad()
        loss.backward()
        self._clip_grads()
        self.actor_optim.step()
        self._step_critic_optims()
        with torch.no_grad():
            approx_kl = ((ratio - 1) - ratio.log()).mean().item()
        return actor_loss.item(), critic_loss.item(), entropy.mean().item(), approx_kl, kl_loss.item()

    def update(self, last_obs, last_done, update_idx):
        v_perm_last, v_trans_last = self.get_value(last_obs)
        advantages, returns = self.buffer.compute_gae(
            v_perm_last + v_trans_last, last_done, self.gamma, self.gae_lambda
        )
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        batch = self.buffer.get_tensors()
        n = batch["obs"].shape[0]
        total_actor = total_critic = total_entropy = total_kl = 0.0
        total_clip = 0.0
        grad_norms = {"trans_actor": 0.0, "perm_actor": 0.0,
                  "trans_critic": 0.0, "perm_critic": 0.0}
        n_updates = 0
        approx_kl = 0.0

        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                mb_obs = batch["obs"][mb_idx]
                mb_actions = batch["actions"][mb_idx]
                mb_old_logprobs = batch["logprobs"][mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]
                if self.normalize_advantage and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_logprobs, entropy = self.actor.evaluate_actions(
                    mb_obs, mb_actions, detach_perm=True
                )
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                ) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.critic_loss(
                    {"obs": mb_obs, "actions": mb_actions}, mb_adv, mb_ret
                )
                kl_prior = self.actor.kl_to_prior(mb_obs).mean()
                kl_loss = self.kl_prior_coef * kl_prior
                loss = actor_loss + critic_loss + kl_loss

                self.actor_optim.zero_grad()
                self.perm_actor_optim.zero_grad()
                self._zero_critic_grads()
                loss.backward()
                grad_norms["trans_actor"] += self._grad_norm(self.actor.trans_mean.parameters())
                grad_norms["perm_actor"] += self._grad_norm(self.actor.perm_mean.parameters())
                grad_norms["trans_critic"] += self._grad_norm(self.critic.trans.parameters())
                grad_norms["perm_critic"] += self._grad_norm(self.critic.perm.parameters())
                self._clip_grads()
                self.actor_optim.step()
                self._step_critic_optims()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                total_actor += actor_loss.item()
                total_critic += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl_prior.item()
                total_clip += ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                n_updates += 1

            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                break

        with torch.no_grad():
            probe = batch["obs"][:min(1024, n)]
            v_perm, v_trans = self.critic(probe)
            mu_perm = self.actor.perm_mean(probe)
            mu_trans = self.actor.trans_mean(probe)
        self.post_update(update_idx)
        divisor = max(n_updates, 1)
        return {
            "actor_loss": total_actor / divisor,
            "critic_loss": total_critic / divisor,
            "kl_prior": total_kl / divisor,
            "entropy": total_entropy / divisor,
            "approx_kl": approx_kl,
            "clip_fraction": total_clip / divisor,
            "value_perm_l2": float(torch.linalg.vector_norm(v_perm)),
            "value_trans_l2": float(torch.linalg.vector_norm(v_trans)),
            "policy_perm_l2": float(torch.linalg.vector_norm(mu_perm)),
            "policy_trans_l2": float(torch.linalg.vector_norm(mu_trans)),
            "log_std_mean": float(self.actor.log_std.detach().mean()),
            "grad_norm_trans_actor": grad_norms["trans_actor"] / divisor,
            "grad_norm_perm_actor": grad_norms["perm_actor"] / divisor,
            "grad_norm_trans_critic": grad_norms["trans_critic"] / divisor,
            "grad_norm_perm_critic": grad_norms["perm_critic"] / divisor,
        }

    def post_update(self, update_idx):
        states = self.buffer.obs.reshape(-1, self.obs_dim)
        self.consolidation_buffer.add_batch(states)
        self._updates_since_consolidation += 1
        if self._updates_since_consolidation >= self.k:
            self._consolidate()

    def on_task_switch(self, step):
        self._consolidate()
        self.consolidation_buffer.clear()
        self._updates_since_consolidation = 0

    def _zero_critic_grads(self):
        self.trans_optim.zero_grad()
        self.perm_optim.zero_grad()

    def _step_critic_optims(self):
        self.trans_optim.step()

    def _critic_parameters(self):
        return list(self.critic.trans.parameters()) + list(self.critic.perm.parameters())

    def _iter_indices(self, n):
        """Minibatch indices for the consolidation regression.

        The consolidation buffer stores states in VISIT ORDER, so sequential minibatches are
        temporally correlated and the newest states always take the final gradient step of every
        epoch — the permanent ends up fit preferentially to the end of the buffer. PPO's own
        epoch loop in `update()` shuffles (np.random.permutation) for exactly this reason; the
        consolidation loop did not, which is an inconsistency inside one file.

        `consolidation_shuffle` defaults to FALSE so every result produced before this flag
        existed reproduces bit-for-bit. Set it true to draw the regression's minibatches i.i.d.
        from the buffer instead.
        """
        for _ in range(self.consolidation_epochs):
            order = (np.random.permutation(n) if self.consolidation_shuffle
                     else np.arange(n))
            for start in range(0, n, self.minibatch_size):
                yield order[start:min(start + self.minibatch_size, n)]

    @staticmethod
    def _stats(values):
        return float(values.mean()), float(torch.linalg.vector_norm(values))

    @staticmethod
    def _grad_norm(parameters):
        squared = [p.grad.detach().pow(2).sum() for p in parameters if p.grad is not None]
        if not squared:
            return 0.0
        return float(torch.sqrt(torch.stack(squared).sum()))

    @staticmethod
    def _absorption(before, after, transient, rho):
        wanted = rho * transient
        denom = float(torch.linalg.vector_norm(wanted))
        if denom <= 1e-12:
            return None, None
        moved = after - before
        wanted_sq = float((wanted * wanted).sum())
        return float(torch.linalg.vector_norm(moved) / denom), float((moved * wanted).sum() / wanted_sq)

    def _set_next_permanent_lrs(self):
        n = self._n_consolidations
        critic_lr = self._base_lr_perm / ((1.0 + n) ** self.rm_power) if self.rm_power > 0 else self._base_lr_perm
        actor_lr = self._base_lr_perm_actor / ((1.0 + n) ** self.rm_power_actor) if self.rm_power_actor > 0 else self._base_lr_perm_actor
        for group in self.perm_optim.param_groups:
            group["lr"] = critic_lr
        for group in self.perm_actor_optim.param_groups:
            group["lr"] = actor_lr

    @staticmethod
    def _flush_optimizer_params(optimizer, parameters):
        for parameter in parameters:
            optimizer.state.pop(parameter, None)

    def _consolidate(self):
        states_np = self.consolidation_buffer.as_array()
        if len(states_np) == 0:
            self.consolidation_buffer.clear()
            self._updates_since_consolidation = 0
            return None

        started = time.perf_counter()
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            old_v_perm, old_v_trans = self.critic(states)
            old_mu_perm = self.actor.perm_mean(states)
            old_mu_trans = self.actor.trans_mean(states)
        self._pre_consolidation_snapshot = (
            copy.deepcopy(self.actor).eval(), copy.deepcopy(self.critic).eval()
        )
        old_v_perm = old_v_perm.detach()
        old_v_trans = old_v_trans.detach()
        old_mu_perm = old_mu_perm.detach()
        old_mu_trans = old_mu_trans.detach()
        self.last_perm_mean_before, self.last_perm_l2_before = self._stats(old_v_perm)
        self.last_trans_mean_before, self.last_trans_l2_before = self._stats(old_v_trans)
        self.last_actor_perm_mean_before = float(old_mu_perm.mean())
        self.last_actor_trans_mean_before = float(old_mu_trans.mean())

        alpha_used = self.perm_optim.param_groups[0]["lr"]
        alpha_actor_used = self.perm_actor_optim.param_groups[0]["lr"]
        critic_curve = []
        actor_curve = []
        n = len(states_np)
        for indices in self._iter_indices(n):
            idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
            pred = self.critic.perm(states[idx]).squeeze(-1)
            target = old_v_perm[idx] + self.rho * old_v_trans[idx]
            loss = 0.5 * ((pred - target) ** 2).mean()
            self.perm_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.perm.parameters(), self.max_grad_norm)
            self.perm_optim.step()
            critic_curve.append(float(loss.detach()))

            pred_mu = self.actor.perm_mean(states[idx])
            target_mu = old_mu_perm[idx] + self.rho * old_mu_trans[idx]
            actor_loss = 0.5 * ((pred_mu - target_mu) ** 2).mean()
            self.perm_actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.perm_mean.parameters(), self.max_grad_norm)
            self.perm_actor_optim.step()
            actor_curve.append(float(actor_loss.detach()))

        with torch.no_grad():
            new_v_perm, _ = self.critic(states)
            new_mu_perm = self.actor.perm_mean(states)
        self.last_absorbed_frac, self.last_absorbed_align = self._absorption(
            old_v_perm, new_v_perm, old_v_trans, self.rho
        )
        self.last_actor_absorbed_frac, self.last_actor_absorbed_align = self._absorption(
            old_mu_perm, new_mu_perm, old_mu_trans, self.rho
        )
        self.last_perm_mean_after, self.last_perm_l2_after = self._stats(new_v_perm)
        self.last_actor_perm_mean_after = float(new_mu_perm.mean())

        # `decay_rho` defaults to rho, which is the preserving rho-split: the permanent absorbs
        # rho and the transient retains (1-rho), so the composed function is unchanged. Setting it
        # separately DECOUPLES the two halves of the mechanism, which no configuration has ever
        # done: rho controls how fast the permanent accumulates, decay_rho how hard the policy is
        # shrunk. Stage 9 showed the shrinkage is the only part that helps under discrete
        # switching, and Stage 15 showed it actively HURTS under monotone drift while the
        # permanent helps -- so "permanent on, shrinkage off" is the untested combination that
        # regime calls for.
        #
        # WARNING: decay_rho < rho is NOT composition-preserving. The permanent absorbs rho*V_T
        # while the transient keeps (1-decay_rho)*V_T, so V jumps by (rho-decay_rho)*V_T at every
        # consolidation. That is the legacy keep=1 overshoot; it is the point of the arm, but it
        # can amplify over many cycles. Watch for divergence.
        self.critic.decay_transient(1.0 - self.decay_rho, mode="output")
        self.actor.decay_transient(1.0 - self.decay_rho)
        self._flush_optimizer_params(self.trans_optim, self.critic.trans.parameters())
        self._flush_optimizer_params(self.actor_optim, self.actor.trans_mean.parameters())
        with torch.no_grad():
            _, new_v_trans = self.critic(states)
            new_mu_trans = self.actor.trans_mean(states)
        self.last_trans_mean_after, self.last_trans_l2_after = self._stats(new_v_trans)
        self.last_actor_trans_mean_after = float(new_mu_trans.mean())

        self._n_consolidations += 1
        self.last_alpha_p = alpha_used
        self.last_alpha_p_actor = alpha_actor_used
        self._set_next_permanent_lrs()
        self.last_consolidation_loss_curve = np.asarray(critic_curve, dtype=np.float32)
        self.actor_consolidation_loss_curve = np.asarray(actor_curve, dtype=np.float32)
        self.consolidation_loss_curves.append(self.last_consolidation_loss_curve.copy())
        self.actor_consolidation_loss_curves.append(self.actor_consolidation_loss_curve.copy())
        record = {
            "index": self._n_consolidations,
            "buffer_size": len(states_np),
            "alpha_p_used": float(alpha_used),
            "alpha_p_actor_used": float(alpha_actor_used),
            "absorbed_frac": self.last_absorbed_frac,
            "absorbed_align": self.last_absorbed_align,
            "actor_absorbed_frac": self.last_actor_absorbed_frac,
            "actor_absorbed_align": self.last_actor_absorbed_align,
            "perm_mean_before": self.last_perm_mean_before,
            "perm_mean_after": self.last_perm_mean_after,
            "perm_l2_before": self.last_perm_l2_before,
            "perm_l2_after": self.last_perm_l2_after,
            "trans_mean_before": self.last_trans_mean_before,
            "trans_mean_after": self.last_trans_mean_after,
            "trans_l2_before": self.last_trans_l2_before,
            "trans_l2_after": self.last_trans_l2_after,
            "actor_perm_mean_before": self.last_actor_perm_mean_before,
            "actor_perm_mean_after": self.last_actor_perm_mean_after,
            "actor_trans_mean_before": self.last_actor_trans_mean_before,
            "actor_trans_mean_after": self.last_actor_trans_mean_after,
            "duration_sec": time.perf_counter() - started,
        }
        self.consolidation_records.append(record)
        self.consolidation_buffer.clear()
        self._updates_since_consolidation = 0
        return record

    def measure_post_consolidation_drift(self, states_np):
        if self._pre_consolidation_snapshot is None:
            return None
        actor_before, critic_before = self._pre_consolidation_snapshot
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        actor_before = actor_before.to(self.device)
        critic_before = critic_before.to(self.device)
        with torch.no_grad():
            old_vp, old_vt = critic_before(states)
            new_vp, new_vt = self.critic(states)
            old_mu = actor_before.perm_mean(states) + actor_before.trans_mean(states)
            new_mu = self.actor.perm_mean(states) + self.actor.trans_mean(states)
            old_value = old_vp + old_vt
            new_value = new_vp + new_vt
            delta_v = float(
                torch.linalg.vector_norm(new_value - old_value)
                / torch.linalg.vector_norm(old_value).clamp_min(1e-8)
            )
            variance = torch.exp(self.actor.log_std.detach()) ** 2
            delta_pi = float(((old_mu - new_mu) ** 2 / (2.0 * variance)).sum(-1).mean())
        self.last_delta_v = delta_v
        self.last_delta_pi = delta_pi
        self._pre_consolidation_snapshot = None
        return {"delta_v": delta_v, "delta_pi": delta_pi}