"""Shared PPO core: rollout collection, GAE and the clipped-surrogate update.

Subclasses supply the critic through these hooks:
    get_value(obs)    -> (v_perm, v_trans)   vanilla returns (V, 0)
    critic_loss(...)  -> scalar loss         added to the actor loss
    post_update(idx)  -> None                PT uses it for the consolidation cadence
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

        # Actor, identical across agents
        hidden = list(cfg.get("hidden_sizes", [256, 256]))
        self.actor = GaussianActor(
            obs_dim, act_dim, hidden_sizes=hidden,
            # Read from the same key as SplitGaussianActor, so every arm starts at the same sigma.
            log_std_init=cfg.get("log_std_init", 0.0),
            # True matches pt's frozen log_std, so the arms do not differ in exploration.
            freeze_log_std=bool(cfg.get("freeze_log_std", False)),
            # One key for every arm (also read in ppo_pt), so all arms share the architecture.
            layer_norm=bool(cfg.get("layer_norm", False)),
        ).to(device)
        # CleanRL sets Adam eps=1e-5 for MuJoCo PPO; default keeps torch's 1e-8.
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=cfg["lr_actor"], eps=cfg.get("adam_eps", 1e-8)
        )

        # Rollout buffer, vectorized over num_envs
        self.num_envs = int(cfg.get("num_envs", 1))
        self.buffer = RolloutBuffer(cfg["n_steps"], self.num_envs, obs_dim, act_dim, device)

        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.clip_coef = cfg["clip_coef"]
        self.epochs = cfg["epochs"]
        self.minibatch_size = cfg["minibatch_size"]
        self.ent_coef = cfg.get("ent_coef", 0.0)
        # KL-to-zero-policy coefficient, see update(). 0.0 disables it.
        self.mu_l2_coef = float(cfg.get("mu_l2_coef", 0.0))
        self.max_grad_norm = cfg["max_grad_norm"]
        self.target_kl = cfg.get("target_kl", None)
        self.normalize_advantage = cfg.get("normalize_advantage", True)

    def _clip_grads(self):
        """Clip the actor and critic gradients separately.

        A joint clip scales every gradient by one factor from the total norm, so the critic's
        gradient (which differs between arms) would change the actor's step size. Separate clips
        keep the actor update identical across arms. `joint_grad_clip: true` clips jointly.
        """
        if self.cfg.get("joint_grad_clip", False):
            all_params = list(self.actor.parameters()) + list(self._critic_parameters())
            torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            return
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_params = list(self._critic_parameters())
        if critic_params:
            torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)

    def _critic_hidden(self):
        """Return the critic's hidden widths (`critic_hidden_sizes`, else `hidden_sizes`).

        Lets a config size the critic separately from the actor, e.g. for parameter matching as in
        the paper's PT-DQN-0.5x, while the actor stays identical across arms.
        """
        return list(self.cfg.get("critic_hidden_sizes",
                                 self.cfg.get("hidden_sizes", [256, 256])))

    @abstractmethod
    def get_value(self, obs_batch_np):
        """Return (v_perm, v_trans) for a batch of observations.

        obs_batch_np: (num_envs, obs_dim) -> two numpy arrays of shape (num_envs,). Vanilla puts the
        full value in v_perm and zeros in v_trans.
        """

    @abstractmethod
    def critic_loss(self, batch, advantages, returns):
        """Return the critic loss tensor for a minibatch dict, advantages and returns."""

    @abstractmethod
    def post_update(self, update_idx):
        """Called after each complete PPO update (all epochs). PT uses it for consolidation."""

    def on_task_switch(self, step):
        """Called when the training loop detects a task boundary."""

    def _extra_loss(self):
        """Extra loss term for the per-step update (e.g. the EWC penalty)."""
        return torch.tensor(0.0, device=self.device)

    def _all_optimizers(self):
        """Return every torch optimizer held as an attribute (actor and all critic optimizers)."""
        return [v for v in self.__dict__.values()
                if isinstance(v, torch.optim.Optimizer)]

    def anneal_lr(self, frac):
        """Set every optimizer's LR to `frac` of its initial value (CleanRL-style).

        `frac` goes 1.0 -> 0.0 over training. The initial LR is stored on the first call.
        """
        for opt in self._all_optimizers():
            for pg in opt.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = frac * pg["initial_lr"]

    def state_dict(self):
        """Return the network state dicts for a checkpoint."""
        state = {"actor": self.actor.state_dict()}
        if hasattr(self, "critic"):
            state["critic"] = self.critic.state_dict()
        return state

    def load_state_dict(self, state):
        """Load network weights from a checkpoint dict."""
        if "actor" in state and hasattr(self, "actor"):
            self.actor.load_state_dict(state["actor"])
        if "critic" in state and hasattr(self, "critic"):
            self.critic.load_state_dict(state["critic"])

    def collect_rollout(self, envs, obs, done):
        """Fill the RolloutBuffer for n_steps from a vectorized env.

        Args:
            envs: a gymnasium VectorEnv (num_envs sub-envs, auto-resetting).
            obs:  current observations, shape (num_envs, obs_dim).
            done: current done flags, shape (num_envs,).

        Returns (last_obs, last_done, episode_returns). episode_returns lists the true
        (un-normalized) return of every episode finished in this rollout, read from
        RecordEpisodeStatistics, which wraps the env before reward normalization.
        Also sets self._velocities: mean x_velocity per step.
        """
        self.buffer.reset()
        episode_returns = []
        velocities = []

        for _ in range(self.cfg["n_steps"]):
            v_perm, v_trans = self.get_value(obs)                 # (num_envs,)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, logprob = self.actor.act(obs_t)
            action_np = action.cpu().numpy()                      # (num_envs, act_dim)
            logprob_np = logprob.cpu().numpy()                    # (num_envs,)

            next_obs, reward, terminated, truncated, infos = envs.step(action_np)
            next_done = np.logical_or(terminated, truncated).astype(np.float32)

            # Store obs with its own done flag (dones[t] marks a post-reset obs) and the reward
            # for the action just taken (CleanRL convention).
            self.buffer.add(obs, action_np, logprob_np, reward, done, v_perm, v_trans)

            xv = infos.get("x_velocity")
            if xv is not None:
                velocities.append(float(np.mean(xv)))

            # True episodic returns for completed episodes this step.
            if "episode" in infos:
                mask = np.asarray(infos["_episode"], dtype=bool)
                episode_returns.extend(np.asarray(infos["episode"]["r"])[mask].tolist())

            obs = next_obs
            done = next_done

        self._velocities = velocities
        self._step_metrics = None
        return obs, done, episode_returns

    # Per-step online update (step_by_step mode)
    def _online_step_update(self, obs_np, action_np, old_logprob_val,
                            reward, next_obs_np, done_flag):
        """One-step online actor-critic update with PPO clipping.

        Uses the 1-step TD advantage r + gamma*(1 - d)*V(s') - V(s) and one gradient step on actor
        and critic. The ratio is about 1 for freshly sampled actions.

        Returns (actor_loss, critic_loss, entropy, approx_kl, extra_loss).
        """
        obs_t = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(
            action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        old_lp_t = torch.tensor(
            [old_logprob_val], dtype=torch.float32, device=self.device)

        # 1-step TD target and advantage, without gradient. get_value expects a batch.
        vp, vt = self.get_value(obs_np[None, :])
        vp_next, vt_next = self.get_value(next_obs_np[None, :])
        v_perm, v_trans = float(vp[0]), float(vt[0])
        v_curr = v_perm + v_trans
        v_next = float(vp_next[0]) + float(vt_next[0])
        target_val = reward + self.gamma * (1.0 - done_flag) * v_next
        adv_val = target_val - v_curr

        adv_t = torch.tensor([adv_val], dtype=torch.float32, device=self.device)
        ret_t = torch.tensor([target_val], dtype=torch.float32, device=self.device)

        # Actor loss (clipped surrogate)
        new_logprobs, entropy = self.actor.evaluate_actions(obs_t, action_t)
        ratio = torch.exp(new_logprobs - old_lp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_coef,
                            1.0 + self.clip_coef) * adv_t
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()

        # Critic loss (delegated to subclass)
        batch = {"obs": obs_t, "actions": action_t,
                 "v_perm": torch.tensor([v_perm], device=self.device),
                 "v_trans": torch.tensor([v_trans], device=self.device)}
        c_loss = self.critic_loss(batch, adv_t, ret_t)

        # Extra loss hook (e.g. EWC penalty)
        extra = self._extra_loss()

        loss = actor_loss + c_loss + self.ent_coef * entropy_loss + extra

        self.actor_optim.zero_grad()
        self._zero_critic_grads()
        loss.backward()
        self._clip_grads()
        self.actor_optim.step()
        self._step_critic_optims()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - ratio.log()).mean().item()

        return (actor_loss.item(), c_loss.item(), entropy.mean().item(),
                approx_kl, extra.item())

    def update(self, last_obs, last_done, update_idx):
        """Run the PPO update (GAE, then several epochs of minibatch SGD). Return scalar metrics."""
        v_perm_last, v_trans_last = self.get_value(last_obs)
        last_value = v_perm_last + v_trans_last
        advantages, returns = self.buffer.compute_gae(last_value, last_done,
                                                      self.gamma, self.gae_lambda)

        adv_split = self._adv_split_metrics(v_perm_last, v_trans_last, last_done, advantages)

        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        batch = self.buffer.get_tensors()
        n = batch["obs"].shape[0]          # flattened batch = n_steps * num_envs
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
                # mu_l2_coef > 0 adds KL(pi || N(0, sigma^2)): pt's KL-to-permanent with the
                # permanent replaced by the zero policy.
                if self.mu_l2_coef > 0.0 and hasattr(self.actor, "kl_to_zero_prior"):
                    loss = loss + self.mu_l2_coef * self.actor.kl_to_zero_prior(mb_obs).mean()

                self.actor_optim.zero_grad()
                self._zero_critic_grads()
                loss.backward()
                self._clip_grads()
                self.actor_optim.step()
                self._step_critic_optims()

                total_actor_loss += actor_loss.item()
                total_critic_loss += c_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

            # Optional early stop on KL.
            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                break

        self.post_update(update_idx)

        log_std = self.actor.log_std.detach()
        out = {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": approx_kl,
            # Sigma for every arm. The min shows a single action dimension collapsing, which the
            # mean hides.
            "log_std_mean": float(log_std.mean()),
            "log_std_min": float(log_std.min()),
        }
        if adv_split is not None:
            out.update(adv_split)
        return out

    def _adv_split_metrics(self, v_perm_last, v_trans_last, last_done, advantages):
        """Split the advantage into reward, permanent-value and transient-value parts (Table 4.1).

        Enabled by `log_adv_split: true`; returns None when off. Read-only, so training is
        unchanged. A helper because PPOPT overrides update() and calls it too.
        """
        if not self.cfg.get("log_adv_split", False):
            return None
        a_r, a_p, a_t = self.buffer.advantage_split(v_perm_last, v_trans_last, last_done,
                                                    self.gamma, self.gae_lambda)
        var_a = float(np.var(advantages))
        sd_p, sd_t = float(np.std(a_p)), float(np.std(a_t))
        return {
            "adv_var_perm_ratio": float(np.var(a_p)) / max(var_a, 1e-12),
            "adv_var_trans_ratio": float(np.var(a_t)) / max(var_a, 1e-12),
            "adv_var_reward_ratio": float(np.var(a_r)) / max(var_a, 1e-12),
            "adv_corr_perm_trans": (float(np.corrcoef(a_p, a_t)[0, 1])
                                    if sd_p > 1e-12 and sd_t > 1e-12 else float("nan")),
            "adv_split_residual": float(np.max(np.abs(a_r + a_p + a_t - advantages))),
        }

    # Critic optimizer hooks, implemented by each subclass
    @abstractmethod
    def _zero_critic_grads(self):
        """Zero-grad all critic optimizers."""

    @abstractmethod
    def _step_critic_optims(self):
        """Step all critic optimizers."""

    @abstractmethod
    def _critic_parameters(self):
        """Return an iterable of all critic parameters (for grad clipping)."""
