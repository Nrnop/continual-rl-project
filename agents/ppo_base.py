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

from ..models.actor import GaussianActor, SplitActor
from ..utils.buffers import RolloutBuffer, StateBuffer


class PPOBase(ABC):
    """Abstract PPO agent with pluggable critic."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.device = device

        # --- actor (shared by both agents) ---
        hidden = list(cfg.get("hidden_sizes", [256, 256]))
        # `split_actor` puts the permanent/transient decomposition on the POLICY instead of (or
        # as well as) the critic. Default False, in which case every line below is exactly what
        # it was before this flag existed and no run is affected — asserted in
        # tests/test_split_actor.py::test_split_actor_off_is_bit_identical.
        self.split_actor = bool(cfg.get("split_actor", False))
        if self.split_actor:
            # Half-width by default so the two heads together match a single actor's parameter
            # count — the same convention `critic_hidden_sizes` applies on the critic side
            # (PT-0.5x, §6.1). Without it a "better" split actor might just be a bigger one.
            actor_hidden = list(cfg.get("actor_hidden_sizes", hidden))
            self.actor = SplitActor(obs_dim, act_dim, hidden_sizes=actor_hidden,
                                    trans_zero_init=bool(cfg.get("actor_trans_zero_init", True))
                                    ).to(device)
        else:
            self.actor = GaussianActor(obs_dim, act_dim, hidden_sizes=hidden).to(device)
        # CleanRL sets Adam eps=1e-5 for MuJoCo PPO; default keeps torch's 1e-8.
        # With a split actor this optimizer holds the TRANSIENT only: mu_perm is detached in the
        # training forward, so it takes no PPO gradient and moves during consolidation alone.
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=cfg["lr_actor"], eps=cfg.get("adam_eps", 1e-8)
        )
        # NOTE: actor consolidation is set up at the END of __init__ — it needs num_envs and
        # n_steps, which the rollout-buffer section below establishes.

        # --- rollout buffer (vectorized over num_envs) ---
        self.num_envs = int(cfg.get("num_envs", 1))
        self.buffer = RolloutBuffer(cfg["n_steps"], self.num_envs, obs_dim, act_dim, device)

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

        if self.split_actor:
            self._init_actor_consolidation(cfg)

    def _clip_grads(self):
        """Clip the actor and the critic SEPARATELY.

        This used to be one joint `clip_grad_norm_(actor_params + critic_params, max_grad_norm)`.
        `clip_grad_norm_` scales EVERY gradient by one factor derived from the TOTAL norm, so with
        a joint clip the critic's gradient magnitude changes the actor's effective step size. That
        silently breaks the premise the entire study rests on — "all three agents share an identical
        actor, only the critic differs" — because PT's critic (two [43,43] nets, a different loss
        surface) contributes a different norm than vanilla's single [64,64]. The actors were
        therefore NOT being trained identically across arms.

        Clipping the two groups independently makes the actor's update depend only on the actor's
        own gradient, so vanilla / pt / ewc actors really are trained the same way.

        `joint_grad_clip: true` restores the old behaviour for reproducing pre-2026-08-04 runs.
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
        """Hidden widths for the CRITIC, which may differ from the actor's.

        The paper's headline agent is `PT-DQN-0.5x`: theta_P and theta_T are each built at half the
        baseline's width so the TOTAL parameter count matches DQN's ("we use half the number of
        parameters as that of DQN for both permanent and transient value networks to ensure the
        total number of parameters across all baselines are same"). Appendix C.3 shows why this
        matters: once the agent's capacity is large relative to the environment, the baseline
        catches up and the decomposition confers no benefit.

        `critic_hidden_sizes` lets a config shrink the critic without touching the actor, so the
        actor stays byte-identical across vanilla / pt / ewc and the comparison stays attributable.
        """
        return list(self.cfg.get("critic_hidden_sizes",
                                 self.cfg.get("hidden_sizes", [256, 256])))

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def get_value(self, obs_batch_np):
        """Return (v_perm, v_trans) for a BATCH of observations.

        obs_batch_np has shape (num_envs, obs_dim); the returns are numpy arrays of
        shape (num_envs,). Vanilla puts the full value in v_perm and zeros in v_trans.
        """

    @abstractmethod
    def critic_loss(self, batch, advantages, returns):
        """Compute the critic loss tensor given a minibatch dict, advantages, and returns."""

    @abstractmethod
    def post_update(self, update_idx):
        """Called after each complete PPO update (all epochs). Used by PT for consolidation."""

    def on_task_switch(self, step):
        """Called when the training loop detects a task boundary. Override in PT.

        Subclasses that override this MUST call super() (PPOPT does), or a split actor will
        never consolidate at a boundary — which is the one moment the mechanism is supposed to
        matter most.
        """
        if self.split_actor:
            # Consolidate first, then decay: lock the finished task's behaviour into mu_perm
            # before dropping the transient, mirroring `on_switch: consolidate` on the critic.
            self._consolidate_actor()
            self._actor_updates_since_consolidation = 0

    # ------------------------------------------------------------------
    # Split actor: consolidation, decay, and the probe that decides the question
    # ------------------------------------------------------------------
    def _init_actor_consolidation(self, cfg):
        """Slow optimizer + rolling state buffer for mu_perm. Mirrors PPOPT.__init__."""
        lr_perm = cfg.get("lr_actor_perm", cfg.get("lr_perm", 2e-4))
        if str(cfg.get("actor_perm_optimizer", "sgd")).lower() == "sgd":
            self.actor_perm_optim = torch.optim.SGD(self.actor.perm_mean.parameters(), lr=lr_perm)
        else:
            self.actor_perm_optim = torch.optim.Adam(
                self.actor.perm_mean.parameters(), lr=lr_perm, eps=cfg.get("adam_eps", 1e-8))
        for g in self.actor_perm_optim.param_groups:
            g["_base_lr_perm"] = g["lr"]
        # Same cadence and decay as the critic unless overridden, so `pt_both` runs one clock.
        self.actor_k = int(cfg.get("actor_k", cfg.get("k", 60)))
        self.actor_decay = float(cfg.get("actor_decay", cfg.get("decay", 0.95)))
        self.actor_rm_power = float(cfg.get("actor_alpha_p_rm_power",
                                            cfg.get("alpha_p_rm_power", 0.0)))
        self.actor_consolidation_epochs = int(cfg.get("actor_consolidation_epochs", 1))
        self.actor_state_buffer = StateBuffer(
            int(cfg.get("actor_consolidation_buffer_size",
                        self.actor_k * cfg["n_steps"] * self.num_envs)))
        self._actor_updates_since_consolidation = 0
        self._n_actor_consolidations = 0
        # Diagnostics, refreshed each consolidation (None until the first).
        self.last_actor_absorbed_frac = None
        self.last_actor_perm_l2 = None
        self.last_actor_trans_l2_before = None
        self.last_actor_trans_l2_after = None
        # THE cancellation number. On the critic, corr(A_perm, A_trans) came out at ~-1.0 and
        # that is what made the decomposition invisible. If mu_perm and mu_trans cancel the same
        # way, the split actor fails for the same reason — and we want to know on run one, not
        # after two sweeps. See TRANSMISSION_RESULTS.md §4.
        self.last_actor_perm_trans_corr = None

    def _actor_post_update(self, update_idx):
        """Bank this rollout's states; consolidate mu_perm every actor_k updates."""
        if not self.split_actor:
            return
        self.actor_state_buffer.add_batch(self.buffer.obs.reshape(-1, self.obs_dim))
        self._actor_updates_since_consolidation += 1
        if self._actor_updates_since_consolidation >= self.actor_k:
            self._consolidate_actor()
            self._actor_updates_since_consolidation = 0

    def _consolidate_actor(self, decay=True):
        """mu_perm regresses onto mu_perm + mu_trans, then mu_trans is decayed.

        The policy transposition of Eq. (4) / Alg. 4 line 15: the permanent absorbs the FULL
        current function (keep = 1), not a shrunken copy of it. As on the critic, the transfer is
        deliberately not composition-preserving — right after it,
        mu = mu_perm_new + decay*mu_trans, an overshoot of decay*mu_trans that PPO corrects over
        the following updates.

        `decay=False` is used by the probe below, which needs to consolidate and measure the
        decay's effect as two separate, individually observable events.
        """
        if len(self.actor_state_buffer) == 0:
            return
        if self.actor_rm_power > 0.0:
            # Theorem 5's Robbins-Monro premise, same treatment as alpha_P on the critic.
            self._n_actor_consolidations += 1
            scale = 1.0 / ((1.0 + self._n_actor_consolidations) ** self.actor_rm_power)
            for g in self.actor_perm_optim.param_groups:
                g["lr"] = g["_base_lr_perm"] * scale

        probe = torch.as_tensor(
            self.actor_state_buffer.as_array()[:4096], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p_before, t_before = self.actor.means(probe)

        for _ in range(self.actor_consolidation_epochs):
            for s_mb in self.actor_state_buffer.iter_minibatches(self.minibatch_size, self.device):
                mu_p, mu_t = self.actor.means(s_mb)
                target = (mu_p + mu_t).detach()
                loss = 0.5 * ((mu_p - target) ** 2).mean()
                self.actor_perm_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.perm_mean.parameters(),
                                               self.max_grad_norm)
                self.actor_perm_optim.step()

        with torch.no_grad():
            p_after, _ = self.actor.means(probe)
            moved, wanted = p_after - p_before, t_before
            wn = float(wanted.pow(2).sum())
            self.last_actor_absorbed_frac = (
                float(moved.norm() / wanted.norm()) if wn > 1e-12 else None)
            self.last_actor_perm_l2 = float(torch.linalg.vector_norm(p_before))
            self.last_actor_trans_l2_before = float(torch.linalg.vector_norm(t_before))
            # Do mu_perm and mu_trans cancel, as V_perm and V_trans do?
            a, b = p_before.flatten(), t_before.flatten()
            a, b = a - a.mean(), b - b.mean()
            den = float(a.norm() * b.norm())
            self.last_actor_perm_trans_corr = (
                float((a * b).sum() / den) if den > 1e-12 else None)

        # Defect #9 cost this project its entire history of PT runs: alpha_P was inherited from
        # the paper's MinAtar setting, never tuned for HalfCheetah, and the permanent transferred
        # 0.04% of the transient while returns and losses both looked healthy. `lr_actor_perm` is
        # a brand-new hyper-parameter on a brand-new component, currently just copied from the
        # critic's. Assume it is wrong until the number says otherwise.
        if (self.last_actor_absorbed_frac is not None
                and self.last_actor_absorbed_frac < 0.01
                and not getattr(self, "_warned_inert_actor_perm", False)):
            self._warned_inert_actor_perm = True
            lr = self.actor_perm_optim.param_groups[0]["lr"]
            print("\n" + "!" * 78 +
                  f"\n[SPLIT ACTOR] INERT PERMANENT POLICY: absorbed only "
                  f"{self.last_actor_absorbed_frac * 100:.3f}% of mu_trans.\n"
                  f"[SPLIT ACTOR] mu_perm is not learning, so there is no slow timescale on the "
                  "policy and this arm\n[SPLIT ACTOR] is a plain actor plus a periodic decay. It "
                  "says nothing about the hypothesis.\n"
                  f"[SPLIT ACTOR] current: actor_perm_optimizer="
                  f"{self.cfg.get('actor_perm_optimizer')} lr_actor_perm={lr:g}\n"
                  "[SPLIT ACTOR] alpha_P had to be swept TWICE on the critic before it worked "
                  "(defect #9).\n" + "!" * 78 + "\n", flush=True)

        if decay:
            self.actor.decay_transient(self.actor_decay)
            with torch.no_grad():
                _, t_after = self.actor.means(probe)
                self.last_actor_trans_l2_after = float(torch.linalg.vector_norm(t_after))
        self.actor_state_buffer.clear()

    @torch.no_grad()
    def actor_decay_only(self):
        """Decay mu_trans without consolidating — the probe's intervention.

        Isolated so the training loop can measure return, apply ONLY this, and measure again with
        no gradient step in between. On a split CRITIC that difference is provably zero: changing
        V changes no action. On a split actor it is the whole mechanism, and the size of it is
        the cleanest evidence the decomposition reaches behaviour at all.
        """
        if self.split_actor:
            self.actor.decay_transient(self.actor_decay)

    def diagnostics(self):
        """Per-update diagnostic scalars, sampled at the TOP of `update()`.

        Sampled before any gradient step so the values describe the agent that actually
        *collected* this rollout, not the one that exists after training on it. Base returns
        nothing; PPOPT adds the consolidation-cycle phase (see its override).
        """
        return {}

    def _extra_loss(self):
        """Hook for additional per-step loss terms (e.g. EWC penalty). Override in subclass."""
        return torch.tensor(0.0, device=self.device)

    def _all_optimizers(self):
        """Every torch optimizer held by this agent (actor + any critic optims).

        Discovered by scanning instance attributes so subclasses (vanilla critic,
        split perm/trans critics, EWC) are all covered without extra plumbing.
        """
        return [v for v in self.__dict__.values()
                if isinstance(v, torch.optim.Optimizer)]

    def anneal_lr(self, frac):
        """Scale every optimizer's LR to `frac` of its initial value (CleanRL-style).

        `frac` goes 1.0 -> 0.0 over training. The base LR is captured lazily on
        first call so this is a no-op-safe wrapper around each param group.
        """
        for opt in self._all_optimizers():
            for pg in opt.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = frac * pg["initial_lr"]

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
    def collect_rollout(self, envs, obs, done):
        """Fill the RolloutBuffer for n_steps from a vectorized env.

        Args:
            envs: a gymnasium VectorEnv (num_envs sub-envs, auto-resetting).
            obs:  current observations, shape (num_envs, obs_dim).
            done: current done flags, shape (num_envs,).

        Returns (last_obs, last_done, episode_returns), where episode_returns is a
        list of TRUE (un-normalized) episodic returns for every episode that
        finished during this rollout — read from RecordEpisodeStatistics, which
        wraps the env BEFORE reward normalization.

        Side effect: self._velocities (list[float]) — mean x_velocity per step.
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

            # Store obs with ITS done flag (dones[t] marks a post-reset obs) and
            # the reward for the action just taken (CleanRL convention).
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

    # ------------------------------------------------------------------
    # Per-step online update (for step_by_step mode)
    # ------------------------------------------------------------------
    def _online_step_update(self, obs_np, action_np, old_logprob_val,
                            reward, next_obs_np, done_flag):
        """One-step online actor-critic update with PPO clipping.

        Computes a 1-step TD advantage δ = r + γ(1−d)V(s') − V(s) and runs
        a single gradient step on actor + critic.  The PPO clip safeguard is
        retained, though the ratio will be ≈1 for freshly-sampled actions.

        Returns (actor_loss, critic_loss, entropy, approx_kl, extra_loss).
        """
        obs_t = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(
            action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        old_lp_t = torch.tensor(
            [old_logprob_val], dtype=torch.float32, device=self.device)

        # 1-step TD target and advantage (no grad — stability).
        # get_value expects a batch (num_envs, obs_dim); wrap the single obs as a 1-row batch.
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

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self, last_obs, last_done, update_idx):
        """Run the full PPO update (GAE + multiple epochs of minibatch SGD).

        Returns a dict of scalar metrics.
        """
        # Diagnostics describing the agent that COLLECTED this rollout, sampled before any
        # gradient step displaces it (see `diagnostics`).
        diag = self.diagnostics()

        # Bootstrap value for GAE
        v_perm_last, v_trans_last = self.get_value(last_obs)
        last_value = v_perm_last + v_trans_last
        advantages, returns = self.buffer.compute_gae(last_value, last_done,
                                                      self.gamma, self.gae_lambda)

        # ---- CRITIC-QUALITY DIAGNOSTICS (transmission hypothesis, D1) ----
        # The question these answer: is PT's critic actually WORSE than vanilla's, or is it
        # equally good / better while the behaviour is worse? In DQN the value function *is* the
        # policy, so a better critic is better behaviour by construction. In an actor-critic the
        # critic reaches behaviour only through the advantage, so the two can come apart — and if
        # they do, the deficit is a transmission problem, not a value-learning problem.
        #
        # These are computed on V AT COLLECTION TIME (buffer.v_perm + buffer.v_trans), i.e. the
        # value function as it was actually used to act and to bootstrap — not the post-update one.
        # `compute_gae` sets returns = advantages + values, so returns - values IS advantages:
        # the residual below is exact, not an approximation.
        #
        # `explained_var` is the scale-free comparator and the one to trust across arms: reward
        # normalisation and differing trajectories mean the raw RMSE lives on different scales for
        # different agents. 1.0 = the critic explains the returns perfectly, 0.0 = no better than
        # predicting the mean, negative = worse than the mean.
        #
        # No RNG is consumed here, so adding these leaves every run seed-identical to before.
        adv_np = np.asarray(advantages, dtype=np.float64)
        ret_np = np.asarray(returns, dtype=np.float64)
        ret_var = float(ret_np.var())
        diag["diag/value_rmse"] = float(np.sqrt((adv_np ** 2).mean()))
        diag["diag/explained_var"] = (
            float(1.0 - adv_np.var() / ret_var) if ret_var > 1e-12 else float("nan"))
        diag["diag/adv_abs_mean"] = float(np.abs(adv_np).mean())
        diag["diag/adv_std"] = float(adv_np.std())
        diag["diag/return_std"] = float(np.sqrt(ret_var))
        diag["diag/value_mean"] = float((ret_np - adv_np).mean())

        # ---- HOW MUCH OF THE POLICY'S UPDATE IS THE CRITIC? (D3) ----
        # The critic reaches behaviour only through the advantage, so the advantage IS the whole
        # channel: whatever the critic knows that does not appear here cannot change any action.
        # `compute_gae_components` splits A exactly into reward / permanent / transient parts.
        #
        # Attribution is by COVARIANCE SHARE, not variance share:
        #     Var(A) = sum_c Cov(A_c, A)     ->     share_c = Cov(A_c, A) / Var(A)
        # The three components are correlated, so variance shares would not sum to 1; covariance
        # shares do, exactly, which makes "the permanent supplies X% of the policy's update
        # signal" a statement that means something.
        #
        # Advantage normalisation is affine and applied per minibatch, so it rescales every
        # component by the same factor and leaves these shares unchanged. It does, however, delete
        # any constant offset — one of two places the permanent's influence is structurally
        # attenuated (the other is that A_perm carries a TEMPORAL DIFFERENCE of V_perm, not its
        # level; see compute_gae_components).
        adv_r, adv_p, adv_t = self.buffer.compute_gae_components(
            v_perm_last, v_trans_last, last_done, self.gamma, self.gae_lambda)
        a_var = adv_np.var()
        if a_var > 1e-12:
            for name, comp in (("reward", adv_r), ("perm", adv_p), ("trans", adv_t)):
                c = np.asarray(comp, dtype=np.float64)
                diag[f"diag/adv_share_{name}"] = float(
                    ((c - c.mean()) * (adv_np - adv_np.mean())).mean() / a_var)
            # Correlation between the update the actor actually gets and the one it would get with
            # a given component removed. 1.0 = that component changes nothing about the direction
            # of the policy update; this is the scale-free version of the same question.
            for name, alt in (("nocritic", adv_r),
                              ("noperm", adv_np - np.asarray(adv_p, dtype=np.float64)),
                              ("notrans", adv_np - np.asarray(adv_t, dtype=np.float64))):
                b = np.asarray(alt, dtype=np.float64)
                sd = adv_np.std() * b.std()
                diag[f"diag/adv_corr_{name}"] = (
                    float(((adv_np - adv_np.mean()) * (b - b.mean())).mean() / sd)
                    if sd > 1e-12 else float("nan"))

        # ---- CAUSAL ABLATION: what the ACTOR is allowed to see ----
        # `actor_advantage_source` removes a component from the advantage the actor is trained on
        # while leaving `returns` — and therefore the critic's own training target — untouched.
        # That separates "what the critic learns" from "what the policy sees", which is exactly
        # the transmission question, and it is a causal test rather than a correlational one:
        #   full        A                (default; identical to before this knob existed)
        #   trans_only  A - A_perm       the permanent cannot influence behaviour at all
        #   perm_only   A - A_trans      only the slow component may influence behaviour
        #   none        A_reward         no critic influence whatsoever (baseline-free)
        # If `trans_only` matches `full`, the permanent's influence on decision-making is zero as
        # measured, not merely small.
        src = str(self.cfg.get("actor_advantage_source", "full")).lower()
        if src != "full":
            if src == "trans_only":
                advantages = adv_np - np.asarray(adv_p, dtype=np.float64)
            elif src == "perm_only":
                advantages = adv_np - np.asarray(adv_t, dtype=np.float64)
            elif src == "none":
                advantages = np.asarray(adv_r, dtype=np.float64)
            else:
                raise ValueError(f"unknown actor_advantage_source: {src!r}")
            advantages = advantages.astype(np.float32)

        # Convert to tensors
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

            # Optional early stop on KL
            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                break

        self.post_update(update_idx)
        self._actor_post_update(update_idx)      # no-op unless split_actor

        for attr, tag in (("last_actor_absorbed_frac", "actor_absorbed_frac"),
                          ("last_actor_perm_trans_corr", "actor_perm_trans_corr"),
                          ("last_actor_perm_l2", "actor_perm_l2"),
                          ("last_actor_trans_l2_before", "actor_trans_l2_before"),
                          ("last_actor_trans_l2_after", "actor_trans_l2_after")):
            val = getattr(self, attr, None)
            if val is not None:
                diag[f"diag/{tag}"] = val

        out = {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": approx_kl,
        }
        out.update(diag)
        return out

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
