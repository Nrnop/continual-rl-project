"""Dual-timescale Permanent-Transient PPO agent (value-function decomposition).

PT is applied to the CRITIC only (matching Anand & Precup): V(s) = V_perm(s) + V_trans(s).
The policy is the same single GaussianActor as the vanilla/EWC baselines, so the comparison
stays apples-to-apples and any difference is attributable to the split critic.

- V_trans (θ_T): fast head, trained every PPO update to predict the residual
                 (returns - V_perm.detach()) above the frozen permanent baseline.
- V_perm  (θ_P): slow head, NOT trained on returns each step. Every k updates it *consolidates*
                 by absorbing the transient (see _consolidate), then θ_T is decayed.

Consolidation is value-preserving for any decay: θ_P regresses to old_θ_P + (1-decay)·V_trans,
so after θ_T ← decay·θ_T the acting value V = V_perm + V_trans is unchanged (no drift). At a task
boundary we consolidate first (locking the just-learned task value into θ_P) then let θ_T re-adapt.
"""
import numpy as np
import torch

from .ppo_base import PPOBase
from ..models.critic import SplitCritic, SharedTrunkSplitCritic
from ..utils.buffers import ConsolidationBuffer


class PPOPT(PPOBase):
    """PPO with a dual-timescale split critic (Permanent + Transient)."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        hidden = list(cfg.get("hidden_sizes", [256, 256]))
        adam_eps = cfg.get("adam_eps", 1e-8)
        lr_trans = cfg.get("lr_trans", cfg["lr_actor"])

        # "separate" (default) = two independent MLP trunks, consolidation by regression (lossy).
        # "shared_trunk"       = one trunk + two linear heads, consolidation by exact weight arithmetic.
        self.critic_arch = str(cfg.get("critic_arch", "separate")).lower()
        self.shared_trunk = self.critic_arch == "shared_trunk"

        if self.shared_trunk:
            self.critic = SharedTrunkSplitCritic(obs_dim, hidden_sizes=hidden).to(device)
            # The shared trunk is learned together with the fast transient head; the permanent head
            # receives NO gradients (v_perm is detached in critic_loss) and moves only at
            # consolidation, via exact weight arithmetic. So lr_perm is unused in this variant.
            self.trans_optim = torch.optim.Adam(
                list(self.critic.trunk.parameters()) + list(self.critic.trans.parameters()),
                lr=lr_trans, eps=adam_eps,
            )
            self.perm_optim = None
        else:
            self.critic = SplitCritic(obs_dim, hidden_sizes=hidden).to(device)
            # Fast optimizer for the transient head (task-specific)
            self.trans_optim = torch.optim.Adam(
                self.critic.trans.parameters(), lr=lr_trans, eps=adam_eps
            )
            # Slow optimizer for the permanent head (task-invariant)
            lr_perm = cfg.get("lr_perm", 3e-5)
            perm_opt_name = cfg.get("perm_optimizer", "adam").lower()
            if perm_opt_name == "sgd":
                self.perm_optim = torch.optim.SGD(self.critic.perm.parameters(), lr=lr_perm)
            else:
                self.perm_optim = torch.optim.Adam(
                    self.critic.perm.parameters(), lr=lr_perm, eps=adam_eps
                )

        # NOTE: PT is critic-only. The actor is the single GaussianActor built by
        # PPOBase.__init__ (lr = lr_actor), identical to vanilla/EWC. We deliberately do NOT
        # split the policy: the paper decomposes the value function, and a split actor with no
        # consolidation just resets the policy at each boundary. Keeping one actor makes the
        # vanilla/PT/EWC comparison clean (only the critic differs).

        # Decay factor for the transient critic head at consolidation / task boundaries.
        self.transient_decay = cfg.get("decay", 0.0)
        # Scaling theta_T only touches the PARAMETERS: Adam's exp_avg / exp_avg_sq for those same
        # parameters survive untouched, so the next step displaces the freshly-decayed weights using
        # momentum from a network that no longer exists — undoing the consistency consolidation just
        # established. Enable this to clear that state alongside the decay. Default False keeps the
        # original behaviour so earlier runs stay reproducible.
        self.reset_trans_optim_on_decay = bool(cfg.get("reset_trans_optim_on_decay", False))

        # --- Permanent consolidation: the actual PT mechanism (was never wired up) ---
        # theta_P is NOT trained on returns every step. Every k updates it absorbs the
        # transient by regressing V_perm(s) -> old_V_perm(s) + V_trans(s).detach() over a
        # rolling buffer of visited states, then the transient head is decayed. This mirrors
        # train_P_Net in the reference PT_DQN and keeps the permanent critic a slow, stable
        # task-invariant baseline instead of a second full-return regressor.
        self.k = int(cfg.get("k", 10))
        self.consolidation_epochs = int(cfg.get("consolidation_epochs", 1))
        # One consolidation cycle banks k updates x (n_steps * num_envs) states.
        buf_cap = int(cfg.get("consolidation_buffer_size",
                              self.k * cfg["n_steps"] * self.num_envs))
        self.consolidation_buffer = ConsolidationBuffer(buf_cap)
        self._updates_since_consolidation = 0
        # % change in the acting value across the most recent consolidation (None until the first
        # one runs). 0 means the transfer was value-preserving; large values mean it was not.
        self.last_consolidation_error = None
        # Same measurement on buffered states EXCLUDED from the regression (needs
        # consolidation_holdout_frac > 0). This is the number that matters operationally.
        self.last_consolidation_error_holdout = None
        # Consolidation-regression diagnostics, refreshed at each consolidation (None until the
        # first one). `_loss_curve` is the full within-consolidation loss trace; the scalars are
        # what gets logged each cycle.
        self.last_consolidation_loss_first = None
        self.last_consolidation_loss_last = None
        self.last_consolidation_loss_mean = None
        self.last_consolidation_loss_curve = None
        # Permanent value statistics over the consolidation batch, before and after the
        # consolidation regression. The transient pair below shows what the decay REMOVES; this
        # pair shows what the transfer actually ADDS — together they are the two-defect picture.
        self.last_perm_mean_before = None
        self.last_perm_mean_after = None
        self.last_perm_l2_before = None
        self.last_perm_l2_after = None
        # Transient value statistics over the consolidation batch, before and after the decay.
        self.last_trans_mean_before = None
        self.last_trans_mean_after = None
        self.last_trans_l2_before = None
        self.last_trans_l2_after = None

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------
    def get_value(self, obs_batch_np):
        """Return (V_perm, V_trans) for a batch: (num_envs, obs_dim) -> arrays (num_envs,)."""
        obs_t = torch.as_tensor(obs_batch_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_perm, v_trans = self.critic(obs_t)
        return v_perm.cpu().numpy(), v_trans.cpu().numpy()

    def critic_loss(self, batch, advantages, returns):
        """Transient-only regression during normal PPO updates.

        theta_T (fast): MSE(V_perm.detach() + V_trans, returns) — the transient head
        learns the residual above the *frozen* permanent baseline.

        theta_P is deliberately NOT trained here. Regressing V_perm directly on returns
        every step (the old behaviour) double-counts against the acting value
        V = V_perm + V_trans and, with no target network / no normalization, drove the
        value divergence we saw (critic_loss ~2e5). The permanent critic is updated only
        during consolidation (see post_update / _consolidate).
        """
        v_perm, v_trans = self.critic(batch["obs"])
        v_combined = v_perm.detach() + v_trans
        return 0.5 * ((v_combined - returns) ** 2).mean()

    def post_update(self, update_idx):
        """After each PPO update: bank this rollout's states, consolidate every k updates."""
        # The shared-trunk variant consolidates by weight arithmetic, so it needs no state buffer.
        if not self.shared_trunk:
            # Snapshot visited states with the CURRENT permanent value. theta_P is frozen
            # between consolidations, so this is old_V_perm at visit time. The rollout
            # buffer is (n_steps, num_envs, obs_dim); flatten to a batch of states.
            states = self.buffer.obs.reshape(-1, self.obs_dim)
            with torch.no_grad():
                s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                old_v_perm, _ = self.critic(s_t)
            self.consolidation_buffer.add_batch(states, old_v_perm.cpu().numpy())

        self._updates_since_consolidation += 1
        if self._updates_since_consolidation >= self.k:
            self._consolidate()
            self._updates_since_consolidation = 0

    def _consolidate(self):
        """Absorb the transient value into the permanent critic, then decay the transient.

        For each stored state s: regress V_perm(s) -> old_V_perm(s) + (1-decay)*V_trans(s).detach().
        Decaying V_trans by `decay` afterwards then leaves the acting value EXACTLY preserved for
        any decay:  V_new = P_new + decay*T = old_P + (1-decay)*T + decay*T = old_P + T = V_old.
        (The earlier target old_P + T with a >0 decay double-counted the transient and inflated V
        by decay*T every cycle.)  decay=0 => hard reset, P absorbs all of T.
        """
        # Shared-trunk variant: exact, O(1), no regression and no value drift.
        if self.shared_trunk:
            # No regression here — the transfer is weight arithmetic — so there is no loss curve.
            # The transient statistics are still meaningful and are recorded.
            probe = None
            if len(self.consolidation_buffer):
                st, _ = self.consolidation_buffer.as_arrays()
                probe = torch.as_tensor(st[: min(4096, len(st))], dtype=torch.float32,
                                        device=self.device)
            self.last_trans_mean_before, self.last_trans_l2_before = self._trans_stats(probe)
            self.last_perm_mean_before, self.last_perm_l2_before = self._perm_stats(probe)
            self.critic.consolidate(self.transient_decay)
            if self.reset_trans_optim_on_decay:
                self._decay_transient(1.0)               # weights unchanged; clears optimiser state
            self.last_trans_mean_after, self.last_trans_l2_after = self._trans_stats(probe)
            self.last_perm_mean_after, self.last_perm_l2_after = self._perm_stats(probe)
            self.last_consolidation_loss_first = None    # not applicable: no regression
            self.last_consolidation_loss_last = None
            self.last_consolidation_loss_mean = None
            self.last_consolidation_loss_curve = None
            self.last_consolidation_error = 0.0          # exact by construction
            self.last_consolidation_error_holdout = 0.0  # ...on every state, not just fitted ones
            return

        if len(self.consolidation_buffer) == 0:
            return

        # Measure how much the acting value V = V_perm + V_trans actually moves across this
        # consolidation. Ideally 0 (the transfer is meant to be value-preserving); in practice it is
        # bounded by how well the regression below fits. Reported as a % of |V|.
        #
        # TWO measurements, because they answer different questions:
        #   - FITTED states: error on the states the regression trained on (in-distribution).
        #   - HELD-OUT states: error on buffered states deliberately EXCLUDED from the regression.
        # The held-out number is the operationally relevant one: after consolidating, the agent
        # immediately collects a NEW rollout and bootstraps from V on states it has not consolidated
        # on. A low fitted error with a high held-out error means the permanent net memorised the
        # buffer and extrapolates badly — good-looking diagnostics while the acting value is
        # corrupted exactly where it is next used. Enable with consolidation_holdout_frac > 0
        # (default 0 keeps the original behaviour, so earlier runs remain reproducible).
        states_np, oldvp_np = self.consolidation_buffer.as_arrays()
        n_total = len(states_np)
        holdout_frac = float(self.cfg.get("consolidation_holdout_frac", 0.0))
        keep = 1.0 - self.transient_decay
        loss_curve = []          # regression loss, one entry per gradient step of this consolidation

        def _probe(arr):
            if arr is None or len(arr) == 0:
                return None
            return torch.as_tensor(arr[: min(4096, len(arr))], dtype=torch.float32,
                                   device=self.device)

        def _value(t):
            if t is None:
                return None
            with torch.no_grad():
                a, b = self.critic(t)
            return a + b

        if holdout_frac <= 0.0:
            # DEFAULT PATH — must consume RNG exactly as the original implementation did, so runs
            # stay comparable across this refactor. iter_minibatches uses np.random.shuffle; adding
            # a np.random.permutation or a torch.randperm here would silently shift every
            # downstream random draw and make same-seed runs non-reproducible.
            fit_probe, hold_probe = _probe(states_np), None
            v_fit_before, v_hold_before = _value(fit_probe), None
            self.last_trans_mean_before, self.last_trans_l2_before = self._trans_stats(fit_probe)
            self.last_perm_mean_before, self.last_perm_l2_before = self._perm_stats(fit_probe)
            for _ in range(self.consolidation_epochs):
                for s_mb, old_vp_mb in self.consolidation_buffer.iter_minibatches(
                        self.minibatch_size, self.device):
                    v_perm, v_trans = self.critic(s_mb)
                    target = old_vp_mb + keep * v_trans.detach()
                    loss = 0.5 * ((v_perm - target) ** 2).mean()
                    loss_curve.append(float(loss.detach()))
                    self.perm_optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.perm.parameters(),
                                                   self.max_grad_norm)
                    self.perm_optim.step()
        else:
            # DIAGNOSTIC PATH — hold part of the buffer out of the regression and measure drift on
            # it separately. This deliberately changes the RNG stream, so runs using it are not
            # seed-comparable with default-path runs.
            perm_idx = np.random.permutation(n_total)
            n_hold = int(n_total * holdout_frac)
            hold_idx, train_idx = perm_idx[:n_hold], perm_idx[n_hold:]
            fit_probe, hold_probe = _probe(states_np[train_idx]), _probe(states_np[hold_idx])
            v_fit_before, v_hold_before = _value(fit_probe), _value(hold_probe)
            self.last_trans_mean_before, self.last_trans_l2_before = self._trans_stats(fit_probe)
            self.last_perm_mean_before, self.last_perm_l2_before = self._perm_stats(fit_probe)
            s_train = torch.as_tensor(states_np[train_idx], dtype=torch.float32, device=self.device)
            v_train = torch.as_tensor(oldvp_np[train_idx], dtype=torch.float32, device=self.device)
            mb = self.minibatch_size
            for _ in range(self.consolidation_epochs):
                order = torch.randperm(len(s_train), device=self.device)
                for i in range(0, len(s_train), mb):
                    sel = order[i:i + mb]
                    v_perm, v_trans = self.critic(s_train[sel])
                    target = v_train[sel] + keep * v_trans.detach()
                    loss = 0.5 * ((v_perm - target) ** 2).mean()
                    loss_curve.append(float(loss.detach()))
                    self.perm_optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.perm.parameters(),
                                                   self.max_grad_norm)
                    self.perm_optim.step()

        # Permanent statistics AFTER the regression (the decay below touches only the transient).
        self.last_perm_mean_after, self.last_perm_l2_after = self._perm_stats(fit_probe)

        # The transient head has been absorbed into the permanent; decay it back down.
        self._decay_transient(self.transient_decay)
        self.last_trans_mean_after, self.last_trans_l2_after = self._trans_stats(fit_probe)
        if loss_curve:
            self.last_consolidation_loss_curve = loss_curve
            self.last_consolidation_loss_first = loss_curve[0]
            self.last_consolidation_loss_last = loss_curve[-1]
            self.last_consolidation_loss_mean = float(np.mean(loss_curve))

        def _drift(t, before):
            if t is None or before is None:
                return None
            after = _value(t)
            return float((after - before).abs().mean() / before.abs().mean().clamp_min(1e-8) * 100.0)


        self.last_consolidation_error = _drift(fit_probe, v_fit_before)
        self.last_consolidation_error_holdout = _drift(hold_probe, v_hold_before)

        self.consolidation_buffer.clear()

    def _perm_stats(self, probe):
        """(mean, L2 norm) of V_perm over a batch of states — the permanent's magnitude."""
        if probe is None:
            return None, None
        with torch.no_grad():
            v_perm, _ = self.critic(probe)
        return float(v_perm.mean()), float(torch.linalg.vector_norm(v_perm))

    def _trans_stats(self, probe):
        """(mean, L2 norm) of V_trans over a batch of states — the transient's magnitude."""
        if probe is None:
            return None, None
        with torch.no_grad():
            _, v_trans = self.critic(probe)
        return float(v_trans.mean()), float(torch.linalg.vector_norm(v_trans))

    def _decay_transient(self, decay):
        """Decay the transient head, optionally clearing its optimiser state at the same time.

        Without the reset, Adam carries momentum for parameters that were just scaled (to zero, when
        decay=0), so the very next update pushes them straight back out — see FINDINGS.md 5.6.
        """
        self.critic.decay_transient(decay)
        if self.reset_trans_optim_on_decay:
            # Drop only the state for the transient parameters (the shared-trunk variant's optimiser
            # also holds the trunk, whose state must be preserved).
            trans_params = set(id(p) for p in self.critic.trans.parameters())
            for group in self.trans_optim.param_groups:
                for p in group["params"]:
                    if id(p) in trans_params:
                        self.trans_optim.state.pop(p, None)

    def on_task_switch(self, step):
        """Task-boundary handling, selectable via cfg['on_switch']:

          - "consolidate" (default): absorb the transient into the permanent (value-preserving),
            then decay the transient — locks the just-learned task value into the permanent baseline.
          - "decay": only decay the transient (no consolidation).
          - "none": do nothing — let the fast transient carry through the boundary and re-adapt on
            its own (periodic k-step consolidation is unaffected).

        Back-compat: if `on_switch` is unset, fall back to `consolidate_on_switch`
        (True -> "consolidate", False -> "decay").
        """
        mode = self.cfg.get("on_switch")
        if mode is None:
            mode = "consolidate" if self.cfg.get("consolidate_on_switch", True) else "decay"
        if mode == "consolidate":
            self._consolidate()                       # absorbs T into P, then decays T
            self._updates_since_consolidation = 0
        elif mode == "decay":
            if self.transient_decay < 1.0:
                self._decay_transient(self.transient_decay)
        # mode == "none": intentionally do nothing

    # ------------------------------------------------------------------
    # Critic optimizer plumbing
    # ------------------------------------------------------------------
    def _zero_critic_grads(self):
        self.trans_optim.zero_grad()
        if self.perm_optim is not None:
            self.perm_optim.zero_grad()

    def _step_critic_optims(self):
        self.trans_optim.step()
        if self.perm_optim is not None:
            self.perm_optim.step()

    def _critic_parameters(self):
        """Critic params that receive gradients (for grad clipping in the main PPO update).

        Shared-trunk: trunk + transient head (the permanent head is updated only by consolidation).
        Separate:     both trunks.
        """
        if self.shared_trunk:
            return list(self.critic.trunk.parameters()) + list(self.critic.trans.parameters())
        return list(self.critic.trans.parameters()) + list(self.critic.perm.parameters())
