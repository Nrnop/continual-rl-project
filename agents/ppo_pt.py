"""Dual-timescale Permanent-Transient PPO agent (value-function decomposition).

PT is applied to the CRITIC only (matching Anand & Precup): V(s) = V_perm(s) + V_trans(s).
The policy is the same single GaussianActor as the vanilla/EWC baselines, so the comparison
stays apples-to-apples and any difference is attributable to the split critic.

- V_trans (θ_T): fast head, trained every PPO update to predict the residual
                 (returns - V_perm.detach()) above the frozen permanent baseline.
- V_perm  (θ_P): slow head, NOT trained on returns each step. Every k updates it *consolidates*
                 by absorbing the transient (see _consolidate), then θ_T is decayed.

Consolidation follows Eq. (4): θ_P regresses onto the full acting value old_V_perm + V_trans, then
θ_T ← decay·θ_T. That target is what gives θ_P the fixed point E_τ[v_τ] (Theorem 5), the mean value
function over the task distribution, which optimises the jumpstart objective (Theorem 6). At a task
boundary we consolidate first (locking the just-learned task value into θ_P) then let θ_T re-adapt.

Initialisation is CONFIGURABLE, because the reference is not consistent about it and the choice
turns out to cost return either way (see models/critic.py and REINVESTIGATION.md §6a):

  θ_T   zero by default (`trans_zero_init`) — Theorem 1's `V^(T)_0 = 0`.
  θ_P   orthogonal gain 1.0 by default; `perm_zero_init` gives the tabular reference's
        `w_1 = np.zeros`; `perm_init_std: 0.01` gives the deep reference's `normal_(0, 0.01)`.

Theorem 1 also requires `V^(TD)_0 = V^(P)` — the baseline must start from the SAME function. Use
`critic_zero_init` on the vanilla arm to satisfy it; without that the equivalence is untested.
"""
import numpy as np
import torch

from .ppo_base import PPOBase
from ..models.critic import SplitCritic
from ..utils.buffers import ConsolidationBuffer


class PPOPT(PPOBase):
    """PPO with a dual-timescale split critic (Permanent + Transient)."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        super().__init__(obs_dim, act_dim, cfg, device)

        # Width for EACH of the two critics. Set `critic_hidden_sizes` below the actor's
        # `hidden_sizes` to reproduce the paper's parameter-matched PT-0.5x (see _critic_hidden).
        hidden = self._critic_hidden()
        adam_eps = cfg.get("adam_eps", 1e-8)
        lr_trans = cfg.get("lr_trans", cfg["lr_actor"])

        # TWO FULLY SEPARATE NETWORKS, as in the reference (T_Net / P_Net in PT_DQN_half.py).
        # No shared trunk: the timescale separation is the point of the decomposition, and sharing
        # features would couple the two components through the fast learner's gradients.
        # perm_zero_init: start theta_P at the zero function instead of a random one. See
        # SplitCritic's docstring — under function approximation a frozen random V_perm is a fixed
        # unstructured offset the transient must cancel on every state, and it fails to cancel it
        # on the NEW states visited after a task switch.
        self.critic = SplitCritic(
            obs_dim, hidden_sizes=hidden,
            perm_zero_init=bool(cfg.get("perm_zero_init", False)),
            # trans_zero_init=False reproduces the REFERENCE's initialisation (both nets random).
            # Default True is Theorem 1's condition — a deviation from the code, see SplitCritic.
            trans_zero_init=bool(cfg.get("trans_zero_init", True)),
            # perm_init_std reproduces the deep reference's normal_(0, 0.01); ignored when
            # perm_zero_init is set. Neither is the default.
            perm_init_std=cfg.get("perm_init_std", None),
        ).to(device)
        # Fast optimizer for the transient net (task-specific). Reference: optim.Adam, --lr2.
        self.trans_optim = torch.optim.Adam(
            self.critic.trans.parameters(), lr=lr_trans, eps=adam_eps
        )
        # Slow optimizer for the permanent net (task-invariant). Reference: optim.SGD, --lr1.
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

        # Decay factor lambda for the transient critic at consolidation / task boundaries.
        self.transient_decay = cfg.get("decay", 0.0)
        # "params" = reference behaviour (over-decays: 0.75 leaves ~0.42 of V_trans on a 3-layer
        # MLP). "output" = exact V_trans <- decay*V_trans, which is what Alg. 2 line 9 specifies.
        # Default "params" so pre-2026-08-04 runs reproduce.
        self.decay_mode = str(cfg.get("decay_mode", "params")).lower()
        # False (default) = the paper's Eq. (4) target, old_V_perm + V_trans.
        # True            = the old keep=(1-decay) target; not the paper, kept for reproducibility.
        self.value_preserving_consolidation = bool(
            cfg.get("value_preserving_consolidation", False))
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
        # Every consolidation's FULL within-cycle loss trace, one array per event, in order.
        # `last_consolidation_loss_*` only keep the first/last/mean, which cannot show whether the
        # regression descends, plateaus or diverges inside a cycle. Kept in memory (25 cycles x
        # ~1920 gradient steps at k=60 is trivial) and dumped at the end of training so the
        # per-consolidation loss curves can be plotted.
        self.consolidation_loss_curves = []
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
        # Fraction of the transient the permanent ACTUALLY absorbed at the last consolidation, and
        # how well that movement aligned with the transient's direction. ~0 means the permanent is
        # inert and PT has no slow timescale — the defect that went undetected across this whole
        # project, because returns and critic_loss both look healthy while it is happening.
        self.last_absorbed_frac = None
        self.last_absorbed_align = None
        # Same, on states EXCLUDED from the regression (needs consolidation_holdout_frac > 0).
        # Decides whether the transfer generalises or is memorised — see _consolidate.
        self.last_absorbed_frac_holdout = None
        self.last_absorbed_align_holdout = None
        # Printed once, loudly, the first time a consolidation transfers essentially nothing.
        self._warned_inert_permanent = False
        # --- Robbins-Monro alpha_P (Theorem 5's premise; see _apply_robbins_monro_alpha_p) ---
        # 0.0 = constant step size (the old behaviour, kept as default so earlier runs reproduce).
        self.rm_power = float(cfg.get("alpha_p_rm_power", 0.0))
        self._n_consolidations = 0
        self.last_alpha_p = self.perm_optim.param_groups[0]["lr"]
        for group in self.perm_optim.param_groups:
            group["_base_lr_perm"] = group["lr"]

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------
    def get_value(self, obs_batch_np):
        """Return (V_perm, V_trans) for a batch: (num_envs, obs_dim) -> arrays (num_envs,)."""
        obs_t = torch.as_tensor(obs_batch_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_perm, v_trans = self.critic(obs_t)
        return v_perm.cpu().numpy(), v_trans.cpu().numpy()

    def diagnostics(self):
        """Where in the k-update consolidation cycle this rollout was collected (D2).

        `_updates_since_consolidation` is incremented in `post_update`, so when read at the TOP of
        `update()` it is exactly how many PPO updates elapsed between the last consolidation and
        the collection of this rollout. age = 0 is the rollout gathered immediately AFTER a
        consolidation — the one acting under a value function that consolidation has just
        displaced.

        Why this matters: the paper's consolidation target is `old_V_perm + V_trans` (keep = 1),
        which is deliberately NOT value-preserving. Right after consolidating,
        V = old_P + T + decay*T — an overshoot of decay*T that the fast transient is supposed to
        correct over the following updates. In DQN that cost is paid for: decaying the transient
        returns behaviour to the task-average policy instantly, which IS the jumpstart. In an
        actor-critic there is no such instantaneous behavioural benefit, so if the perturbation
        costs anything, it is a cost with no compensation.

        Binning returns and advantage statistics by this age tests that directly, and it tests it
        AWAY from task boundaries — a dip locked to the k-grid rather than to the switch grid
        cannot be explained by non-stationarity.
        """
        return {
            "diag/consol_age": float(self._updates_since_consolidation),
            "diag/consol_k": float(self.k),
        }

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
        # Snapshot visited states with the CURRENT permanent value. theta_P is frozen
        # between consolidations, so this is old_V_perm at visit time. The rollout
        # buffer is (n_steps, num_envs, obs_dim); flatten to a batch of states.
        # (Reference: exp_replay_PM.store(cs, c_action, val_p) at action-selection time.)
        states = self.buffer.obs.reshape(-1, self.obs_dim)
        with torch.no_grad():
            s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            old_v_perm, _ = self.critic(s_t)
        self.consolidation_buffer.add_batch(states, old_v_perm.cpu().numpy())

        self._updates_since_consolidation += 1
        if self._updates_since_consolidation >= self.k:
            self._apply_robbins_monro_alpha_p()
            self._consolidate()
            self._updates_since_consolidation = 0

    def _apply_robbins_monro_alpha_p(self):
        """alpha_P <- lr_perm / (1 + n)^rm_power, with n = consolidations so far.

        Theorem 5 ("the sequence of updates computed by Eq. (4) contracts to a unique fixed point
        E_tau[v_tau]") holds *under Robbins-Monro step-size conditions* — sum(a) = inf,
        sum(a^2) < inf — i.e. a DECREASING step size. We were using a constant one, which violates
        the theorem's premise: with a fixed step the permanent TRACKS the most recent task instead
        of converging to the average over tasks, so it arrives at each switch confidently wrong and
        the transient has to cancel it.

        rm_power = 0.0 disables this and restores the constant-alpha behaviour (the default, so
        earlier runs reproduce). rm_power in (0.5, 1.0] satisfies the conditions; 0.6 is a common
        choice that decays slowly enough to keep learning over a finite run.
        """
        if self.rm_power <= 0.0:
            return
        self._n_consolidations += 1
        scale = 1.0 / ((1.0 + self._n_consolidations) ** self.rm_power)
        for group in self.perm_optim.param_groups:
            group["lr"] = group.get("_base_lr_perm", group["lr"]) * scale
        self.last_alpha_p = self.perm_optim.param_groups[0]["lr"]

    def _consolidate(self):
        """Absorb the transient value into the permanent critic, then decay the transient.

        DEFAULT (`value_preserving_consolidation: false`) is the paper's Eq. (4):

            theta <- theta + alpha_bar * (V^(PT)(S) - V^(P)(S)) * grad V^(P)(S)

        i.e. the permanent regresses onto the FULL acting value old_V_perm + V_trans (keep = 1),
        which is also Alg. 4 line 15 (`y_hat = Q^(P)(S,A) + Q^(T)(S,A;w)`). This is what gives the
        permanent its fixed point E_tau[v_tau] (Theorem 5) — the mean value function over the task
        distribution, which optimises the jumpstart objective (Theorem 6). Using keep = 1-decay
        instead shrinks every consolidation step toward a different, biased fixed point, so the
        theory no longer applies.

        The paper's transfer is therefore NOT value-preserving: right after consolidation
        V = P_new + decay*T = old_P + T + decay*T, an overshoot of decay*T that the fast transient
        corrects over the following updates. That is by design.

        Set `value_preserving_consolidation: true` for the old behaviour (keep = 1-decay, exactly
        value-preserving for any decay) — kept only so earlier runs remain reproducible.
        """
        if len(self.consolidation_buffer) == 0:
            return

        # Measure how much the acting value V = V_perm + V_trans actually moves across this
        # consolidation, as a % of |V|.
        #
        # NOTE: under Eq. (4) (keep = 1, the default) a NON-ZERO reading is CORRECT, not a fault.
        # The paper's operator overshoots by decay*V_trans by design, and the fast transient
        # corrects it over the following updates. Read this number as "how far the acting value was
        # displaced", not as an error to be driven to zero — only the old
        # `value_preserving_consolidation: true` path targets 0.
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
        # keep = 1 is Eq. (4) / Alg. 4 line 15 (the paper). keep = 1-decay is the old
        # value-preserving variant, retained behind a flag for reproducing earlier runs.
        keep = (1.0 - self.transient_decay) if self.value_preserving_consolidation else 1.0
        loss_curve = []          # regression loss, one entry per gradient step of this consolidation
        hp_before_vec = ht_before_vec = None   # only filled on the holdout path

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

        def _components(t):
            """(V_perm, V_trans) as detached vectors, for the absorbed-fraction diagnostic."""
            if t is None:
                return None, None
            with torch.no_grad():
                a, b = self.critic(t)
            return a.clone(), b.clone()

        if holdout_frac <= 0.0:
            # DEFAULT PATH — must consume RNG exactly as the original implementation did, so runs
            # stay comparable across this refactor. iter_minibatches uses np.random.shuffle; adding
            # a np.random.permutation or a torch.randperm here would silently shift every
            # downstream random draw and make same-seed runs non-reproducible.
            fit_probe, hold_probe = _probe(states_np), None
            v_fit_before, v_hold_before = _value(fit_probe), None
            self.last_trans_mean_before, self.last_trans_l2_before = self._trans_stats(fit_probe)
            self.last_perm_mean_before, self.last_perm_l2_before = self._perm_stats(fit_probe)
            v_perm_before_vec, v_trans_before_vec = _components(fit_probe)
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
            # Components on the HELD-OUT states, so absorbed_frac can be measured off-distribution.
            hp_before_vec, ht_before_vec = _components(hold_probe)
            self.last_trans_mean_before, self.last_trans_l2_before = self._trans_stats(fit_probe)
            self.last_perm_mean_before, self.last_perm_l2_before = self._perm_stats(fit_probe)
            v_perm_before_vec, v_trans_before_vec = _components(fit_probe)
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

        # ---- DID THE TRANSFER ACTUALLY HAPPEN? ----
        # This is the number whose absence hid the project's central defect for its entire history.
        # `absorbed_frac`  = ||V_perm_after - V_perm_before|| / ||keep * V_trans||  -> 1.0 = the
        #                    permanent took on the whole transient; 0.0 = it did not move.
        # `absorbed_align` = the same movement PROJECTED onto the transient direction. A permanent
        #                    can drift (nonzero frac) while learning nothing useful (align ~ 0).
        # At the inherited sgd/lr_perm=1e-5 these read 0.0004 and 0.000: the dual-timescale
        # mechanism is not running at all, while returns and critic_loss both look healthy.
        #
        # `_holdout` is the SAME quantity on states deliberately EXCLUDED from the regression
        # (needs consolidation_holdout_frac > 0). This is the number that decides whether the
        # transfer is real or memorised: after consolidating, the agent immediately collects a NEW
        # rollout and bootstraps GAE from V_perm + V_trans on states it never consolidated on. A
        # healthy fitted absorbed_frac with a near-zero held-out one means the permanent memorised
        # the buffer and the acting value is corrupted exactly where it is next used — worst right
        # after a switch, when the state distribution has just moved.
        #
        # Offline, the target IS fittable (FINDINGS 6.3: a [256,256] net reaches 3.2% train error)
        # but held-out error floors at 38-44% and gets WORSE the harder it is fitted. This measures
        # whether that gap appears in situ. It has never been measured on a permanent that actually
        # moves — the one earlier holdout run was made when theta_P was inert, so both numbers were
        # trivially near zero.
        self.last_absorbed_frac = None
        self.last_absorbed_align = None
        self.last_absorbed_frac_holdout = None
        self.last_absorbed_align_holdout = None

        def _absorbed(probe, p_before, t_before):
            if probe is None or p_before is None:
                return None, None
            with torch.no_grad():
                p_after, _ = self.critic(probe)
            moved, wanted = p_after - p_before, keep * t_before
            wn = float(wanted.pow(2).sum())
            if wn <= 1e-12:
                return None, None
            return float(moved.norm() / wanted.norm()), float((moved * wanted).sum() / wn)

        self.last_absorbed_frac, self.last_absorbed_align = _absorbed(
            fit_probe, v_perm_before_vec, v_trans_before_vec)
        if holdout_frac > 0.0:
            self.last_absorbed_frac_holdout, self.last_absorbed_align_holdout = _absorbed(
                hold_probe, hp_before_vec, ht_before_vec)

        if self.last_absorbed_frac is not None:
                    if (not self._warned_inert_permanent
                            and self.last_absorbed_frac < 0.01):
                        self._warned_inert_permanent = True
                        lr = self.perm_optim.param_groups[0]["lr"]
                        print(
                            "\n" + "!" * 78 +
                            "\n[PT] INERT PERMANENT: this consolidation transferred "
                            f"{self.last_absorbed_frac * 100:.3f}% of the transient "
                            f"(alignment {self.last_absorbed_align:+.3f}).\n"
                            f"[PT] theta_P is not learning, so PT has NO slow timescale and is "
                            "equivalent to\n[PT] vanilla plus a frozen random offset plus a "
                            "periodic decay. Results from this\n[PT] configuration say nothing "
                            "about the permanent-transient method.\n"
                            f"[PT] current: perm_optimizer={self.cfg.get('perm_optimizer')} "
                            f"lr_perm={lr:g} consolidation_epochs={self.consolidation_epochs}\n"
                            "[PT] alpha_P must be TUNED PER DOMAIN (the paper's own ranges span "
                            "7 orders of\n[PT] magnitude). Sweep it before trusting any PT "
                            "number.\n" + "!" * 78 + "\n", flush=True)

        # The transient head has been absorbed into the permanent; decay it back down.
        self._decay_transient(self.transient_decay)
        self.last_trans_mean_after, self.last_trans_l2_after = self._trans_stats(fit_probe)
        if loss_curve:
            self.last_consolidation_loss_curve = loss_curve
            self.last_consolidation_loss_first = loss_curve[0]
            self.last_consolidation_loss_last = loss_curve[-1]
            self.last_consolidation_loss_mean = float(np.mean(loss_curve))
            # Keep the whole trace, not just its endpoints — see __init__.
            self.consolidation_loss_curves.append(np.asarray(loss_curve, dtype=np.float32))

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
        self.critic.decay_transient(decay, mode=self.decay_mode)
        if self.reset_trans_optim_on_decay:
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
        super().on_task_switch(step)          # split actor, when enabled (pt_both)
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
        self.perm_optim.zero_grad()

    def _step_critic_optims(self):
        self.trans_optim.step()
        # No-op during PPO updates: v_perm is detached in critic_loss, so theta_P has no gradient
        # (zero_grad leaves p.grad as None and every torch optimizer skips such params). theta_P
        # moves only inside _consolidate. Kept for symmetry with _zero_critic_grads.
        self.perm_optim.step()

    def _critic_parameters(self):
        """Both trunks, for grad clipping in the main PPO update (only theta_T carries grads)."""
        return list(self.critic.trans.parameters()) + list(self.critic.perm.parameters())
