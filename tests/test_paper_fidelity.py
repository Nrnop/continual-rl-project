"""The algorithmic conditions from Anand & Precup (2023) that the port previously violated.

Each test names the result it enforces, so a future refactor that breaks one fails loudly rather
than silently reintroducing a deviation.

PORTED IN PHASE 2 (T2b). This suite was written against the old critic-only PT agent, which
Phase 2 removes. The theorems it guards are about the VALUE DECOMPOSITION, which the surviving
agent also has (`SplitCritic`), so almost everything ports with an agent-class swap plus config-key
updates. Three deliberate differences from the old file, each of them a real behavioural change in
the agent rather than a test being weakened:

1. `keep` is now `rho`. The old agent regressed the permanent onto `old_V_perm + keep*V_trans`
   with keep = 1 (Eq. 4 / Alg. 4 line 15) and decayed the transient by a SEPARATE `decay`. The
   surviving agent ties the two halves together: the permanent absorbs `rho*V_trans` and the
   transient retains `(1-rho)`, so the composition is preserved and behaviour does not jump. The
   paper's keep = 1 is the special case rho = 1, which is what
   `test_consolidation_target_is_the_full_acting_value` now pins. The old
   `value_preserving_consolidation` flag has no counterpart; `decay_rho` is its nearest relative
   and is tested below.
2. Robbins-Monro alpha_P now defaults ON at 0.6. The old agent defaulted it off for
   reproducibility of pre-existing runs; there is nothing left to reproduce, and Theorem 5's
   premise is a DECREASING step size. The test asserts the decay and that it reaches the actor's
   permanent optimiser too, which is new.
3. Parameter parity is now a property of the ACTOR as well, because the actor is split too. The
   old test asserted `n(pt.actor) == n(vanilla.actor)`, which is false by construction for a split
   actor at equal width — see `test_parameter_parity_is_reachable_for_a_split_actor_and_critic`
   for the PT-0.5x construction that restores it, and
   `test_default_widths_double_the_actor_and_critic` for the bound that guards against the 13.9x
   blowup Phase 1 found in a published config.

EVERYTHING PORTED. `test_absorbed_frac_is_also_measured_off_distribution` briefly could not: the
old agent's `consolidation_holdout_frac` path — which holds part of the buffer out of the
regression and measures absorption on it, separating a permanent that GENERALISES from one that
memorised the buffer — had no counterpart in the surviving agent. That path has since been
implemented in `_consolidate` for BOTH the critic and the actor, and the test is live again,
alongside `test_the_holdout_split_actually_withholds_states`, which checks the split really
withholds rather than just reporting a second number.
"""
import copy

import numpy as np
import pytest
import torch

from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.models.critic import SplitCritic, VanillaCritic
from src_continuous_control.utils.metrics import JumpstartTracker, RetentionProbe


def _cfg(**kw):
    c = dict(hidden_sizes=[32, 32], lr_actor=3e-4, adam_eps=1e-5, num_envs=2, n_steps=8,
             gamma=0.99, gae_lambda=0.95, clip_coef=0.2, epochs=1, minibatch_size=8,
             ent_coef=0.0, max_grad_norm=0.5, target_kl=None, normalize_advantage=True,
             lr_trans=3e-4, lr_perm=1e-5, perm_optimizer="sgd", rho=0.5, k=2,
             kl_prior_coef=0.01, consolidation_epochs=1, consolidation_buffer_size=64)
    c.update(kw)
    return c


def _agent(**kw):
    return PPOPT(17, 6, _cfg(**kw), torch.device("cpu"))


def _n_params(module):
    return sum(p.numel() for p in module.parameters())


# --------------------------------------------------------------- Theorem 1: V^(T)_0 = 0
def test_transient_is_zero_at_initialisation():
    """Theorem 1 requires V^(T)_0 = 0 for PT to reduce exactly to TD at t=0."""
    torch.manual_seed(0)
    critic = SplitCritic(17, hidden_sizes=(64, 64))
    probe = torch.randn(512, 17)
    with torch.no_grad():
        v_perm, v_trans = critic(probe)
    assert torch.all(v_trans == 0), "theta_T must start at the zero function"
    assert v_perm.abs().mean() > 0, "theta_P keeps the ordinary value-head init"
    # ...and therefore the acting value at t=0 is exactly the permanent alone.
    assert torch.allclose(v_perm + v_trans, v_perm)


def test_the_split_actor_also_starts_at_the_zero_function():
    """The same condition on the policy side: mu_T(s) == 0 at init, so pi_PT == pi_P exactly.

    New in Phase 2 — the old critic-only agent had no transient policy to check. Without this the
    agent starts as the sum of two independent random policies, which is strictly noisier than the
    vanilla actor it is compared against.
    """
    torch.manual_seed(0)
    agent = _agent()
    probe = torch.randn(256, 17)
    with torch.no_grad():
        mu_trans = agent.actor.trans_mean(probe)
        mu_perm = agent.actor.perm_mean(probe)
    assert torch.all(mu_trans == 0), "mu_T must start at the zero function"
    assert mu_perm.abs().mean() > 0, "mu_P keeps the ordinary policy-head init"
    assert torch.allclose(agent.actor.act_deterministic(probe), mu_perm)
    # KL(pi_PT || pi_P) is therefore exactly 0 at init.
    assert float(agent.actor.kl_to_prior(probe).detach().max()) == 0.0


def test_perm_zero_init_makes_the_pt_loss_identical_to_vanilla():
    """With V^(P) = 0 and theta_P frozen, PT's semi-gradient loss IS vanilla's loss.

    This is the only configuration in which Theorem 1's equivalence is meaningful under function
    approximation. With a RANDOM frozen V_perm the transient's target is R - V_perm — the value
    function minus a fixed unstructured function it must cancel on every state — which it cannot do
    on the new states visited after a task switch (Job D: phase 1 p=0.940, phase 2 p=0.001).
    """
    torch.manual_seed(0)
    probe = torch.randn(256, 17)
    returns = torch.randn(256)

    zero = SplitCritic(17, hidden_sizes=(64, 64), perm_zero_init=True)
    with torch.no_grad():
        v_perm, v_trans = zero(probe)
    assert torch.all(v_perm == 0), "perm_zero_init must give V^(P) == 0 exactly"

    # PT's loss with a zero permanent == a single critic's loss on the same transient net.
    pt_loss = 0.5 * ((v_perm.detach() + v_trans - returns) ** 2).mean()
    vanilla_equivalent = 0.5 * ((v_trans - returns) ** 2).mean()
    assert torch.allclose(pt_loss, vanilla_equivalent)

    # Default stays random, so existing configs are unaffected.
    rand = SplitCritic(17, hidden_sizes=(64, 64))
    with torch.no_grad():
        vp, _ = rand(probe)
    assert vp.abs().mean() > 0


def test_theorem1_second_condition_can_now_be_met():
    """Theorem 1 needs V^(TD)_0 = V^(P) as well as V^(T)_0 = 0.

    With perm_zero_init the PT acting value starts at exactly 0, so the TD baseline must start
    there too or the two agents never begin from the same function — which is what the theorem
    asserts equivalence between. `critic_zero_init` supplies that.
    """
    torch.manual_seed(0)
    probe = torch.randn(256, 17)

    default = VanillaCritic(17, hidden_sizes=(64, 64))
    matched = VanillaCritic(17, hidden_sizes=(64, 64), zero_init=True)
    pt = SplitCritic(17, hidden_sizes=(64, 64), perm_zero_init=True)

    with torch.no_grad():
        v_default, v_matched = default(probe), matched(probe)
        p, t = pt(probe)

    assert v_default.abs().mean() > 0, "the default baseline starts at a random function"
    assert torch.all(v_matched == 0), "critic_zero_init must give V(s) == 0 exactly"
    # ...and that now equals the PT agent's acting value at t=0.
    assert torch.allclose(v_matched, p + t)


def test_perm_zero_init_still_lets_the_permanent_learn():
    """Zeroing only the output layer must not freeze theta_P — consolidation still moves it."""
    torch.manual_seed(0)
    critic = SplitCritic(17, hidden_sizes=(32, 32), perm_zero_init=True)
    opt = torch.optim.SGD(critic.perm.parameters(), lr=1e-2)
    obs, target = torch.randn(64, 17), torch.randn(64)
    for _ in range(50):
        v_perm, _ = critic(obs)
        loss = ((v_perm - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        v_perm, _ = critic(obs)
    assert v_perm.abs().mean() > 1e-3, "permanent stayed pinned at zero"


def test_transient_leaves_zero_after_training():
    """Zeroing only the OUTPUT layer must not freeze the transient."""
    torch.manual_seed(0)
    critic = SplitCritic(17, hidden_sizes=(32, 32))
    opt = torch.optim.Adam(critic.trans.parameters(), lr=1e-2)
    obs, target = torch.randn(64, 17), torch.randn(64)
    for _ in range(20):
        _, v_trans = critic(obs)
        loss = ((v_trans - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, v_trans = critic(obs)
    assert v_trans.abs().mean() > 1e-3, "transient stayed pinned at zero"


# ------------------------------------------------- Eq. (4) / Alg. 4 line 15: keep = 1
def _drive_consolidation(agent, states, trans_scale=2.0):
    """Give both transients a learned magnitude, then run one consolidation.

    Returns (old_v_perm, old_v_trans, new_v_perm) over `states`, all detached.
    """
    with torch.no_grad():
        for p in agent.critic.trans[-1].parameters():
            p.add_(torch.randn_like(p) * trans_scale)
        for p in agent.actor.trans_mean[-1].parameters():
            p.add_(torch.randn_like(p) * trans_scale)
        s_t = torch.as_tensor(states)
        old_vp, old_vt = agent.critic(s_t)
    agent.consolidation_buffer.add_batch(states)
    agent._consolidate()
    with torch.no_grad():
        new_vp, _ = agent.critic(s_t)
    return old_vp, old_vt, new_vp


def test_consolidation_target_is_the_full_acting_value():
    """Eq. (4) regresses V^(P) onto V^(PT) = old_V_perm + V_trans. That is rho = 1 here."""
    torch.manual_seed(0)
    agent = _agent(rho=1.0, lr_perm=1e-2, perm_optimizer="adam", rm_power=0.0,
                   consolidation_epochs=200, consolidation_buffer_size=512)
    states = np.random.randn(256, 17).astype(np.float32)
    old_vp, old_vt, new_vp = _drive_consolidation(agent, states)

    moved, wanted = (new_vp - old_vp), old_vt
    # The permanent should have moved toward the FULL transient, not toward some fraction of it.
    frac = float((moved * wanted).sum() / (wanted * wanted).sum())
    assert frac > 0.5, f"permanent absorbed only {frac:.2f} of V_trans; expected -> 1.0 (rho=1)"


def test_consolidation_absorbs_exactly_rho_of_the_transient():
    """rho < 1 must scale the target, not the direction: theta_P regresses onto P + rho*T.

    This is the half of the mechanism that pairs with the (1-rho) decay below. Splitting them —
    absorbing rho while decaying by something else — duplicates the transient's contribution, which
    is what `decay_rho` exists to reproduce and why it is not a knob to tune.
    """
    torch.manual_seed(0)
    agent = _agent(rho=0.5, lr_perm=1e-2, perm_optimizer="adam", rm_power=0.0,
                   consolidation_epochs=200, consolidation_buffer_size=512)
    states = np.random.randn(256, 17).astype(np.float32)
    old_vp, old_vt, new_vp = _drive_consolidation(agent, states)

    moved, wanted = (new_vp - old_vp), old_vt
    frac = float((moved * wanted).sum() / (wanted * wanted).sum())
    assert 0.3 < frac < 0.7, f"expected the permanent to absorb ~rho=0.5 of V_trans, got {frac:.2f}"
    # ...and the agent's own diagnostic, which normalises by rho*V_trans, should read ~1.
    assert agent.last_absorbed_align == pytest.approx(1.0, abs=0.25)


def test_the_decay_is_the_other_half_of_the_same_transfer():
    """Absorb rho, retain (1-rho) — one transfer, so the composed function does not jump.

    Replaces the old `value_preserving_consolidation` flag, which decoupled the two halves for
    reproducibility. Here `decay_rho` is that flag's nearest relative: it DEFAULTS to rho (the
    composition-preserving split) and exists only to reproduce the measured divergence when the
    two are decoupled.
    """
    torch.manual_seed(0)
    agent = _agent(rho=0.25, lr_perm=0.0, rm_power=0.0)
    assert agent.decay_rho == agent.rho == 0.25

    states = np.random.randn(64, 17).astype(np.float32)
    s_t = torch.as_tensor(states)
    with torch.no_grad():
        for p in agent.critic.trans[-1].parameters():
            p.add_(torch.randn_like(p) * 2.0)
        for p in agent.actor.trans_mean[-1].parameters():
            p.add_(torch.randn_like(p) * 2.0)
        _, vt_before = agent.critic(s_t)
        mu_t_before = agent.actor.trans_mean(s_t).clone()

    agent.consolidation_buffer.add_batch(states)
    agent._consolidate()

    with torch.no_grad():
        _, vt_after = agent.critic(s_t)
        mu_t_after = agent.actor.trans_mean(s_t)
    # Output-layer scaling, so the retained fraction is EXACT for both components.
    assert torch.allclose(vt_after, 0.75 * vt_before, atol=1e-5)
    assert torch.allclose(mu_t_after, 0.75 * mu_t_before, atol=1e-5)

    # The decoupled setting stays reachable — it is the control that produced the divergence.
    decoupled = _agent(rho=0.25, decay_rho=0.9)
    assert decoupled.rho == 0.25 and decoupled.decay_rho == 0.9
    with pytest.raises(ValueError):
        _agent(decay_rho=1.5)


# ------------------------------------------- Sec. 6.1 / C.3: parameter-matched PT-0.5x
def test_critic_hidden_sizes_shrinks_the_critic_but_not_the_actor():
    """PT-0.5x needs a narrower critic while the actor stays identical across agents."""
    torch.manual_seed(0)
    wide = _agent(hidden_sizes=[64, 64])
    narrow = _agent(hidden_sizes=[64, 64], critic_hidden_sizes=[43, 43],
                    critic_trans_hidden_sizes=[43, 43])
    assert _n_params(wide.actor) == _n_params(narrow.actor), \
        "critic_hidden_sizes must not touch the actor"
    assert _n_params(narrow.critic) < _n_params(wide.critic)


def test_parameter_parity_is_reachable_for_a_split_actor_and_critic():
    """PT-0.5x extended to the actor: two half-width nets ~= one full-width net.

    "we use half the number of parameters as that of DQN for both permanent and transient value
    networks to ensure the total number of parameters across all baselines are same" (Sec. 6.1).
    With a SPLIT ACTOR the same argument applies to the policy, or PT gets 2x the capacity for
    free and Appendix C.3's "big world - small agent" boundary condition is silently violated.
    """
    torch.manual_seed(0)
    pt = _agent(hidden_sizes=[43, 43], actor_trans_hidden_sizes=[43, 43],
                critic_hidden_sizes=[43, 43], critic_trans_hidden_sizes=[43, 43])
    van = PPOVanilla(17, 6, _cfg(hidden_sizes=[64, 64], critic_hidden_sizes=[64, 64]),
                     torch.device("cpu"))

    actor_ratio = _n_params(pt.actor) / _n_params(van.actor)
    critic_ratio = _n_params(pt.critic) / _n_params(van.critic)
    assert 0.9 < actor_ratio < 1.1, f"PT actor is {actor_ratio:.2f}x vanilla's; expected ~1.0"
    assert 0.9 < critic_ratio < 1.1, f"PT critic is {critic_ratio:.2f}x vanilla's; expected ~1.0"


def test_default_widths_double_the_actor_and_critic():
    """At equal width PT has exactly 2x the parameters — bounded, and never more.

    Phase 1 found a published config that handed PT 13.9x the baseline's parameters. This pins the
    honest number for the default config so a width change cannot quietly inflate it; T8's
    pre-flight parameter-count print is the run-time version of the same check.
    """
    torch.manual_seed(0)
    pt = _agent(hidden_sizes=[64, 64], actor_trans_hidden_sizes=[64, 64],
                critic_trans_hidden_sizes=[64, 64])
    van = PPOVanilla(17, 6, _cfg(hidden_sizes=[64, 64]), torch.device("cpu"))
    # log_std (6 params) is shared, not duplicated, so the ratio is just under 2.
    assert 1.9 < _n_params(pt.actor) / _n_params(van.actor) <= 2.0
    assert 1.9 < _n_params(pt.critic) / _n_params(van.critic) <= 2.0


def test_default_critic_width_follows_hidden_sizes():
    """Without critic_hidden_sizes nothing changes, so existing configs behave as before."""
    torch.manual_seed(0)
    pt = _agent(hidden_sizes=[64, 64], critic_trans_hidden_sizes=[64, 64])
    ref = SplitCritic(17, hidden_sizes=(64, 64), trans_hidden_sizes=(64, 64))
    assert _n_params(pt.critic) == _n_params(ref)


# --------------------------------------------------------- Thm 6/7/8 instrumentation
def test_jumpstart_tracker_windows_the_post_switch_returns():
    t = JumpstartTracker(window_steps=30)
    t.on_switch(100)
    assert t.update(110, 5.0) is None
    assert t.update(120, 7.0) is None
    rec = t.update(130, 9.0)
    assert rec is not None
    assert rec["first"] == 5.0 and rec["end"] == 9.0 and rec["gain"] == 4.0
    assert rec["mean"] == 7.0
    assert t.mean_jumpstart() == 7.0


def test_retention_probe_scores_only_inactive_tasks():
    probe = np.random.randn(32, 17).astype(np.float32)
    rp = RetentionProbe()
    # No snapshots yet -> nothing to report.
    assert rp.measure(1, {"full": lambda s: np.zeros(len(s))}, probe) == {}

    # v_1 == 1.0 everywhere (the converged acting value at the end of task 1).
    rp.snapshot(1, lambda s: np.ones(len(s)), probe)
    # Task 1 is still active -> still nothing.
    assert rp.measure(1, {"perm": lambda s: np.zeros(len(s))}, probe) == {}
    # Now on task -1, both components are scored against the SAME reference v_1.
    out = rp.measure(-1, {"perm": lambda s: np.zeros(len(s)),
                          "full": lambda s: np.full(len(s), 3.0)}, probe)
    assert np.isclose(out["perm"], 1.0) and np.isclose(out["full"], 4.0)


def test_output_mode_decay_is_exact_and_params_mode_is_not():
    """Alg. 2 line 9 decays the VALUE FUNCTION by lambda. Only mode='output' actually does that.

    mode='params' is the reference's `p.data *= decay` over every parameter. On a nonlinear net
    that over-decays badly: at decay=0.75 it leaves ~0.42 of V_trans, so 58% is deleted where the
    algorithm says 25%. The decay is a fixed cost paid every k updates, so this inflates it
    throughout training.
    """
    torch.manual_seed(0)
    probe = torch.randn(512, 17)
    for decay in (0.25, 0.5, 0.75, 0.95):
        exact = SplitCritic(17, hidden_sizes=(43, 43))
        with torch.no_grad():
            for p in exact.trans[-1].parameters():
                p.add_(torch.randn_like(p) * 2.0)
            _, before = exact(probe)
        loose = copy.deepcopy(exact)

        exact.decay_transient(decay, mode="output")
        loose.decay_transient(decay, mode="params")
        with torch.no_grad():
            _, after_exact = exact(probe)
            _, after_loose = loose(probe)

        assert torch.allclose(after_exact, decay * before, atol=1e-5), (
            f"decay={decay}: output mode must scale V_trans exactly")
        ratio = float(after_loose.abs().mean() / before.abs().mean())
        assert ratio < decay, (
            f"decay={decay}: params mode must OVER-decay, got ratio {ratio:.3f}")
        if decay == 0.75:      # the production value
            # How far params-mode overshoots depends on the transient's learned magnitude (a
            # larger output layer makes the net more linear-dominated). On this probe it retains
            # ~0.59 against an intended 0.75; the in-situ measurement in FINDINGS 6.1 at decay=0.5
            # retained 0.166 against an intended 0.5, i.e. considerably worse under real training.
            assert ratio < decay - 0.10, (
                f"expected params mode to over-decay materially at decay=0.75 "
                f"(exact would be 0.75); got {ratio:.3f}")


def test_actor_and_critic_are_clipped_separately():
    """The actor's update must not depend on the critic's gradient magnitude.

    With a JOINT clip_grad_norm_ over actor+critic, one scale factor derived from the total norm
    is applied to every gradient — so a critic with a different loss surface (PT's two [43,43]
    nets vs vanilla's single [64,64]) changes the ACTOR's effective step size. That silently
    breaks "all agents share an identical actor; only the critic differs", which the whole study
    rests on.
    """
    torch.manual_seed(0)
    agent = _agent()

    def actor_grad_after_clip(a, critic_grad_scale):
        trainable = [p for p in a.actor.parameters() if p.requires_grad]
        for p in trainable:
            p.grad = torch.ones_like(p) * 0.1
        for p in a._critic_parameters():
            p.grad = torch.ones_like(p) * critic_grad_scale
        a._clip_grads()
        return torch.cat([p.grad.flatten() for p in trainable]).norm().item()

    small = actor_grad_after_clip(agent, 0.01)
    huge = actor_grad_after_clip(agent, 100.0)
    assert np.isclose(small, huge, rtol=1e-5), (
        f"actor gradient changed with the critic's magnitude ({small:.5f} vs {huge:.5f}) — "
        "the clip is still joint")

    # ...and the old behaviour stays reachable for reproducing earlier runs.
    joint = _agent(joint_grad_clip=True)
    assert not np.isclose(actor_grad_after_clip(joint, 0.01),
                          actor_grad_after_clip(joint, 100.0), rtol=1e-5)


def test_robbins_monro_alpha_p_decays_and_is_on_by_default():
    """Theorem 5 requires a DECREASING alpha_P; a constant one makes theta_P track, not average.

    Changed in Phase 2: this now DEFAULTS ON at 0.6 (the old agent defaulted it off so that
    pre-existing runs reproduced), and it governs the actor's permanent optimiser as well as the
    critic's — the actor is split now, so it has its own alpha_P.
    """
    on = _agent(lr_perm=1e-2, lr_perm_actor=1e-2)
    assert on.rm_power == 0.6 and on.rm_power_actor == 0.6, "Theorem 5's premise must be the default"
    base_c = on.perm_optim.param_groups[0]["lr"]
    base_a = on.perm_actor_optim.param_groups[0]["lr"]

    critic_lrs, actor_lrs = [], []
    for _ in range(4):
        on._n_consolidations += 1
        on._set_next_permanent_lrs()
        critic_lrs.append(on.perm_optim.param_groups[0]["lr"])
        actor_lrs.append(on.perm_actor_optim.param_groups[0]["lr"])
    for seen, base in ((critic_lrs, base_c), (actor_lrs, base_a)):
        assert all(seen[i] > seen[i + 1] for i in range(len(seen) - 1)), f"alpha_P not decreasing: {seen}"
        assert np.isclose(seen[0], base / 2 ** 0.6)

    # ...and it stays constant when switched off, so the constant-alpha control is still available.
    off = _agent(lr_perm=1e-2, rm_power=0.0, rm_power_actor=0.0)
    lr0 = off.perm_optim.param_groups[0]["lr"]
    for _ in range(5):
        off._n_consolidations += 1
        off._set_next_permanent_lrs()
    assert off.perm_optim.param_groups[0]["lr"] == lr0

    # The old config key still resolves, so archived overlays keep meaning what they said.
    alias = _agent(alpha_p_rm_power=0.3)
    assert alias.rm_power == 0.3


def test_absorbed_frac_detects_an_inert_permanent():
    """sgd/1e-5 must report absorbed_frac ~ 0; a real transfer must report ~1.

    This diagnostic is the one that was missing for the whole project: with an inert permanent,
    returns and critic_loss both look healthy while PT has no slow timescale at all.
    """
    states = np.random.randn(2048, 17).astype(np.float32)

    def absorbed(lr, opt):
        torch.manual_seed(0)
        # 3 epochs, because the actor's permanent regresses a 6-dim target against the critic's
        # scalar one and needs more than a single pass to transfer most of it.
        agent = _agent(lr_perm=lr, lr_perm_actor=lr, perm_optimizer=opt, k=1, rm_power=0.0,
                       consolidation_epochs=3, consolidation_buffer_size=4096)
        _drive_consolidation(agent, states, trans_scale=3.0)
        return agent.last_absorbed_frac, agent.last_actor_absorbed_frac

    inert_c, inert_a = absorbed(1e-5, "sgd")
    working_c, working_a = absorbed(1e-3, "adam")
    assert None not in (inert_c, inert_a, working_c, working_a)
    assert inert_c < 0.05, f"sgd/1e-5 should be inert, got absorbed_frac={inert_c:.4f}"
    assert working_c > 0.5, f"adam/1e-3 should transfer most of it, got {working_c:.4f}"
    # The ACTOR's permanent is the one CLAUDE.md flags as must-check before trusting a run:
    # below 0.01 the permanent policy is inert and the arm says nothing.
    assert inert_a < 0.05, f"actor permanent should be inert too, got {inert_a:.4f}"
    assert working_a > 0.5, f"actor permanent should transfer, got {working_a:.4f}"


def test_absorbed_frac_is_also_measured_off_distribution():
    """With a holdout, absorbed_frac must be reported on states the regression never saw.

    absorbed_frac alone is in-distribution: a permanent that memorised the consolidation buffer
    reports a healthy number while being wrong on the NEW states the next rollout bootstraps from.
    Offline, the target is fittable to 3.2% train error while held-out error floors at 38-44%
    (FINDINGS 6.3) — this is the in-situ version of that measurement.
    """
    states = np.random.randn(2048, 17).astype(np.float32)
    torch.manual_seed(0)
    agent = _agent(lr_perm=1e-3, lr_perm_actor=1e-3, perm_optimizer="adam", k=1, rm_power=0.0,
                   consolidation_epochs=3, consolidation_buffer_size=4096,
                   consolidation_holdout_frac=0.25)
    _drive_consolidation(agent, states, trans_scale=3.0)

    assert agent.last_absorbed_frac is not None
    assert agent.last_absorbed_frac_holdout is not None, \
        "holdout absorbed_frac must be measured when consolidation_holdout_frac > 0"
    assert agent.last_absorbed_align_holdout is not None
    # The actor's permanent gets the same treatment — it is the component that acts.
    assert agent.last_actor_absorbed_frac_holdout is not None
    assert agent.last_actor_absorbed_align_holdout is not None
    # Both numbers reach the saved record, or the sweep cannot be audited afterwards.
    record = agent.consolidation_records[-1]
    assert record["absorbed_frac_holdout"] == agent.last_absorbed_frac_holdout
    assert record["actor_absorbed_frac_holdout"] == agent.last_actor_absorbed_frac_holdout

    # ...and it stays None when no holdout is requested, so the default path is unchanged.
    torch.manual_seed(0)
    plain = _agent(lr_perm=1e-3, perm_optimizer="adam", k=1, rm_power=0.0,
                   consolidation_buffer_size=4096)
    _drive_consolidation(plain, states, trans_scale=3.0)
    assert plain.last_absorbed_frac is not None
    assert plain.last_absorbed_frac_holdout is None


def test_the_holdout_split_actually_withholds_states():
    """The held-out states must be excluded from the regression, not merely measured separately.

    A "holdout" that the regression still trained on would report a healthy number on memorised
    states and give exactly the false reassurance it exists to prevent — this project's failure
    mode #2, a manipulation that never fires.
    """
    torch.manual_seed(0)
    agent = _agent(lr_perm=1e-2, lr_perm_actor=1e-2, perm_optimizer="adam", k=1, rm_power=0.0,
                   consolidation_epochs=1, consolidation_buffer_size=4096,
                   consolidation_holdout_frac=0.25)
    states = np.random.randn(256, 17).astype(np.float32)
    seen = []
    real_iter = agent._iter_indices

    def spy(n, subset=None):
        for batch in real_iter(n, subset=subset):
            seen.append(np.asarray(batch))
            yield batch

    agent._iter_indices = spy
    _drive_consolidation(agent, states)

    trained_on = set(np.concatenate(seen).tolist())
    assert len(trained_on) < len(states), "every state was trained on — nothing was held out"
    # One epoch over 75% of 256 states.
    assert 0.7 * len(states) <= len(trained_on) <= 0.8 * len(states)


def test_boundary_tracker_rejects_a_sub_update_window():
    """A window shorter than 2 updates reports drop=0 by construction — must fail loudly.

    This is the bug that produced 0.00 at boundaries 1-3 for all 40 runs of the 2026-08-03 sweep:
    the window was n_steps*5 = 1280 env steps against a 2048-step batch, so the tracker finalised
    on its first post-switch sample.
    """
    from src_continuous_control.utils.metrics import BoundaryReturnTracker
    batch = 2048
    with pytest.raises(ValueError, match="finalise on its first sample"):
        BoundaryReturnTracker(post_window_steps=1280, min_useful_steps=batch)
    # A 5-update window is fine.
    BoundaryReturnTracker(post_window_steps=5 * batch, min_useful_steps=batch)


def test_boundary_tracker_captures_a_real_trough():
    """With a proper window the tracker must see a trough several updates after the switch."""
    from src_continuous_control.utils.metrics import BoundaryReturnTracker
    batch = 2048
    t = BoundaryReturnTracker(post_window_steps=5 * batch, min_useful_steps=batch)
    t.on_switch(0, 1000.0)
    for i, ret in enumerate([900.0, 400.0, 250.0, 600.0], start=1):   # trough at update 3
        assert t.update(i * batch, ret) is None
    rec = t.update(5 * batch, 800.0)
    assert rec is not None
    assert rec["pre"] == 1000.0 and rec["trough"] == 250.0 and rec["drop"] == 750.0


def test_retention_does_not_reward_a_frozen_permanent():
    """A permanent that never moves must NOT score a perfect 0 — it retains nothing about v_i.

    Guards against scoring each component against its own earlier snapshot, which makes a frozen
    network look ideal while containing no task knowledge.
    """
    probe = np.random.randn(32, 17).astype(np.float32)
    rp = RetentionProbe()
    frozen = lambda s: np.zeros(len(s))          # a permanent stuck at its init
    rp.snapshot(1, lambda s: np.ones(len(s)), probe)   # v_1 = 1.0
    out = rp.measure(-1, {"perm": frozen}, probe)
    assert out["perm"] > 0.5, "a frozen permanent must report a large retention error"


def test_retention_control_baselines_expose_an_inert_permanent():
    """The perm_init baseline must make an inert permanent detectable.

    This is the check that was missing when FINDINGS 8.3.2 wrongly read `mse_perm < mse_full` as
    confirming Theorem 7: on a sign-flip task pair a permanent frozen at zero beats an adapted
    estimate automatically. Scored against its OWN initialisation, an inert permanent is exposed.
    """
    probe = np.random.randn(64, 17).astype(np.float32)
    v_i = np.linspace(-5, 5, 64).astype(np.float32)          # task i's converged values
    perm_at_init = np.zeros(64, dtype=np.float32)

    rp = RetentionProbe()
    rp.set_baseline("perm_init", perm_at_init)
    rp.set_baseline("zero", np.zeros(64, dtype=np.float32))
    rp.snapshot(1, lambda s: v_i, probe)

    # (a) INERT permanent: still sitting at its init. Beats `full` (which flipped sign) but is
    #     indistinguishable from perm_init -> correctly flagged as having learned nothing.
    out = rp.measure(-1, {"perm": lambda s: perm_at_init, "full": lambda s: -v_i}, probe)
    assert out["perm"] < out["full"], "sanity: the sign-flip artifact reproduces"
    assert np.isclose(out["perm"], out["perm_init"]), \
        "an inert permanent must be indistinguishable from its own initialisation"

    # (b) WORKING permanent: has moved toward v_i. Must beat BOTH controls.
    out = rp.measure(-1, {"perm": lambda s: 0.7 * v_i, "full": lambda s: -v_i}, probe)
    assert out["perm"] < out["perm_init"], "a learning permanent must beat its initialisation"
    assert out["perm"] < out["zero"], "a learning permanent must beat storing nothing"


def test_vanilla_reports_identical_perm_and_full_values():
    """A single critic has no separately-retained component — the contrast Thm 7 draws."""
    van = PPOVanilla(17, 6, _cfg(), torch.device("cpu"))
    obs = np.random.randn(16, 17).astype(np.float32)
    v_perm, v_trans = van.get_value(obs)
    assert np.allclose(v_trans, 0.0)
    assert np.allclose(v_perm, v_perm + v_trans)
