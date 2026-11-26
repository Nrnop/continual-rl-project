"""Tests that the PT agent satisfies the algorithmic conditions of Anand & Precup (2023).

Each test names the result it checks (Theorem 1, Eq. 4, Theorem 5, the PT-0.5x parameter
matching), plus the diagnostics used to verify a run: absorbed_frac, the retention probe and the
boundary tracker. `rho` plays the role of the paper's `keep`; rho = 1 is the paper's keep = 1.
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


# Theorem 1: V^(T)_0 = 0
def test_transient_is_zero_at_initialisation():
    """The critic transient is the zero function at init (Theorem 1: V^(T)_0 = 0)."""
    torch.manual_seed(0)
    critic = SplitCritic(17, hidden_sizes=(64, 64))
    probe = torch.randn(512, 17)
    with torch.no_grad():
        v_perm, v_trans = critic(probe)
    assert torch.all(v_trans == 0), "theta_T must start at the zero function"
    assert v_perm.abs().mean() > 0, "theta_P keeps the ordinary value-head init"
    assert torch.allclose(v_perm + v_trans, v_perm)


def test_the_split_actor_also_starts_at_the_zero_function():
    """mu_T(s) = 0 at init, so the composed policy equals pi_P and KL to the prior is 0."""
    torch.manual_seed(0)
    agent = _agent()
    probe = torch.randn(256, 17)
    with torch.no_grad():
        mu_trans = agent.actor.trans_mean(probe)
        mu_perm = agent.actor.perm_mean(probe)
    assert torch.all(mu_trans == 0), "mu_T must start at the zero function"
    assert mu_perm.abs().mean() > 0, "mu_P keeps the ordinary policy-head init"
    assert torch.allclose(agent.actor.act_deterministic(probe), mu_perm)
    assert float(agent.actor.kl_to_prior(probe).detach().max()) == 0.0


def test_perm_zero_init_makes_the_pt_loss_identical_to_vanilla():
    """With perm_zero_init (V^(P) = 0), PT's semi-gradient critic loss equals vanilla's.

    With a random frozen V_perm the transient must instead learn R - V_perm.
    """
    torch.manual_seed(0)
    probe = torch.randn(256, 17)
    returns = torch.randn(256)

    zero = SplitCritic(17, hidden_sizes=(64, 64), perm_zero_init=True)
    with torch.no_grad():
        v_perm, v_trans = zero(probe)
    assert torch.all(v_perm == 0), "perm_zero_init must give V^(P) == 0 exactly"

    pt_loss = 0.5 * ((v_perm.detach() + v_trans - returns) ** 2).mean()
    vanilla_equivalent = 0.5 * ((v_trans - returns) ** 2).mean()
    assert torch.allclose(pt_loss, vanilla_equivalent)

    # The default permanent init stays random.
    rand = SplitCritic(17, hidden_sizes=(64, 64))
    with torch.no_grad():
        vp, _ = rand(probe)
    assert vp.abs().mean() > 0


def test_theorem1_second_condition_can_now_be_met():
    """critic_zero_init starts vanilla at PT's initial value (Theorem 1: V^(TD)_0 = V^(P)).

    With perm_zero_init PT's acting value starts at 0, so the baseline must start at 0 too.
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
    assert torch.allclose(v_matched, p + t)


def test_perm_zero_init_still_lets_the_permanent_learn():
    """With perm_zero_init the permanent can still be trained away from zero."""
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
    """The zero-initialised transient can be trained away from zero."""
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


# Eq. (4) / Alg. 4 line 15: consolidation target
def _drive_consolidation(agent, states, trans_scale=2.0):
    """Give both transients a non-zero output, then run one consolidation.

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
    """With rho = 1 the permanent regresses onto old_V_perm + V_trans (Eq. 4)."""
    torch.manual_seed(0)
    agent = _agent(rho=1.0, lr_perm=1e-2, perm_optimizer="adam", rm_power=0.0,
                   consolidation_epochs=200, consolidation_buffer_size=512)
    states = np.random.randn(256, 17).astype(np.float32)
    old_vp, old_vt, new_vp = _drive_consolidation(agent, states)

    moved, wanted = (new_vp - old_vp), old_vt
    frac = float((moved * wanted).sum() / (wanted * wanted).sum())
    assert frac > 0.5, f"permanent absorbed only {frac:.2f} of V_trans; expected -> 1.0 (rho=1)"


def test_consolidation_absorbs_exactly_rho_of_the_transient():
    """With rho < 1 the permanent regresses onto P + rho * T."""
    torch.manual_seed(0)
    agent = _agent(rho=0.5, lr_perm=1e-2, perm_optimizer="adam", rm_power=0.0,
                   consolidation_epochs=200, consolidation_buffer_size=512)
    states = np.random.randn(256, 17).astype(np.float32)
    old_vp, old_vt, new_vp = _drive_consolidation(agent, states)

    moved, wanted = (new_vp - old_vp), old_vt
    frac = float((moved * wanted).sum() / (wanted * wanted).sum())
    assert 0.3 < frac < 0.7, f"expected the permanent to absorb ~rho=0.5 of V_trans, got {frac:.2f}"
    # last_absorbed_align normalises by rho * V_trans, so it should read ~1.
    assert agent.last_absorbed_align == pytest.approx(1.0, abs=0.25)


def test_the_decay_is_the_other_half_of_the_same_transfer():
    """Consolidation scales both transients by exactly (1 - rho); decay_rho defaults to rho.

    A separate decay_rho decouples absorb and decay and is kept only as a control.
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
    # Output-layer scaling makes the retained fraction exact.
    assert torch.allclose(vt_after, 0.75 * vt_before, atol=1e-5)
    assert torch.allclose(mu_t_after, 0.75 * mu_t_before, atol=1e-5)

    # The decoupled control is still configurable and validated.
    decoupled = _agent(rho=0.25, decay_rho=0.9)
    assert decoupled.rho == 0.25 and decoupled.decay_rho == 0.9
    with pytest.raises(ValueError):
        _agent(decay_rho=1.5)


# Sec. 6.1 / C.3: parameter-matched PT-0.5x
def test_critic_hidden_sizes_shrinks_the_critic_but_not_the_actor():
    """critic_hidden_sizes narrows the critic and leaves the actor unchanged."""
    torch.manual_seed(0)
    wide = _agent(hidden_sizes=[64, 64])
    narrow = _agent(hidden_sizes=[64, 64], critic_hidden_sizes=[43, 43],
                    critic_trans_hidden_sizes=[43, 43])
    assert _n_params(wide.actor) == _n_params(narrow.actor), \
        "critic_hidden_sizes must not touch the actor"
    assert _n_params(narrow.critic) < _n_params(wide.critic)


def test_parameter_parity_is_reachable_for_a_split_actor_and_critic():
    """Two [43, 43] nets match one [64, 64] net within 10%, for both actor and critic.

    This is the PT-0.5x construction of Sec. 6.1, applied to the split actor as well.
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
    """At equal width PT has just under 2x the parameters of vanilla, and never more."""
    torch.manual_seed(0)
    pt = _agent(hidden_sizes=[64, 64], actor_trans_hidden_sizes=[64, 64],
                critic_trans_hidden_sizes=[64, 64])
    van = PPOVanilla(17, 6, _cfg(hidden_sizes=[64, 64]), torch.device("cpu"))
    # log_std (6 params) is shared, not duplicated, so the ratio is just under 2.
    assert 1.9 < _n_params(pt.actor) / _n_params(van.actor) <= 2.0
    assert 1.9 < _n_params(pt.critic) / _n_params(van.critic) <= 2.0


def test_default_critic_width_follows_hidden_sizes():
    """Without critic_hidden_sizes the critic width follows hidden_sizes."""
    torch.manual_seed(0)
    pt = _agent(hidden_sizes=[64, 64], critic_trans_hidden_sizes=[64, 64])
    ref = SplitCritic(17, hidden_sizes=(64, 64), trans_hidden_sizes=(64, 64))
    assert _n_params(pt.critic) == _n_params(ref)


# Theorems 6/7/8: instrumentation
def test_jumpstart_tracker_windows_the_post_switch_returns():
    """JumpstartTracker reports first, end, gain and mean over the post-switch window."""
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
    """RetentionProbe scores components against the snapshot of a task only when it is inactive."""
    probe = np.random.randn(32, 17).astype(np.float32)
    rp = RetentionProbe()
    # No snapshots yet -> nothing to report.
    assert rp.measure(1, {"full": lambda s: np.zeros(len(s))}, probe) == {}

    # v_1 == 1.0 everywhere (the converged acting value at the end of task 1).
    rp.snapshot(1, lambda s: np.ones(len(s)), probe)
    # Task 1 is still active -> still nothing.
    assert rp.measure(1, {"perm": lambda s: np.zeros(len(s))}, probe) == {}
    # On task -1, both components are scored against the same reference v_1.
    out = rp.measure(-1, {"perm": lambda s: np.zeros(len(s)),
                          "full": lambda s: np.full(len(s), 3.0)}, probe)
    assert np.isclose(out["perm"], 1.0) and np.isclose(out["full"], 4.0)


def test_output_mode_decay_is_exact_and_params_mode_is_not():
    """mode='output' scales V_trans by exactly lambda (Alg. 2 line 9); mode='params' over-decays.

    mode='params' is the reference code's `p.data *= decay` over every parameter, which on a
    nonlinear net removes more than the intended fraction.
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
        if decay == 0.75:
            # The overshoot depends on the transient's magnitude; on this probe params mode
            # retains ~0.59 against an intended 0.75.
            assert ratio < decay - 0.10, (
                f"expected params mode to over-decay materially at decay=0.75 "
                f"(exact would be 0.75); got {ratio:.3f}")


def test_actor_and_critic_are_clipped_separately():
    """The actor's clipped gradient does not depend on the critic's gradient magnitude.

    A joint clip_grad_norm_ would let the critic (which differs between agents) change the actor's
    effective step size.
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
        f"actor gradient changed with the critic's magnitude ({small:.5f} vs {huge:.5f}) - "
        "the clip is still joint")

    # joint_grad_clip=True restores the joint clip.
    joint = _agent(joint_grad_clip=True)
    assert not np.isclose(actor_grad_after_clip(joint, 0.01),
                          actor_grad_after_clip(joint, 100.0), rtol=1e-5)


def test_robbins_monro_alpha_p_decays_and_is_on_by_default():
    """alpha_P decays as a Robbins-Monro schedule (Theorem 5), by default, for actor and critic.

    The default power is 0.6. rm_power = 0 gives a constant alpha_P.
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

    # With the power set to 0, alpha_P stays constant.
    off = _agent(lr_perm=1e-2, rm_power=0.0, rm_power_actor=0.0)
    lr0 = off.perm_optim.param_groups[0]["lr"]
    for _ in range(5):
        off._n_consolidations += 1
        off._set_next_permanent_lrs()
    assert off.perm_optim.param_groups[0]["lr"] == lr0

    # The legacy key alpha_p_rm_power is still accepted.
    alias = _agent(alpha_p_rm_power=0.3)
    assert alias.rm_power == 0.3


def test_absorbed_frac_detects_an_inert_permanent():
    """absorbed_frac is ~0 for an inert permanent (sgd/1e-5) and high for a working one.

    With an inert permanent, returns and critic_loss can look healthy while PT has no slow
    timescale, so this diagnostic must detect it.
    """
    states = np.random.randn(2048, 17).astype(np.float32)

    def absorbed(lr, opt):
        torch.manual_seed(0)
        # 3 epochs: the actor's 6-dim target needs more than one pass to transfer most of it.
        agent = _agent(lr_perm=lr, lr_perm_actor=lr, perm_optimizer=opt, k=1, rm_power=0.0,
                       consolidation_epochs=3, consolidation_buffer_size=4096)
        _drive_consolidation(agent, states, trans_scale=3.0)
        return agent.last_absorbed_frac, agent.last_actor_absorbed_frac

    inert_c, inert_a = absorbed(1e-5, "sgd")
    working_c, working_a = absorbed(1e-3, "adam")
    assert None not in (inert_c, inert_a, working_c, working_a)
    assert inert_c < 0.05, f"sgd/1e-5 should be inert, got absorbed_frac={inert_c:.4f}"
    assert working_c > 0.5, f"adam/1e-3 should transfer most of it, got {working_c:.4f}"
    # The actor's absorbed_frac is the one to check before trusting a run.
    assert inert_a < 0.05, f"actor permanent should be inert too, got {inert_a:.4f}"
    assert working_a > 0.5, f"actor permanent should transfer, got {working_a:.4f}"


def test_absorbed_frac_is_also_measured_off_distribution():
    """With consolidation_holdout_frac > 0, absorbed_frac is also reported on held-out states.

    The in-sample absorbed_frac cannot distinguish a permanent that generalises from one that
    memorised the buffer.
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
    assert agent.last_actor_absorbed_frac_holdout is not None
    assert agent.last_actor_absorbed_align_holdout is not None
    # Both values are saved in the consolidation record.
    record = agent.consolidation_records[-1]
    assert record["absorbed_frac_holdout"] == agent.last_absorbed_frac_holdout
    assert record["actor_absorbed_frac_holdout"] == agent.last_actor_absorbed_frac_holdout

    # Without a holdout the holdout value is None.
    torch.manual_seed(0)
    plain = _agent(lr_perm=1e-3, perm_optimizer="adam", k=1, rm_power=0.0,
                   consolidation_buffer_size=4096)
    _drive_consolidation(plain, states, trans_scale=3.0)
    assert plain.last_absorbed_frac is not None
    assert plain.last_absorbed_frac_holdout is None


def test_the_holdout_split_actually_withholds_states():
    """The held-out states are excluded from the consolidation regression."""
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
    assert len(trained_on) < len(states), "every state was trained on - nothing was held out"
    # One epoch over 75% of 256 states.
    assert 0.7 * len(states) <= len(trained_on) <= 0.8 * len(states)


def test_boundary_tracker_rejects_a_sub_update_window():
    """BoundaryReturnTracker raises ValueError if its window is shorter than one update.

    Such a window finalises on its first post-switch sample and always reports drop = 0.
    """
    from src_continuous_control.utils.metrics import BoundaryReturnTracker
    batch = 2048
    with pytest.raises(ValueError, match="finalise on its first sample"):
        BoundaryReturnTracker(post_window_steps=1280, min_useful_steps=batch)
    # A 5-update window is fine.
    BoundaryReturnTracker(post_window_steps=5 * batch, min_useful_steps=batch)


def test_boundary_tracker_captures_a_real_trough():
    """With a multi-update window the tracker finds a trough several updates after the switch."""
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
    """A permanent frozen at init gets a large retention error, not a perfect 0.

    Components are scored against the acting value v_i, not against their own earlier snapshot.
    """
    probe = np.random.randn(32, 17).astype(np.float32)
    rp = RetentionProbe()
    frozen = lambda s: np.zeros(len(s))          # a permanent stuck at its init
    rp.snapshot(1, lambda s: np.ones(len(s)), probe)   # v_1 = 1.0
    out = rp.measure(-1, {"perm": frozen}, probe)
    assert out["perm"] > 0.5, "a frozen permanent must report a large retention error"


def test_retention_control_baselines_expose_an_inert_permanent():
    """The perm_init baseline exposes an inert permanent.

    On a sign-flip task pair a permanent frozen at zero beats the adapted estimate automatically,
    so `mse_perm < mse_full` alone does not confirm Theorem 7.
    """
    probe = np.random.randn(64, 17).astype(np.float32)
    v_i = np.linspace(-5, 5, 64).astype(np.float32)          # task i's converged values
    perm_at_init = np.zeros(64, dtype=np.float32)

    rp = RetentionProbe()
    rp.set_baseline("perm_init", perm_at_init)
    rp.set_baseline("zero", np.zeros(64, dtype=np.float32))
    rp.snapshot(1, lambda s: v_i, probe)

    # (a) Inert permanent at its init: beats `full` (which flipped sign) but equals perm_init.
    out = rp.measure(-1, {"perm": lambda s: perm_at_init, "full": lambda s: -v_i}, probe)
    assert out["perm"] < out["full"], "sanity: the sign-flip artifact reproduces"
    assert np.isclose(out["perm"], out["perm_init"]), \
        "an inert permanent must be indistinguishable from its own initialisation"

    # (b) Permanent that moved toward v_i: beats both baselines.
    out = rp.measure(-1, {"perm": lambda s: 0.7 * v_i, "full": lambda s: -v_i}, probe)
    assert out["perm"] < out["perm_init"], "a learning permanent must beat its initialisation"
    assert out["perm"] < out["zero"], "a learning permanent must beat storing nothing"


def test_vanilla_reports_identical_perm_and_full_values():
    """Vanilla reports V_trans = 0, so its perm and full values are identical."""
    van = PPOVanilla(17, 6, _cfg(), torch.device("cpu"))
    obs = np.random.randn(16, 17).astype(np.float32)
    v_perm, v_trans = van.get_value(obs)
    assert np.allclose(v_trans, 0.0)
    assert np.allclose(v_perm, v_perm + v_trans)
