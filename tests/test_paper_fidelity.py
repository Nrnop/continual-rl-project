"""The three algorithmic conditions from Anand & Precup (2023) that the port previously violated.

Each test names the result it enforces, so a future refactor that breaks one fails loudly rather
than silently reintroducing a deviation.
"""
import copy
import numpy as np
import torch

from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_vanilla import PPOVanilla
from src_continuous_control.models.critic import SplitCritic, VanillaCritic
from src_continuous_control.utils.metrics import JumpstartTracker, RetentionProbe


def _cfg(**kw):
    c = dict(hidden_sizes=[32, 32], lr_actor=3e-4, adam_eps=1e-5, num_envs=2, n_steps=8,
             gamma=0.99, gae_lambda=0.95, clip_coef=0.2, epochs=1, minibatch_size=8,
             ent_coef=0.0, max_grad_norm=0.5, target_kl=None, normalize_advantage=True,
             lr_trans=3e-4, lr_perm=1e-5, perm_optimizer="sgd", decay=0.75, k=2,
             consolidation_epochs=1, consolidation_buffer_size=64, on_switch="consolidate")
    c.update(kw)
    return c


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
def test_consolidation_target_is_the_full_acting_value():
    """Eq. (4) regresses V^(P) onto V^(PT) = old_V_perm + V_trans, i.e. keep = 1."""
    agent = PPOPT(17, 6, _cfg(), torch.device("cpu"))
    assert agent.value_preserving_consolidation is False

    # Give the transient signal and drive the permanent hard enough to see the direction of travel.
    with torch.no_grad():
        for p in agent.critic.trans[-1].parameters():
            p.add_(torch.randn_like(p) * 2.0)
    agent.perm_optim = torch.optim.Adam(agent.critic.perm.parameters(), lr=1e-2)

    states = np.random.randn(256, 17).astype(np.float32)
    with torch.no_grad():
        s_t = torch.as_tensor(states)
        old_vp, vt = agent.critic(s_t)
    agent.consolidation_buffer.add_batch(states, old_vp.numpy())
    agent.consolidation_epochs = 200
    agent._consolidate()

    with torch.no_grad():
        new_vp, _ = agent.critic(s_t)
    moved, wanted = (new_vp - old_vp), vt
    # The permanent should have moved toward the FULL transient, not toward (1-decay)*transient.
    frac = float((moved * wanted).sum() / (wanted * wanted).sum())
    assert frac > 0.5, f"permanent absorbed only {frac:.2f} of V_trans; expected -> 1.0 (keep=1)"


def test_value_preserving_flag_restores_the_old_target():
    """The pre-paper keep=(1-decay) behaviour stays reachable for reproducing earlier runs."""
    agent = PPOPT(17, 6, _cfg(value_preserving_consolidation=True), torch.device("cpu"))
    assert agent.value_preserving_consolidation is True


# ------------------------------------------- Sec. 6.1 / C.3: parameter-matched PT-0.5x
def test_critic_hidden_sizes_shrinks_the_critic_but_not_the_actor():
    """PT-0.5x needs a narrower critic while the actor stays identical across agents."""
    cfg = _cfg(hidden_sizes=[64, 64], critic_hidden_sizes=[43, 43])
    pt = PPOPT(17, 6, cfg, torch.device("cpu"))
    van = PPOVanilla(17, 6, dict(cfg, critic_hidden_sizes=[64, 64]), torch.device("cpu"))

    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(pt.actor) == n(van.actor), "the actor must be identical across agents"
    ratio = n(pt.critic) / n(van.critic)
    assert 0.9 < ratio < 1.1, f"PT critic is {ratio:.2f}x vanilla's; expected ~1.0 (parameter parity)"


def test_default_critic_width_follows_hidden_sizes():
    """Without critic_hidden_sizes nothing changes, so existing configs behave as before."""
    pt = PPOPT(17, 6, _cfg(hidden_sizes=[64, 64]), torch.device("cpu"))
    ref = SplitCritic(17, hidden_sizes=(64, 64))
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(pt.critic) == n(ref)


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
    agent = PPOPT(17, 6, _cfg(), torch.device("cpu"))

    def actor_grad_after_clip(critic_grad_scale):
        for p in agent.actor.parameters():
            p.grad = torch.ones_like(p) * 0.1
        for p in agent._critic_parameters():
            p.grad = torch.ones_like(p) * critic_grad_scale
        agent._clip_grads()
        return torch.cat([p.grad.flatten() for p in agent.actor.parameters()]).norm().item()

    small = actor_grad_after_clip(0.01)
    huge = actor_grad_after_clip(100.0)
    assert np.isclose(small, huge, rtol=1e-5), (
        f"actor gradient changed with the critic's magnitude ({small:.5f} vs {huge:.5f}) — "
        "the clip is still joint")

    # ...and the old behaviour stays reachable for reproducing earlier runs.
    joint = PPOPT(17, 6, _cfg(joint_grad_clip=True), torch.device("cpu"))
    def joint_actor_norm(scale):
        for p in joint.actor.parameters():
            p.grad = torch.ones_like(p) * 0.1
        for p in joint._critic_parameters():
            p.grad = torch.ones_like(p) * scale
        joint._clip_grads()
        return torch.cat([p.grad.flatten() for p in joint.actor.parameters()]).norm().item()
    assert not np.isclose(joint_actor_norm(0.01), joint_actor_norm(100.0), rtol=1e-5)


def test_robbins_monro_alpha_p_decays_and_defaults_off():
    """Theorem 5 requires a DECREASING alpha_P; a constant one makes theta_P track, not average."""
    off = PPOPT(17, 6, _cfg(), torch.device("cpu"))
    assert off.rm_power == 0.0
    lr0 = off.perm_optim.param_groups[0]["lr"]
    for _ in range(5):
        off._apply_robbins_monro_alpha_p()
    assert off.perm_optim.param_groups[0]["lr"] == lr0, "default must stay constant"

    on = PPOPT(17, 6, _cfg(alpha_p_rm_power=0.6, lr_perm=1e-2), torch.device("cpu"))
    base = on.perm_optim.param_groups[0]["lr"]
    seen = []
    for _ in range(4):
        on._apply_robbins_monro_alpha_p()
        seen.append(on.perm_optim.param_groups[0]["lr"])
    assert all(seen[i] > seen[i + 1] for i in range(len(seen) - 1)), f"alpha_P not decreasing: {seen}"
    assert np.isclose(seen[0], base / 2 ** 0.6)
    assert on.last_alpha_p == seen[-1]


def test_absorbed_frac_detects_an_inert_permanent():
    """The shipped sgd/1e-5 must report absorbed_frac ~ 0; a real transfer must report ~1.

    This diagnostic is the one that was missing for the whole project: with an inert permanent,
    returns and critic_loss both look healthy while PT has no slow timescale at all.
    """
    states = np.random.randn(2048, 17).astype(np.float32)

    def absorbed(lr, opt):
        agent = PPOPT(17, 6, _cfg(lr_perm=lr, perm_optimizer=opt, k=1,
                                  consolidation_buffer_size=4096), torch.device("cpu"))
        with torch.no_grad():                       # give theta_T a learned magnitude
            for p in agent.critic.trans[-1].parameters():
                p.add_(torch.randn_like(p) * 3.0)
            old_vp, _ = agent.critic(torch.as_tensor(states))
        agent.consolidation_buffer.add_batch(states, old_vp.numpy())
        agent._consolidate()
        return agent.last_absorbed_frac

    inert = absorbed(1e-5, "sgd")
    working = absorbed(1e-3, "adam")
    assert inert is not None and working is not None
    assert inert < 0.05, f"sgd/1e-5 should be inert, got absorbed_frac={inert:.4f}"
    assert working > 0.5, f"adam/1e-3 should transfer most of it, got {working:.4f}"


def test_absorbed_frac_is_also_measured_off_distribution():
    """With a holdout, absorbed_frac must be reported on states the regression never saw.

    absorbed_frac alone is in-distribution: a permanent that memorised the consolidation buffer
    reports a healthy number while being wrong on the NEW states the next rollout bootstraps from.
    Offline, the target is fittable to 3.2% train error while held-out error floors at 38-44%
    (FINDINGS 6.3) — this is the in-situ version of that measurement.
    """
    states = np.random.randn(2048, 17).astype(np.float32)
    agent = PPOPT(17, 6, _cfg(lr_perm=1e-3, perm_optimizer="adam", k=1,
                              consolidation_buffer_size=4096,
                              consolidation_holdout_frac=0.25), torch.device("cpu"))
    with torch.no_grad():
        for p in agent.critic.trans[-1].parameters():
            p.add_(torch.randn_like(p) * 3.0)
        old_vp, _ = agent.critic(torch.as_tensor(states))
    agent.consolidation_buffer.add_batch(states, old_vp.numpy())
    agent._consolidate()

    assert agent.last_absorbed_frac is not None
    assert agent.last_absorbed_frac_holdout is not None, \
        "holdout absorbed_frac must be measured when consolidation_holdout_frac > 0"
    assert agent.last_absorbed_align_holdout is not None

    # ...and it must stay None when no holdout is requested, so the default path is unchanged.
    plain = PPOPT(17, 6, _cfg(lr_perm=1e-3, perm_optimizer="adam", k=1,
                              consolidation_buffer_size=4096), torch.device("cpu"))
    with torch.no_grad():   # theta_T is zero-initialised; with V_trans == 0 there is nothing to
        for p in plain.critic.trans[-1].parameters():   # absorb and the diagnostic is undefined.
            p.add_(torch.randn_like(p) * 3.0)
        plain_vp, _ = plain.critic(torch.as_tensor(states))
    plain.consolidation_buffer.add_batch(states, plain_vp.numpy())
    plain._consolidate()
    assert plain.last_absorbed_frac is not None
    assert plain.last_absorbed_frac_holdout is None


def test_boundary_tracker_rejects_a_sub_update_window():
    """A window shorter than 2 updates reports drop=0 by construction — must fail loudly.

    This is the bug that produced 0.00 at boundaries 1-3 for all 40 runs of the 2026-08-03 sweep:
    the window was n_steps*5 = 1280 env steps against a 2048-step batch, so the tracker finalised
    on its first post-switch sample.
    """
    from src_continuous_control.utils.metrics import BoundaryReturnTracker
    import pytest
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
