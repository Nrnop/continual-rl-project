"""The family's 24 config overlays: what a run would ACTUALLY get, not what the file says.

MULTIENV_TASK.md section 9 asks for these to be "verified by loading the merged config rather than
reading the file", and that wording is earned. `test_config_encoding.py` already checks the bytes
on disk; what it cannot check is the result of the four-layer merge
(default.yaml <- ppo_<agent>.yaml <- overlay <- CLI), which is the thing training actually runs on.

Two failures this project has paid for live here:

  * a config key one agent read and another ignored, which handed one arm 3x the exploration of
    the arms it was compared against and produced a spectacular fake result
        -> `test_arms_agree_on_every_environment_key`, `test_sigma_is_matched_across_arms`
  * `pt`'s widths not transferring between environments — 0.99x at HalfCheetah's dimensions and
    0.931x at cartpole's, a 7% capacity handicap with no config key mentioning it
        -> `test_pt_hits_parameter_parity_on_constructed_modules`
"""
import argparse
import contextlib
import io
import os

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control", reason="the dm_control family needs dm_control")
pytest.importorskip("shimmy", reason="the dm_control family needs shimmy")

from src_continuous_control.agents import AGENTS                            # noqa: E402
from src_continuous_control.envs.dm_control_drift import SPECS              # noqa: E402
from src_continuous_control.scripts.make_multienv_configs import (          # noqa: E402
    ARMS,
    ENVIRONMENTS,
    WIDTHS,
    stem_for,
)
from src_continuous_control.train import build_config                       # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
ARM_AGENT = {"vanilla": "vanilla", "ewc": "ewc", "pt": "pt", "pt_frozen": "pt"}
COMPARABLE_ARMS = ("vanilla", "ewc", "pt")

# Keys that define the ENVIRONMENT and the protocol. Every arm must agree on all of them; an arm
# that differs here is running a different experiment from the ones it is bar-charted against.
ENVIRONMENT_KEYS = ("env_mode", "dmc_env", "drift_schedule", "drift_targets", "task_multipliers",
                    "max_episode_steps", "switch", "total_steps", "gamma", "gae_lambda",
                    "n_steps", "num_envs", "epochs", "minibatch_size", "clip_coef",
                    "normalize_obs", "normalize_reward", "normalizer_freeze_after",
                    "anneal_lr", "freeze_log_std", "log_std_init", "transfer_eval_episodes")

CASES = [(env, arm) for env in ENVIRONMENTS for arm in ARMS]
CASE_IDS = ["%s-%s" % (e, a) for e, a in CASES]


def merged(env, arm):
    """The config a real run of this arm would get — the same merge train.py performs."""
    return build_config(argparse.Namespace(agent=ARM_AGENT[arm], config=stem_for(env, arm)))


def build_agent(env, arm, cfg=None):
    cfg = cfg or merged(env, arm)
    spec = SPECS[env]
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):     # the PT constructor prints a summary
        return AGENTS[ARM_AGENT[arm]](spec.obs_dim, spec.act_dim, cfg, torch.device("cpu"))


# ---------------------------------------------------------------------------
# The files exist, are ASCII, and match their generator
# ---------------------------------------------------------------------------
def test_every_environment_has_every_arm():
    """6 environments x 4 arms. A missing overlay would silently fall back to HalfCheetah's keys."""
    missing = [stem_for(e, a) + ".yaml" for e, a in CASES
               if not os.path.exists(os.path.join(CFG_DIR, stem_for(e, a) + ".yaml"))]
    assert not missing, f"missing config(s): {missing}"
    assert len(CASES) == 28


@pytest.mark.parametrize("env,arm", CASES, ids=CASE_IDS)
def test_config_is_pure_ascii(env, arm):
    """Stricter than the repo-wide UTF-8 rule, and deliberately so.

    These files are generated, so there is no reason for a typographic character to appear in one,
    and ASCII is the one encoding no platform default can corrupt. Twelve configs written with the
    platform default were cp1252 and blocked a 160-run job on the rented box.
    """
    path = os.path.join(CFG_DIR, stem_for(env, arm) + ".yaml")
    raw = open(path, "rb").read()
    raw.decode("ascii")          # raises if a non-ASCII byte crept in


def test_configs_match_their_generator():
    """A hand-edit to one of 24 files breaks the symmetry the family depends on."""
    from src_continuous_control.scripts.make_multienv_configs import render
    stale = [stem_for(e, a) for e, a in CASES
             if io.open(os.path.join(CFG_DIR, stem_for(e, a) + ".yaml"),
                        encoding="ascii").read() != render(e, a)]
    assert not stale, (f"config(s) no longer match scripts/make_multienv_configs.py: {stale}; "
                       f"edit the generator and regenerate rather than editing the YAML")


# ---------------------------------------------------------------------------
# The merged config selects the environment it claims to
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env,arm", CASES, ids=CASE_IDS)
def test_merged_config_selects_the_right_environment(env, arm):
    cfg = merged(env, arm)
    spec = SPECS[env]
    assert cfg["env_mode"] == "dmc"
    assert cfg["dmc_env"] == env
    assert cfg["drift_schedule"] == "step"
    assert tuple(cfg["drift_targets"]) == spec.default_targets
    assert cfg["max_episode_steps"] == 1000


@pytest.mark.parametrize("env,arm", CASES, ids=CASE_IDS)
def test_task_sequence_cycles_and_revisits(env, arm):
    """Backward transfer is undefined on a sequence that never revisits a task."""
    mults = list(merged(env, arm)["task_multipliers"])
    assert len(mults) == 5
    assert mults[1] == mults[3] and mults[2] == mults[4]
    assert len(set(mults)) > 1, "a constant sequence would never change the physics"


# ---------------------------------------------------------------------------
# The arms differ ONLY in the agent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_arms_agree_on_every_environment_key(env):
    """failure mode #3: a key one arm reads and another ignores.

    The comparison is only apples-to-apples if the three arms differ in the AGENT and in the widths
    parity forces, and in nothing else at all.
    """
    reference = merged(env, "vanilla")
    for arm in ("ewc", "pt"):
        cfg = merged(env, arm)
        for key in ENVIRONMENT_KEYS:
            assert cfg.get(key) == reference.get(key), (
                f"{env}: {arm} and vanilla disagree on {key!r} "
                f"({cfg.get(key)!r} vs {reference.get(key)!r})")


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_sigma_is_matched_across_arms(env):
    """Asserted on LIVE TENSORS, not on the config key — that distinction is the whole point.

    On HalfCheetah, learned sigma collapses at different rates per arm and costs a factor of three
    even with no task switches at all. The family runs standard PPO exploration, which is only a
    fair comparison if all three arms share the schedule.
    """
    sigmas, learned = {}, {}
    for arm in COMPARABLE_ARMS:
        actor = build_agent(env, arm).actor
        sigmas[arm] = float(torch.exp(actor.log_std.detach()).mean())
        learned[arm] = bool(actor.log_std.requires_grad)
    assert len(set(learned.values())) == 1, f"{env}: exploration SCHEDULE differs: {learned}"
    assert all(learned.values()), f"{env}: the family runs standard PPO — log_std must be learned"
    assert np.allclose(list(sigmas.values()), sigmas["vanilla"], atol=1e-6), \
        f"{env}: exploration LEVEL differs at initialisation: {sigmas}"


# ---------------------------------------------------------------------------
# Parameter parity, measured on constructed modules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_pt_hits_parameter_parity_on_constructed_modules(env):
    """Counted off the real networks, never computed from the config keys.

    `pt` carries four networks to the baseline's two, so parity depends on the input and output
    dimensions — which differ in all six environments. Reusing HalfCheetah's [51,51]/[32,32] would
    land at 0.931x on cartpole and 1.0155x on the walker pair, and no config key would say so.
    """
    def total(arm):
        agent = build_agent(env, arm)
        return (sum(p.numel() for p in agent.actor.parameters())
                + sum(p.numel() for p in agent.critic.parameters()))

    base, pt_total = total("vanilla"), total("pt")
    ratio = pt_total / base
    assert abs(ratio - 1.0) <= 0.005, (
        f"{env}: pt is at {ratio:.4f}x the baseline's parameters ({pt_total:,} vs {base:,}); "
        f"re-derive its widths for obs {SPECS[env].obs_dim} / act {SPECS[env].act_dim}")
    # And the numbers recorded in the generator's comments must be the real ones, or the write-up
    # would quote figures nobody measured.
    assert pt_total == WIDTHS[env]["pt_total"]
    assert base == WIDTHS[env]["base"]


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_ewc_matches_the_baseline_architecture(env):
    """EWC is a penalty on the baseline network, not a different architecture."""
    assert merged(env, "ewc")["hidden_sizes"] == merged(env, "vanilla")["hidden_sizes"]


# ---------------------------------------------------------------------------
# The ablation arm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_frozen_arm_matches_pt_everywhere_except_the_permanent(env):
    """The ablation is only an ablation if the two arms differ in exactly one thing.

    An ablation arm that silently ran a different environment, or different widths, would be
    measuring a different benchmark from the arm it is bar-charted against.
    """
    live, frozen = merged(env, "pt"), merged(env, "pt_frozen")
    for key in ENVIRONMENT_KEYS + ("hidden_sizes", "actor_trans_hidden_sizes",
                                   "critic_hidden_sizes", "critic_trans_hidden_sizes",
                                   "rho", "k", "kl_prior_coef"):
        assert frozen.get(key) == live.get(key), f"{env}: pt_frozen differs from pt on {key!r}"
    assert frozen["lr_perm"] == 0.0
    assert frozen["lr_perm_actor"] == 0.0
    assert live["lr_perm"] > 0.0, "the live arm's permanent must actually learn"


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_frozen_arm_still_has_a_transient_and_a_decay(env):
    """`lr_perm = 0` freezes the permanent's LEARNING and nothing else.

    Failure mode #1 in CLAUDE.md is a control that was not actually off. The frozen arm must still
    carry the split, the decay and the KL anchor, or the ablation decomposes the wrong thing.
    """
    cfg = merged(env, "pt_frozen")
    assert cfg["rho"] > 0.0, "the transient decay must still run"
    assert cfg["k"] > 0, "the consolidation cadence must still run"
    agent = build_agent(env, "pt_frozen")
    assert getattr(agent.actor, "trans_mean", None) is not None, "the split actor must survive"
    assert getattr(agent.critic, "trans", None) is not None, "the split critic must survive"


# ---------------------------------------------------------------------------
# The walker pair, once more, at the config level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", list(ARMS))
def test_walker_pair_configs_differ_only_in_the_environment(arm):
    """The study's one controlled comparison must be controlled in the configs too."""
    stand, walk = merged("walker-stand", arm), merged("walker-walk", arm)
    assert stand["dmc_env"] != walk["dmc_env"]
    for key in ENVIRONMENT_KEYS + ("hidden_sizes",):
        if key == "dmc_env":
            continue
        assert stand.get(key) == walk.get(key), f"walker pair disagrees on {key!r} for arm {arm}"


# ---------------------------------------------------------------------------
# The two boundary-free drift settings
# ---------------------------------------------------------------------------
from src_continuous_control.scripts.make_multienv_configs import (   # noqa: E402
    DRIFT_ENVIRONMENTS,
    SETTINGS,
    drift_stem_for,
    render_drift,
)

DRIFT_CASES = [(s, e, a) for s in SETTINGS for e in DRIFT_ENVIRONMENTS for a in ARMS]
DRIFT_IDS = ["%s-%s-%s" % (s, e, a) for s, e, a in DRIFT_CASES]


def merged_drift(setting, env, arm):
    return build_config(argparse.Namespace(agent=ARM_AGENT[arm],
                                           config=drift_stem_for(env, arm, setting)))


@pytest.mark.parametrize("setting,env,arm", DRIFT_CASES, ids=DRIFT_IDS)
def test_drift_config_has_no_boundaries(setting, env, arm):
    """The whole point of these settings: the physics move every step and no task index exists.

    A drift config that silently kept `drift_schedule: step` would run the boundary benchmark under
    a drift filename -- and the log would say `SWITCH to task`, which nobody reads on run 200 of
    300.
    """
    cfg = merged_drift(setting, env, arm)
    assert cfg["drift_schedule"] == "sin"
    # FWT/BWT are indexed by task number, which does not exist here.
    assert cfg["transfer_eval_episodes"] == 0


@pytest.mark.parametrize("setting,env,arm", DRIFT_CASES, ids=DRIFT_IDS)
def test_drift_config_matches_drift_results_settings(setting, env, arm):
    """Amplitudes and periods are copied from DRIFT_RESULTS.md and must not drift.

    cartpole-swingup appears in both that study and this one, so identical settings make its cell
    here a free replication. A "tidied" amplitude would silently throw that away.
    """
    cfg = merged_drift(setting, env, arm)
    for key, expected in SETTINGS[setting]["keys"].items():
        assert cfg[key] == expected, f"{setting}/{env}/{arm}: {key} is {cfg[key]}, not {expected}"
    assert cfg["dmc_reload_tol"] == 0.005


@pytest.mark.parametrize("setting,env,arm", DRIFT_CASES, ids=DRIFT_IDS)
def test_drift_config_enables_the_ewc_timer(setting, env, arm):
    """Online EWC cannot run without boundaries unless a timer consolidates it.

    Without this it never accumulates a Fisher, the penalty is identically zero, and the EWC arm is
    vanilla PPO wearing a different name -- measured at p = 1.000 in Phase 1.
    """
    assert merged_drift(setting, env, arm)["ewc_consolidate_every"] == 10


@pytest.mark.parametrize("setting,env,arm", DRIFT_CASES, ids=DRIFT_IDS)
def test_drift_config_keeps_anneal_lr_off(setting, env, arm):
    """Decaying every learning rate toward zero is fatal where the point is continuous adaptation."""
    assert merged_drift(setting, env, arm)["anneal_lr"] is False


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("env", list(DRIFT_ENVIRONMENTS))
def test_drift_arms_agree_on_every_environment_key(setting, env):
    reference = merged_drift(setting, env, "vanilla")
    keys = ("env_mode", "dmc_env", "drift_schedule", "drift_targets", "drift_amplitude",
            "drift_period", "drift_amplitude2", "drift_period2", "max_episode_steps",
            "total_steps", "freeze_log_std", "log_std_init", "transfer_eval_episodes",
            "anneal_lr", "dmc_reload_tol")
    for arm in ("ewc", "pt"):
        cfg = merged_drift(setting, env, arm)
        for key in keys:
            assert cfg.get(key) == reference.get(key), (
                f"{setting}/{env}: {arm} and vanilla disagree on {key!r}")


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("env", list(DRIFT_ENVIRONMENTS))
def test_drift_pt_hits_parameter_parity(setting, env):
    """Re-derived widths must survive into the drift configs too, measured on real modules."""
    def total(arm):
        cfg = merged_drift(setting, env, arm)
        spec = SPECS[env]
        torch.manual_seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            a = AGENTS[ARM_AGENT[arm]](spec.obs_dim, spec.act_dim, cfg, torch.device("cpu"))
        return (sum(p.numel() for p in a.actor.parameters())
                + sum(p.numel() for p in a.critic.parameters()))
    ratio = total("pt") / total("vanilla")
    assert abs(ratio - 1.0) <= 0.005, f"{setting}/{env}: pt at {ratio:.4f}x the baseline"


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("env", list(DRIFT_ENVIRONMENTS))
def test_drift_targets_match_the_boundary_study(setting, env):
    """Same physics parameters as the boundary runs, or the two settings are not comparable."""
    assert (list(merged_drift(setting, env, "pt")["drift_targets"])
            == list(merged(env, "pt")["drift_targets"]))


def test_ball_in_cup_is_excluded_from_the_drift_settings():
    """Deliberate, and worth a test so it cannot be "fixed" back in without a decision.

    It saturated in the boundary study -- all three arms within 13 points of each other at 84% of
    the 1000 ceiling -- so it separated nothing, and it is one of the expensive rebuild
    environments.
    """
    assert "ball_in_cup-catch" not in DRIFT_ENVIRONMENTS
    assert len(DRIFT_ENVIRONMENTS) == 5


def test_lipschitz2_adds_a_fast_component_and_lipschitz1_does_not():
    """The two settings differ in exactly the thing they are meant to differ in."""
    one, two = SETTINGS["lipschitz1"]["keys"], SETTINGS["lipschitz2"]["keys"]
    assert one["drift_amplitude2"] == 0.0
    assert two["drift_amplitude2"] > 0.0
    assert one["drift_period"] == two["drift_period"]
    # The slow tide is smaller in Lipschitz2 so the two worlds cover a COMPARABLE total range --
    # comparable, not equal, and the difference is worth pinning rather than rounding away.
    assert two["drift_amplitude"] < one["drift_amplitude"]
    span_one = 2 * one["drift_amplitude"]                                    # 0.5 .. 1.5
    span_two = 2 * (two["drift_amplitude"] + two["drift_amplitude2"])        # 0.4 .. 1.6
    assert span_one == pytest.approx(1.0)
    assert span_two == pytest.approx(1.2)
    # LIPSCHITZ2 CARRIES 20% MORE TOTAL DRIFT. So part of any Lipschitz1 -> Lipschitz2 difference
    # is "more physics variation" rather than "a fast component was added". These are the numbers
    # DRIFT_RESULTS.md used and they are kept for comparability, but the confound is real and
    # belongs in the write-up beside any A-vs-B contrast.
    assert span_two / span_one == pytest.approx(1.2)


# =================================================================================================
# Round 2: the k/rho sweep and the stationary control
# =================================================================================================
from src_continuous_control.scripts.make_multienv_configs import (      # noqa: E402
    K_RHO_SWEEP,
    SWEEP_ENVIRONMENTS,
    k_sweep_stem_for,
    stationary_stem_for,
)

KSWEEP_CASES = [(e, v) for e in SWEEP_ENVIRONMENTS for v in K_RHO_SWEEP]
KSWEEP_IDS = ["%s-k%d-rho%03d" % (e, v["k"], round(v["rho"] * 100)) for e, v in KSWEEP_CASES]
STATIONARY_CASES = [(e, a) for e in SWEEP_ENVIRONMENTS for a in ("vanilla", "ewc", "pt")]
STATIONARY_IDS = ["%s-%s" % (e, a) for e, a in STATIONARY_CASES]


def merged_ksweep(env, variant):
    return build_config(argparse.Namespace(agent="pt", config=k_sweep_stem_for(env, variant)))


def merged_stationary(env, arm):
    return build_config(argparse.Namespace(agent=ARM_AGENT[arm],
                                           config=stationary_stem_for(env, arm)))


@pytest.mark.parametrize("env,variant", KSWEEP_CASES, ids=KSWEEP_IDS)
def test_k_sweep_realises_its_two_knobs(env, variant):
    """The filename is a label; this asserts what a run would ACTUALLY get.

    Failure mode #3 in CLAUDE.md is a config key one arm reads and another ignores. This sweep's
    entire content is two knobs, so they are read back off the merged config rather than trusted.
    """
    cfg = merged_ksweep(env, variant)
    assert int(cfg["k"]) == variant["k"]
    assert float(cfg["rho"]) == pytest.approx(variant["rho"])


@pytest.mark.parametrize("env,variant", KSWEEP_CASES, ids=KSWEEP_IDS)
def test_k_sweep_keeps_the_transfer_composition_preserving(env, variant):
    """`decay_rho` must track `rho`.

    The permanent absorbs rho and the transient retains (1-rho): one transfer, so the composed
    policy does not jump at a consolidation. Setting them apart makes the value jump by
    (rho - decay_rho)*V_T every time, which compounds -- see the warning in ppo_pt.py.
    """
    cfg = merged_ksweep(env, variant)
    assert float(cfg.get("decay_rho", cfg["rho"])) == pytest.approx(float(cfg["rho"]))


@pytest.mark.parametrize("env,variant", KSWEEP_CASES, ids=KSWEEP_IDS)
def test_k_sweep_differs_from_its_lipschitz2_baseline_in_k_and_rho_ONLY(env, variant):
    """The sweep is only interpretable if nothing else moved.

    Its baseline is the shipped multienv_lipschitz2_<env>_pt at k=10, rho=0.5, which is NOT re-run.
    So every other key -- widths, drift amplitudes and periods, exploration, episode length -- has
    to be identical, or the comparison measures two things at once.
    """
    base = merged_drift("lipschitz2", env, "pt")
    got = merged_ksweep(env, variant)
    assert set(base) == set(got), "the sweep config added or dropped a key"
    differing = {key for key in base if base[key] != got[key]}
    assert differing <= {"k", "rho", "config"}, "changed more than the two knobs: %s" % differing
    assert "k" in differing or variant["k"] == int(base["k"])


@pytest.mark.parametrize("env,variant", KSWEEP_CASES, ids=KSWEEP_IDS)
def test_k_sweep_rho_stays_inside_the_agent_s_bounds(env, variant):
    """rho is a FRACTION of the transient and ppo_pt.py raises outside [0, 1].

    This is why exact compensation exists at only one point: holding the per-update transfer rate
    rho/k constant against the k=10, rho=0.5 baseline needs rho = 0.5*(k/10), which is 1.0 at
    k = 20 and impossible above it. The sweep is designed around that ceiling, so the ceiling is
    pinned here.
    """
    rho = float(merged_ksweep(env, variant)["rho"])
    assert 0.0 <= rho <= 1.0
    exact = 0.5 * (variant["k"] / 10.0)
    if exact <= 1.0:
        assert variant["rho"] in (0.5, exact), "at k<=20 the sweep should carry both rho points"


def test_k_sweep_contains_the_compensated_pair():
    """k = 20 at two rho values is the control that isolates rho from k.

    Without it, every arm changes the consolidation interval AND the total transfer at once, and a
    difference cannot be attributed. Losing this pair silently would gut the design.
    """
    at_20 = sorted(v["rho"] for v in K_RHO_SWEEP if v["k"] == 20)
    assert at_20 == [0.5, 1.0]
    assert sorted({v["k"] for v in K_RHO_SWEEP}) == [20, 30, 60]


@pytest.mark.parametrize("env,arm", STATIONARY_CASES, ids=STATIONARY_IDS)
def test_stationary_config_disables_boundaries(env, arm):
    cfg = merged_stationary(env, arm)
    assert cfg.get("disable_task_switch") is True
    assert cfg["drift_schedule"] == "step"
    # No boundaries means no task index, so the transfer matrix is undefined rather than empty.
    assert int(cfg["transfer_eval_episodes"]) == 0


@pytest.mark.parametrize("env,arm", STATIONARY_CASES, ids=STATIONARY_IDS)
def test_stationary_config_enables_the_ewc_timer(env, arm):
    """Online EWC accumulates its Fisher at a boundary; there are none here.

    Without the timer the penalty is identically zero and the EWC arm is vanilla PPO under another
    name -- which would make the control's own baseline fake.
    """
    assert int(merged_stationary(env, arm)["ewc_consolidate_every"]) > 0


@pytest.mark.parametrize("env", list(SWEEP_ENVIRONMENTS))
def test_stationary_arms_agree_on_every_environment_key(env):
    """The three arms must differ only in the agent, exactly as in the rest of the family."""
    keys = ("dmc_env", "drift_targets", "task_multipliers", "drift_schedule",
            "disable_task_switch", "max_episode_steps", "total_steps", "transfer_eval_episodes")
    seen = {}
    for arm in ("vanilla", "ewc", "pt"):
        cfg = merged_stationary(env, arm)
        for key in keys:
            seen.setdefault(key, set()).add(repr(cfg.get(key)))
    differing = {k for k, v in seen.items() if len(v) > 1}
    assert not differing, "%s: arms disagree on %s" % (env, differing)


@pytest.mark.parametrize("env", list(SWEEP_ENVIRONMENTS))
def test_stationary_physics_really_do_not_move(env):
    """REALISED behaviour, not the config key.

    Failure mode #1 in CLAUDE.md is a control that was not actually off: `lr_perm = 0` disables the
    permanent's learning but not its decay, and nobody noticed for a week. Here the claim is that
    the physics never change, so this drives the env the way a stationary run does -- construct it,
    never call set_task, step it -- and asserts the scaled parameter is BIT-IDENTICAL throughout.
    """
    from src_continuous_control.envs.dm_control_drift import DmControlDrift

    cfg = merged_stationary(env, "pt")
    kw = dict(env_name=env, drift_targets=list(cfg["drift_targets"]),
              task_multipliers=list(cfg["task_multipliers"]),
              schedule=cfg["drift_schedule"], max_episode_steps=200)
    drift_env = DmControlDrift(**kw)
    try:
        drift_env.reset(seed=3)
        start = drift_env.multiplier()
        # The first multiplier IS the unperturbed one; a stationary run must sit at x1.0.
        assert start == pytest.approx(1.0)
        rng = np.random.RandomState(5)
        low, high = drift_env.action_space.low, drift_env.action_space.high
        for _ in range(400):
            obs, rew, term, trunc, info = drift_env.step(rng.uniform(low, high))
            if term or trunc:
                drift_env.reset(seed=3)
            # Bit-identical, not approximately: nothing should be touching this at all.
            assert drift_env.multiplier() == start
    finally:
        drift_env.close()
