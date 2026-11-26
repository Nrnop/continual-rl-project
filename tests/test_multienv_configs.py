"""Tests for the generated dm_control family overlays, checked on the merged config.

The merge is default.yaml <- ppo_<agent>.yaml <- overlay <- CLI, the config training runs on.
These tests check that arms agree on every environment key, that exploration and parameter counts
are matched on constructed modules, and that each control (frozen permanent, drift, stationary,
k/rho sweep) realises what its name says.
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

# Keys that define the environment and the protocol; every arm must agree on all of them.
ENVIRONMENT_KEYS = ("env_mode", "dmc_env", "drift_schedule", "drift_targets", "task_multipliers",
                    "max_episode_steps", "switch", "total_steps", "gamma", "gae_lambda",
                    "n_steps", "num_envs", "epochs", "minibatch_size", "clip_coef",
                    "normalize_obs", "normalize_reward", "normalizer_freeze_after",
                    "anneal_lr", "freeze_log_std", "log_std_init", "transfer_eval_episodes")

CASES = [(env, arm) for env in ENVIRONMENTS for arm in ARMS]
CASE_IDS = ["%s-%s" % (e, a) for e, a in CASES]


def merged(env, arm):
    """The config a run of this arm would get, merged as train.py does."""
    return build_config(argparse.Namespace(agent=ARM_AGENT[arm], config=stem_for(env, arm)))


def build_agent(env, arm, cfg=None):
    cfg = cfg or merged(env, arm)
    spec = SPECS[env]
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):     # the pt constructor prints a summary
        return AGENTS[ARM_AGENT[arm]](spec.obs_dim, spec.act_dim, cfg, torch.device("cpu"))


# The files exist, are ASCII, and match their generator
def test_every_environment_has_every_arm():
    """Every environment has every arm's overlay (a missing one falls back to HalfCheetah keys)."""
    missing = [stem_for(e, a) + ".yaml" for e, a in CASES
               if not os.path.exists(os.path.join(CFG_DIR, stem_for(e, a) + ".yaml"))]
    assert not missing, f"missing config(s): {missing}"
    assert len(CASES) == 28


@pytest.mark.parametrize("env,arm", CASES, ids=CASE_IDS)
def test_config_is_pure_ascii(env, arm):
    """Generated configs are pure ASCII, which no platform default encoding can misread."""
    path = os.path.join(CFG_DIR, stem_for(env, arm) + ".yaml")
    raw = open(path, "rb").read()
    raw.decode("ascii")          # raises if a non-ASCII byte crept in


def test_configs_match_their_generator():
    """The files on disk equal what the generator renders (no hand edits)."""
    from src_continuous_control.scripts.make_multienv_configs import render
    stale = [stem_for(e, a) for e, a in CASES
             if io.open(os.path.join(CFG_DIR, stem_for(e, a) + ".yaml"),
                        encoding="ascii").read() != render(e, a)]
    assert not stale, (f"config(s) no longer match scripts/make_multienv_configs.py: {stale}; "
                       f"edit the generator and regenerate rather than editing the YAML")


# The merged config selects the environment it claims to
@pytest.mark.parametrize("env,arm", CASES, ids=CASE_IDS)
def test_merged_config_selects_the_right_environment(env, arm):
    """The merged config selects the named dm_control env, targets and episode length."""
    cfg = merged(env, arm)
    spec = SPECS[env]
    assert cfg["env_mode"] == "dmc"
    assert cfg["dmc_env"] == env
    assert cfg["drift_schedule"] == "step"
    assert tuple(cfg["drift_targets"]) == spec.default_targets
    assert cfg["max_episode_steps"] == 1000


@pytest.mark.parametrize("env,arm", CASES, ids=CASE_IDS)
def test_task_sequence_cycles_and_revisits(env, arm):
    """The task sequence revisits tasks, so backward transfer is defined."""
    mults = list(merged(env, arm)["task_multipliers"])
    assert len(mults) == 5
    assert mults[1] == mults[3] and mults[2] == mults[4]
    assert len(set(mults)) > 1, "a constant sequence would never change the physics"


# The arms differ only in the agent
@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_arms_agree_on_every_environment_key(env):
    """ewc and pt agree with vanilla on every environment and protocol key."""
    reference = merged(env, "vanilla")
    for arm in ("ewc", "pt"):
        cfg = merged(env, arm)
        for key in ENVIRONMENT_KEYS:
            assert cfg.get(key) == reference.get(key), (
                f"{env}: {arm} and vanilla disagree on {key!r} "
                f"({cfg.get(key)!r} vs {reference.get(key)!r})")


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_sigma_is_matched_across_arms(env):
    """All three arms learn log_std and start at the same sigma, checked on live tensors."""
    sigmas, learned = {}, {}
    for arm in COMPARABLE_ARMS:
        actor = build_agent(env, arm).actor
        sigmas[arm] = float(torch.exp(actor.log_std.detach()).mean())
        learned[arm] = bool(actor.log_std.requires_grad)
    assert len(set(learned.values())) == 1, f"{env}: exploration SCHEDULE differs: {learned}"
    assert all(learned.values()), f"{env}: the family runs standard PPO - log_std must be learned"
    assert np.allclose(list(sigmas.values()), sigmas["vanilla"], atol=1e-6), \
        f"{env}: exploration LEVEL differs at initialisation: {sigmas}"


# Parameter parity, measured on constructed modules
@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_pt_hits_parameter_parity_on_constructed_modules(env):
    """pt's parameter total is within 0.5% of vanilla's, counted on the constructed networks."""
    def total(arm):
        agent = build_agent(env, arm)
        return (sum(p.numel() for p in agent.actor.parameters())
                + sum(p.numel() for p in agent.critic.parameters()))

    base, pt_total = total("vanilla"), total("pt")
    ratio = pt_total / base
    assert abs(ratio - 1.0) <= 0.005, (
        f"{env}: pt is at {ratio:.4f}x the baseline's parameters ({pt_total:,} vs {base:,}); "
        f"re-derive its widths for obs {SPECS[env].obs_dim} / act {SPECS[env].act_dim}")
    # The totals recorded in the generator must match the measured ones.
    assert pt_total == WIDTHS[env]["pt_total"]
    assert base == WIDTHS[env]["base"]


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_ewc_matches_the_baseline_architecture(env):
    """EWC uses the same hidden sizes as vanilla."""
    assert merged(env, "ewc")["hidden_sizes"] == merged(env, "vanilla")["hidden_sizes"]


# The ablation arm
@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_frozen_arm_matches_pt_everywhere_except_the_permanent(env):
    """pt_frozen matches pt on every key except the permanent's learning rates."""
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
    """With lr_perm = 0 the split, the decay and the consolidation cadence are still present."""
    cfg = merged(env, "pt_frozen")
    assert cfg["rho"] > 0.0, "the transient decay must still run"
    assert cfg["k"] > 0, "the consolidation cadence must still run"
    agent = build_agent(env, "pt_frozen")
    assert getattr(agent.actor, "trans_mean", None) is not None, "the split actor must survive"
    assert getattr(agent.critic, "trans", None) is not None, "the split critic must survive"


# The walker pair at the config level
@pytest.mark.parametrize("arm", list(ARMS))
def test_walker_pair_configs_differ_only_in_the_environment(arm):
    """walker-stand and walker-walk configs differ only in dmc_env."""
    stand, walk = merged("walker-stand", arm), merged("walker-walk", arm)
    assert stand["dmc_env"] != walk["dmc_env"]
    for key in ENVIRONMENT_KEYS + ("hidden_sizes",):
        if key == "dmc_env":
            continue
        assert stand.get(key) == walk.get(key), f"walker pair disagrees on {key!r} for arm {arm}"


# The two boundary-free drift settings
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
    """Drift configs use the sine schedule and turn off the transfer matrix."""
    cfg = merged_drift(setting, env, arm)
    assert cfg["drift_schedule"] == "sin"
    # FWT/BWT are indexed by task number, which does not exist here.
    assert cfg["transfer_eval_episodes"] == 0


@pytest.mark.parametrize("setting,env,arm", DRIFT_CASES, ids=DRIFT_IDS)
def test_drift_config_matches_drift_results_settings(setting, env, arm):
    """Drift amplitudes, periods and reload tolerance match the cartpole drift study's values."""
    cfg = merged_drift(setting, env, arm)
    for key, expected in SETTINGS[setting]["keys"].items():
        assert cfg[key] == expected, f"{setting}/{env}/{arm}: {key} is {cfg[key]}, not {expected}"
    assert cfg["dmc_reload_tol"] == 0.005


@pytest.mark.parametrize("setting,env,arm", DRIFT_CASES, ids=DRIFT_IDS)
def test_drift_config_enables_the_ewc_timer(setting, env, arm):
    """Drift configs set the EWC consolidation timer, since there are no boundaries."""
    assert merged_drift(setting, env, arm)["ewc_consolidate_every"] == 10


@pytest.mark.parametrize("setting,env,arm", DRIFT_CASES, ids=DRIFT_IDS)
def test_drift_config_keeps_anneal_lr_off(setting, env, arm):
    """Drift configs keep learning-rate annealing off."""
    assert merged_drift(setting, env, arm)["anneal_lr"] is False


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("env", list(DRIFT_ENVIRONMENTS))
def test_drift_arms_agree_on_every_environment_key(setting, env):
    """ewc and pt agree with vanilla on every drift and environment key."""
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
    """pt is within 0.5% of vanilla's parameter count in the drift configs too."""
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
    """Drift configs scale the same physics parameters as the boundary configs."""
    assert (list(merged_drift(setting, env, "pt")["drift_targets"])
            == list(merged(env, "pt")["drift_targets"]))


def test_ball_in_cup_is_excluded_from_the_drift_settings():
    """ball_in_cup-catch is excluded from the drift settings (it saturated with boundaries)."""
    assert "ball_in_cup-catch" not in DRIFT_ENVIRONMENTS
    assert len(DRIFT_ENVIRONMENTS) == 5


def test_lipschitz2_adds_a_fast_component_and_lipschitz1_does_not():
    """Lipschitz2 adds a fast component and Lipschitz1 has none."""
    one, two = SETTINGS["lipschitz1"]["keys"], SETTINGS["lipschitz2"]["keys"]
    assert one["drift_amplitude2"] == 0.0
    assert two["drift_amplitude2"] > 0.0
    assert one["drift_period"] == two["drift_period"]
    # The slow amplitude is smaller in Lipschitz2 so the total ranges are comparable, not equal.
    assert two["drift_amplitude"] < one["drift_amplitude"]
    span_one = 2 * one["drift_amplitude"]                                    # 0.5 .. 1.5
    span_two = 2 * (two["drift_amplitude"] + two["drift_amplitude2"])        # 0.4 .. 1.6
    assert span_one == pytest.approx(1.0)
    assert span_two == pytest.approx(1.2)
    # Lipschitz2 has 20% more total drift, so a Lipschitz1 vs Lipschitz2 difference mixes more
    # variation with the added fast component.
    assert span_two / span_one == pytest.approx(1.2)


# The k/rho sweep and the stationary control
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
    """The merged config carries the k and rho the sweep variant names."""
    cfg = merged_ksweep(env, variant)
    assert int(cfg["k"]) == variant["k"]
    assert float(cfg["rho"]) == pytest.approx(variant["rho"])


@pytest.mark.parametrize("env,variant", KSWEEP_CASES, ids=KSWEEP_IDS)
def test_k_sweep_keeps_the_transfer_composition_preserving(env, variant):
    """decay_rho equals rho, so consolidation does not change the composed policy.

    Setting them apart makes the value jump by (rho - decay_rho) * V_T at every consolidation.
    """
    cfg = merged_ksweep(env, variant)
    assert float(cfg.get("decay_rho", cfg["rho"])) == pytest.approx(float(cfg["rho"]))


@pytest.mark.parametrize("env,variant", KSWEEP_CASES, ids=KSWEEP_IDS)
def test_k_sweep_differs_from_its_lipschitz2_baseline_in_k_and_rho_ONLY(env, variant):
    """Sweep configs differ from multienv_lipschitz2_<env>_pt (k=10, rho=0.5) only in k, rho."""
    base = merged_drift("lipschitz2", env, "pt")
    got = merged_ksweep(env, variant)
    assert set(base) == set(got), "the sweep config added or dropped a key"
    differing = {key for key in base if base[key] != got[key]}
    assert differing <= {"k", "rho", "config"}, "changed more than the two knobs: %s" % differing
    assert "k" in differing or variant["k"] == int(base["k"])


@pytest.mark.parametrize("env,variant", KSWEEP_CASES, ids=KSWEEP_IDS)
def test_k_sweep_rho_stays_inside_the_agent_s_bounds(env, variant):
    """rho stays in [0, 1]; at k <= 20 the sweep uses rho = 0.5 or the compensated value.

    Holding rho/k at the k=10, rho=0.5 rate needs rho = 0.5*(k/10), which exceeds 1 above k = 20.
    """
    rho = float(merged_ksweep(env, variant)["rho"])
    assert 0.0 <= rho <= 1.0
    exact = 0.5 * (variant["k"] / 10.0)
    if exact <= 1.0:
        assert variant["rho"] in (0.5, exact), "at k<=20 the sweep should carry both rho points"


def test_k_sweep_contains_the_compensated_pair():
    """The sweep has k = 20 at rho 0.5 and 1.0 (isolating rho), plus k = 30 and 60."""
    at_20 = sorted(v["rho"] for v in K_RHO_SWEEP if v["k"] == 20)
    assert at_20 == [0.5, 1.0]
    assert sorted({v["k"] for v in K_RHO_SWEEP}) == [20, 30, 60]


@pytest.mark.parametrize("env,arm", STATIONARY_CASES, ids=STATIONARY_IDS)
def test_stationary_config_disables_boundaries(env, arm):
    """Stationary configs disable task switches and the transfer matrix."""
    cfg = merged_stationary(env, arm)
    assert cfg.get("disable_task_switch") is True
    assert cfg["drift_schedule"] == "step"
    assert int(cfg["transfer_eval_episodes"]) == 0


@pytest.mark.parametrize("env,arm", STATIONARY_CASES, ids=STATIONARY_IDS)
def test_stationary_config_enables_the_ewc_timer(env, arm):
    """Stationary configs set the EWC consolidation timer, since there are no boundaries."""
    assert int(merged_stationary(env, arm)["ewc_consolidate_every"]) > 0


@pytest.mark.parametrize("env", list(SWEEP_ENVIRONMENTS))
def test_stationary_arms_agree_on_every_environment_key(env):
    """The three stationary arms agree on every environment key."""
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
    """Stepping the env without set_task leaves the multiplier at exactly 1.0 throughout."""
    from src_continuous_control.envs.dm_control_drift import DmControlDrift

    cfg = merged_stationary(env, "pt")
    kw = dict(env_name=env, drift_targets=list(cfg["drift_targets"]),
              task_multipliers=list(cfg["task_multipliers"]),
              schedule=cfg["drift_schedule"], max_episode_steps=200)
    drift_env = DmControlDrift(**kw)
    try:
        drift_env.reset(seed=3)
        start = drift_env.multiplier()
        assert start == pytest.approx(1.0)
        rng = np.random.RandomState(5)
        low, high = drift_env.action_space.low, drift_env.action_space.high
        for _ in range(400):
            obs, rew, term, trunc, info = drift_env.step(rng.uniform(low, high))
            if term or trunc:
                drift_env.reset(seed=3)
            # Exact equality: nothing should touch the multiplier.
            assert drift_env.multiplier() == start
    finally:
        drift_env.close()
