"""Generate the config overlays for the dm_control family.

    python -m src_continuous_control.scripts.make_multienv_configs           # write them
    python -m src_continuous_control.scripts.make_multienv_configs --check   # verify, write nothing

Run from the parent directory of the package. The overlays must agree in every key except the one
under test, so they are emitted from one table instead of written by hand. Files are written as
ASCII, so a stray non-ASCII character fails here rather than when another platform reads them.
"""
import argparse
import io
import os
import textwrap

# Per environment: observation and action sizes, `why` (the reason for the scaled physics
# parameter) and `note` (a paragraph for the file header). Both strings go into the generated files.
ENVIRONMENTS = {
    "cartpole-swingup": dict(
        obs=5, act=1,
        why="the pole is the body, so its length and mass set the control problem",
        note="The pt widths are the same as in configs/cartpole_pt.yaml, so these runs are "
             "comparable with the standalone cartpole-swingup runs.",
    ),
    "cartpole-swingup_sparse": dict(
        obs=5, act=1,
        why="identical to cartpole-swingup except for dm_control's `sparse` reward flag",
        note="Not part of the six-environment family. A controlled test of reward density: the "
             "sparse reward is 1 only when the cart is within +-0.25 of centre and the pole is "
             "within 5.7 degrees of vertical, else 0 (a random policy scores 27.6 on dense and "
             "0.0 on sparse). Everything else matches cartpole-swingup.",
    ),
    "reacher-easy": dict(
        obs=6, act=2,
        why="the arm is the body; the direct analogue of the pole, with no balance and no gait",
        note="reacher stores its goal in the model, so a physics rebuild would move the target "
             "mid-episode. envs/dm_control_drift.py preserves it (see "
             "tests/test_dmc_drift_env.py).",
    ),
    "ball_in_cup-catch": dict(
        obs=8, act=2,
        why="the task is swung momentum, so the ball's mass is the knob that changes the control "
            "problem",
        note="Check the dynamic range: DART reports this task near 800/1000, which may leave "
             "too little room for the arms to separate.",
    ),
    "walker-stand": dict(
        obs=24, act=6,
        why="joint damping is only 0.1, so limb mass is scaled instead (torso unscaled)",
        note="Pairs with walker-walk: the two configs differ only in the task name.",
    ),
    "walker-walk": dict(
        obs=24, act=6,
        why="identical to walker-stand; the pair is only meaningful if nothing else differs",
        note="Same robot, motors, observations, physics change and multipliers as walker-stand; "
             "only the goal differs. cartpole versus HalfCheetah confounds body size with task "
             "type, and this pair does not.",
    ),
    "cheetah-run": dict(
        obs=17, act=6,
        why="the same parameters as the gymnasium HalfCheetah runs",
        note="obs 17 / act 6, the same dimensions as gymnasium HalfCheetah, which links the "
             "family to those results.",
    ),
}

# pt widths per environment (permanent, transient) and the measured parameter totals against
# vanilla's [64,64]. Chosen for parity within 0.5% with the transient about half the permanent's
# width (the paper's PT-0.5x). scripts/preflight.py --multienv asserts them on constructed modules.
WIDTHS = {
    "cartpole-swingup": dict(perm=55, trans=30, pt_total=9215, base=9219, ratio=0.9996),
    # Same obs 5 / act 1 as dense cartpole, so the same widths and the same capacity.
    "cartpole-swingup_sparse": dict(perm=55, trans=30, pt_total=9215, base=9219, ratio=0.9996),
    "reacher-easy": dict(perm=54, trans=31, pt_total=9377, base=9413, ratio=0.9962),
    "ball_in_cup-catch": dict(perm=55, trans=29, pt_total=9672, base=9669, ratio=1.0003),
    "walker-stand": dict(perm=53, trans=28, pt_total=11985, base=11981, ratio=1.0003),
    "walker-walk": dict(perm=53, trans=28, pt_total=11985, base=11981, ratio=1.0003),
    "cheetah-run": dict(perm=54, trans=28, pt_total=11110, base=11085, ratio=1.0023),
}

# One multiplier sequence for every environment. The spread is not tuned per environment to
# equalise the boundary drop: that disruption is what carry-over varies with, so equalising it
# would remove the variation the cross-environment comparison needs. The realised disruption is
# reported per environment instead.
# The sequence revisits (tasks 1/3 and 2/4 share physics), so backward transfer is defined.
STANDARD_SPREAD = [1.0, 1.6, 0.6, 1.6, 0.6]
MULTIPLIERS = {name: list(STANDARD_SPREAD) for name in ENVIRONMENTS}

ARMS = ("vanilla", "ewc", "pt", "pt_frozen")

# The two boundary-free drift settings. Lipschitz1 is one slow sine; Lipschitz2 adds a fast
# component, which gives pt's transient something to track. The values match the cartpole drift
# study, so the cartpole cell here replicates it. Lipschitz2's slow amplitude is smaller (0.4 vs
# 0.5) so the two settings cover a comparable range.
DRIFT_KEYS_COMMON = """
# No task boundaries, so no transfer matrix: FWT and BWT are indexed by task number.
transfer_eval_episodes: 0

# Online EWC consolidates at task boundaries. Without this timer it never accumulates a Fisher and
# its penalty stays zero.
ewc_consolidate_every: 10

# anneal_lr would decay every learning rate toward zero and stop the agent adapting to the drift.
anneal_lr: false

# How far the multiplier must move before the MJCF is recompiled (0.4% of the drift range), so the
# drift is a fine staircase. Every arm sees the same environment.
dmc_reload_tol: 0.005
"""

SETTINGS = {
    "lipschitz1": dict(
        stem="multienv_lipschitz1",
        title="Lipschitz1 (one slow sine)",
        blurb="One sine, 2.5 cycles across the run, no fast component.",
        keys=dict(drift_schedule="sin", drift_amplitude=0.5, drift_period=1228800,
                  drift_amplitude2=0.0, drift_period2=30720, drift_phase=0.0),
    ),
    "lipschitz2": dict(
        stem="multienv_lipschitz2",
        title="Lipschitz2 (slow sine plus fast ripple)",
        blurb="The same slow sine at a smaller amplitude, plus a fast ripple of about 100 cycles "
              "across the run. The fast component is what the transient is meant to track.",
        keys=dict(drift_schedule="sin", drift_amplitude=0.4, drift_period=1228800,
                  drift_amplitude2=0.2, drift_period2=30720, drift_phase=0.0),
    ),
}

# ball_in_cup-catch is left out of the drift settings: in the boundary study all three arms finished
# at 839-852 of 1000, so it does not separate them, and its physics rebuild is expensive.
DRIFT_ENVIRONMENTS = ("cartpole-swingup", "reacher-easy", "walker-stand", "walker-walk",
                      "cheetah-run")

HEADER = """\
# {arm_title} on {env}, with task boundaries.
#
# Generated by scripts/make_multienv_configs.py, do not edit by hand. Edit the script and
# regenerate.
#
#   python -m src_continuous_control.train --agent {agent} --config {stem} --seed 0
#
# Environment: {env}, {act} motor(s), {obs} observations.
# Scaled at each boundary: {targets}
#   ({why})
#
# The return ceiling is 1000: reward is in [0,1] per step over exactly 1000 steps with no early
# termination, so a return reads as a percentage of optimal.
#
# {note}
"""

SIGMA_BLOCK = """
# Exploration: learned log_std starting at sigma = 1.0 (CleanRL's ppo_continuous_action, as in
# DART). Set in every arm's overlay so all arms share it; preflight checks the realised value.
freeze_log_std: false
log_std_init: 0.0
"""


def _env_block(env, cfg):
    return (
        '\nenv_mode: "dmc"\n'
        'dmc_env: "%s"\n'
        '\n'
        '# Same schedule and multipliers across the family; environments differ only in which\n'
        '# physics parameters are scaled.\n'
        'drift_schedule: "step"\n'
        'drift_targets: [%s]\n'
        'task_multipliers: %s\n'
        '\n'
        '# Exactly 1000 steps with no early termination; the 1000 ceiling depends on it.\n'
        'max_episode_steps: 1000\n'
        % (env,
           ", ".join('"%s"' % t for t in cfg["targets"]),
           # repr(float) keeps the decimal point so YAML parses every entry as a float
           # ("%g" renders 1.0 as 1, an int).
           "[" + ", ".join(repr(float(m)) for m in cfg["multipliers"]) + "]")
    )


def _widths_block(env, arm):
    w = WIDTHS[env]
    if arm in ("vanilla", "ewc"):
        same = ("Same width as vanilla: EWC adds a penalty, not a different architecture."
                if arm == "ewc" else
                "The baseline; pt's widths are matched against this total.")
        return ("\n# Widths\n# %s\n# Measured at obs %d / act %d: %s parameters in total.\n"
                "hidden_sizes: [64, 64]\n"
                % (same, ENVIRONMENTS[env]["obs"], ENVIRONMENTS[env]["act"],
                   format(w["base"], ",")))
    return (
        "\n# Widths, derived for obs %d / act %d.\n"
        "#\n"
        "# pt has four networks to the baseline's two, so parity depends on the input and output\n"
        "# dimensions. Parameter totals, counted on constructed modules:\n"
        "#\n"
        "#   vanilla / ewc  [64,64]                   total %9s   1.0000x\n"
        "#   pt  perm [%d,%d] + trans [%d,%d]  total %9s   %.4fx\n"
        "#   pt  perm [51,51] + trans [32,32]   (the HalfCheetah widths)  %s\n"
        "#\n"
        "# The permanent is near baseline width and the transient about half (the paper's\n"
        "# PT-0.5x), within 0.5%% of the baseline's parameter count. PPO's gradient reaches only\n"
        "# the transient (mu_P is detached), so the trainable count is below the total; preflight\n"
        "# prints both.\n"
        "hidden_sizes: [%d, %d]                  # permanent actor\n"
        "actor_trans_hidden_sizes: [%d, %d]      # transient actor, about half width\n"
        "critic_hidden_sizes: [%d, %d]           # permanent critic\n"
        "critic_trans_hidden_sizes: [%d, %d]     # transient critic, about half width\n"
        % (ENVIRONMENTS[env]["obs"], ENVIRONMENTS[env]["act"],
           format(WIDTHS[env]["base"], ","),
           WIDTHS[env]["perm"], WIDTHS[env]["perm"], WIDTHS[env]["trans"], WIDTHS[env]["trans"],
           format(WIDTHS[env]["pt_total"], ","), WIDTHS[env]["ratio"],
           _shipped_note(env),
           WIDTHS[env]["perm"], WIDTHS[env]["perm"],
           WIDTHS[env]["trans"], WIDTHS[env]["trans"],
           WIDTHS[env]["perm"], WIDTHS[env]["perm"],
           WIDTHS[env]["trans"], WIDTHS[env]["trans"])
    )


# The parity ratio pt would get with the HalfCheetah widths, shown in each file for reference.
_SHIPPED = {
    "cartpole-swingup": "0.931x  <- a 7% handicap",
    "cartpole-swingup_sparse": "0.931x  <- a 7% handicap",
    "reacher-easy": "0.939x  <- a 6% handicap",
    "ball_in_cup-catch": "0.948x  <- a 5% handicap",
    "walker-stand": "1.0155x <- a 1.5% advantage",
    "walker-walk": "1.0155x <- a 1.5% advantage",
    "cheetah-run": "0.993x",
}


def _shipped_note(env):
    return _SHIPPED[env]


FROZEN_BLOCK = """
# Ablation: the permanent's learning is off.
#   vanilla  -> this arm: what the split itself adds (transient, decay, KL anchor to a fixed
#                         permanent)
#   this arm -> pt:       what the permanent's learning adds
#
# lr_perm = 0 stops only the permanent's learning; the transient decay, the consolidation cadence
# and the KL anchor still run. Check that the realised `actor_absorbed_frac` is about 0 here and
# not in pt. Report this arm's return beside any retention metric, since retention improves when
# an agent learns less.
lr_perm: 0.0
lr_perm_actor: 0.0

# Robbins-Monro annealing of a zero learning rate is a no-op, so it is turned off.
rm_power: 0.0
rm_power_actor: 0.0
"""

ARM_TITLE = {"vanilla": "vanilla PPO", "ewc": "Online EWC",
             "pt": "`pt` (split actor + split critic)",
             "pt_frozen": "`pt` ablation, permanent frozen"}
ARM_AGENT = {"vanilla": "vanilla", "ewc": "ewc", "pt": "pt", "pt_frozen": "pt"}


def stem_for(env, arm):
    """multienv_<environment>_<arm>, with '-' in the dm_control name replaced by '_'."""
    return "multienv_%s_%s" % (env.replace("-", "_"), arm)


def _wrap(text, width=96):
    """Wrap a note to the comment width of the generated file."""
    return "\n# ".join(textwrap.wrap(" ".join(text.split()), width=width))


def render(env, arm):
    from ..envs.dm_control_drift import SPECS
    spec = SPECS[env]
    meta = ENVIRONMENTS[env]
    cfg = dict(targets=spec.default_targets, multipliers=MULTIPLIERS[env])
    text = HEADER.format(
        arm_title=ARM_TITLE[arm], env=env, agent=ARM_AGENT[arm], stem=stem_for(env, arm),
        obs=meta["obs"], act=meta["act"], targets=", ".join(spec.default_targets),
        why=meta["why"], note=_wrap(meta["note"]))
    text += _env_block(env, cfg)
    text += SIGMA_BLOCK
    text += _widths_block(env, arm)
    if arm == "pt_frozen":
        text += FROZEN_BLOCK
    return text


DRIFT_HEADER = """\
# {arm_title} on {env}: {title}.
#
# Generated by scripts/make_multienv_configs.py, do not edit by hand.
#
#   python -m src_continuous_control.train --agent {agent} --config {stem} --seed 0
#
# Environment: {env}, {act} motor(s), {obs} observations.
# Drifting: {targets}
#   ({why})
#
# {blurb}
#
# No task boundaries: the physics change every step on a sine schedule, with no task index and no
# switch signal. The return ceiling is 1000 (reward in [0,1] per step, 1000 steps, no early
# termination).
"""


def drift_stem_for(env, arm, setting):
    """multienv_<setting>_<environment>_<arm>, with the setting name written in full."""
    return "%s_%s_%s" % (SETTINGS[setting]["stem"], env.replace("-", "_"), arm)


def _drift_env_block(env, spec, setting):
    k = SETTINGS[setting]["keys"]
    return (
        '\nenv_mode: "dmc"\n'
        'dmc_env: "%s"\n'
        '\n'
        '# The same physics parameters as the boundary configs, so the two settings compare.\n'
        'drift_targets: [%s]\n'
        '\n'
        '# Drift schedule. The values match the cartpole drift study, so the cartpole cell\n'
        '# replicates it.\n'
        'drift_schedule: "%s"\n'
        'drift_amplitude: %g          # slow component: the permanent\'s job\n'
        'drift_period: %d       # one slow cycle in global env steps (about 2.5 per run)\n'
        'drift_amplitude2: %g         # fast component: the transient\'s job\n'
        'drift_period2: %d          # about 15 PPO updates per fast cycle\n'
        'drift_phase: %g\n'
        % (env, ", ".join('"%s"' % t for t in spec.default_targets),
           k["drift_schedule"], k["drift_amplitude"], k["drift_period"],
           k["drift_amplitude2"], k["drift_period2"], k["drift_phase"])
        + DRIFT_KEYS_COMMON +
        '\nmax_episode_steps: 1000\n'
    )


def render_drift(env, arm, setting):
    from ..envs.dm_control_drift import SPECS
    spec = SPECS[env]
    meta = ENVIRONMENTS[env]
    s = SETTINGS[setting]
    text = DRIFT_HEADER.format(
        arm_title=ARM_TITLE[arm], env=env, title=s["title"], agent=ARM_AGENT[arm],
        stem=drift_stem_for(env, arm, setting), obs=meta["obs"], act=meta["act"],
        targets=", ".join(spec.default_targets), why=meta["why"], blurb=_wrap(s["blurb"]))
    text += _drift_env_block(env, spec, setting)
    text += SIGMA_BLOCK
    text += _widths_block(env, arm)
    if arm == "pt_frozen":
        text += FROZEN_BLOCK
    return text


# Follow-up experiments on cartpole-swingup and cheetah-run only: a k/rho sweep and a stationary
# control. reacher is excluded because whether its arm can reach the target, and so its ceiling,
# changes with the task. These arms are compared against EWC, the stronger baseline.
SWEEP_ENVIRONMENTS = ("cartpole-swingup", "cheetah-run")

# k/rho sweep: is the permanent actually slow? One PPO update is 256 steps x 8 envs = 2048 env
# steps, so Lipschitz2's fast ripple (30,720 steps) lasts 15 updates while pt consolidates every
# k = 10, about 1.5 rewrites of the permanent per fast cycle.
#
# Raising k slows the permanent but also cuts consolidations per run from 150 to 1500/k. Holding
# the transfer rate rho/k fixed needs
#
#       rho_k = 0.5 * (k / 10)
#
# which is 1.0 at k = 20 and impossible above it (ppo_pt.py rejects rho outside [0, 1]). So:
#
#   k = 20, rho = 0.50   and   k = 20, rho = 1.00     same k, two rho: isolates rho
#   k = 30, rho = 0.50   and   k = 60, rho = 0.50     larger k at fixed rho
#
# k = 10, rho = 0.50 is the existing multienv_lipschitz2_<env>_pt. Check `actor_absorbed_frac` on
# every arm, since the transfer shortfall is measurable.
K_RHO_SWEEP = (
    dict(k=20, rho=0.50, why="same k as the rho = 1.0 arm, with the standard rho"),
    dict(k=20, rho=1.00, why="the only exactly compensated point: rho * (1500/k) matches the "
                             "k = 10 baseline's total transfer. rho = 1.0 also zeroes the "
                             "transient at every consolidation"),
    dict(k=30, rho=0.50, why="2 fast ripples per consolidation instead of 0.67"),
    dict(k=60, rho=0.50, why="4 fast ripples per consolidation; the slowest permanent tested"),
)

# Stationary control. pt's transient starts at the zero function (Theorem 1), so early in a run the
# policy is the permanent alone, which only moves at consolidation. Holding the physics at x1.0
# (`disable_task_switch`, so set_task is never called) separates that startup cost from any effect
# of non-stationarity. Every other key matches the boundary configs.
STATIONARY_ARMS = ("vanilla", "ewc", "pt")

STATIONARY_KEYS = """
# The physics never change: set_task() is never called, so the multiplier stays at
# task_multipliers[0] = 1.0 for the whole run. The sequence is kept as in the boundary config.
disable_task_switch: true

# No boundaries, so no transfer matrix (FWT and BWT are indexed by task number).
transfer_eval_episodes: 0

# The decay-gain probe runs at boundaries and would never fire, so it is turned off explicitly.
decay_gain_probe: false

# Online EWC consolidates at boundaries; without this timer its penalty stays zero.
ewc_consolidate_every: 10
"""


def k_sweep_stem_for(env, variant):
    """multienv_lipschitz2_<environment>_pt_k<k>_rho<rho*100>.

    Both k and rho are in the name because the two k = 20 arms differ only in rho.
    """
    return "%s_%s_pt_k%d_rho%03d" % (SETTINGS["lipschitz2"]["stem"], env.replace("-", "_"),
                                     variant["k"], round(variant["rho"] * 100))


def stationary_stem_for(env, arm):
    return "multienv_stationary_%s_%s" % (env.replace("-", "_"), arm)


K_SWEEP_HEADER = """\
# `pt` on {env}: Lipschitz2 with k = {k}, rho = {rho:g}.
#
# Generated by scripts/make_multienv_configs.py, do not edit by hand.
#
#   python -m src_continuous_control.train --agent pt --config {stem} --seed 0
#
# Purpose:
# {why}
#
# One PPO update is 2048 env steps, so a run is 1500 updates and Lipschitz2's fast ripple is 15
# updates long. This arm consolidates every {k} updates: {ripples:.2f} fast ripples per
# consolidation and {consolidations:.0f} consolidations per run (k = 10 gives 0.67 and 150).
#
# Compare against EWC and report `actor_absorbed_frac` beside the return. At rho = {rho:g} the
# per-update transfer rate rho/k is {rate:.4f}, against 0.0500 for k = 10; a lower rate means less
# total transfer into the permanent.
#
# Every key except k and rho matches multienv_lipschitz2_{env_stem}_pt.
"""

STATIONARY_HEADER = """\
# {arm_title} on {env}: stationary control.
#
# Generated by scripts/make_multienv_configs.py, do not edit by hand.
#
#   python -m src_continuous_control.train --agent {agent} --config {stem} --seed 0
#
# Environment: {env}, {act} motor(s), {obs} observations.
#
# In the family study pt trails vanilla in the first fifth of the run in 14 of 16 cells and leads
# in the last fifth in 13. pt's transient starts at the zero function (Theorem 1), so the early
# deficit may come from that initialisation rather than from the changing physics. If the same
# shape appears here, it is a property of how pt starts.
#
# The return ceiling is 1000 (reward in [0,1] per step, 1000 steps, no early termination).
"""


def render_k_sweep(env, variant):
    from ..envs.dm_control_drift import SPECS
    spec = SPECS[env]
    k, rho = variant["k"], variant["rho"]
    text = K_SWEEP_HEADER.format(
        env=env, k=k, rho=rho, stem=k_sweep_stem_for(env, variant),
        why=_wrap(variant["why"]), env_stem=env.replace("-", "_"),
        ripples=k / 15.0, consolidations=1500.0 / k, rate=rho / k)
    text += _drift_env_block(env, spec, "lipschitz2")
    text += SIGMA_BLOCK
    text += _widths_block(env, "pt")
    text += (
        "\n# The two knobs under test.\n"
        "# k: PPO updates between consolidations. rho: the fraction of the transient the\n"
        "# permanent absorbs at each one; the transient's output layer is then scaled by\n"
        "# (1 - rho). They are one transfer (see decay_rho in ppo_pt.py).\n"
        "k: %d\n"
        "rho: %g\n" % (k, rho)
    )
    return text


def render_stationary(env, arm):
    from ..envs.dm_control_drift import SPECS
    spec = SPECS[env]
    meta = ENVIRONMENTS[env]
    cfg = dict(targets=spec.default_targets, multipliers=MULTIPLIERS[env])
    text = STATIONARY_HEADER.format(
        arm_title=ARM_TITLE[arm], env=env, agent=ARM_AGENT[arm],
        stem=stationary_stem_for(env, arm), obs=meta["obs"], act=meta["act"])
    text += _env_block(env, cfg)
    text += STATIONARY_KEYS
    text += SIGMA_BLOCK
    text += _widths_block(env, arm)
    return text


def all_cases():
    """(path stem, rendered text) for every config this generator owns."""
    out = [(stem_for(e, a), render(e, a)) for e in ENVIRONMENTS for a in ARMS]
    out += [(drift_stem_for(e, a, s), render_drift(e, a, s))
            for s in SETTINGS for e in DRIFT_ENVIRONMENTS for a in ARMS]
    out += [(k_sweep_stem_for(e, v), render_k_sweep(e, v))
            for e in SWEEP_ENVIRONMENTS for v in K_RHO_SWEEP]
    out += [(stationary_stem_for(e, a), render_stationary(e, a))
            for e in SWEEP_ENVIRONMENTS for a in STATIONARY_ARMS]
    return out


def main():
    p = argparse.ArgumentParser(description="generate the multienv config overlays")
    p.add_argument("--check", action="store_true",
                   help="verify the files on disk match what would be generated; write nothing")
    p.add_argument("--configs-dir", default=None)
    args = p.parse_args()

    root = args.configs_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    cases = all_cases()
    stale = []
    for stem, text in cases:
        path = os.path.join(root, stem + ".yaml")
        if args.check:
            on_disk = io.open(path, encoding="ascii").read() if os.path.exists(path) else None
            if on_disk != text:
                stale.append(os.path.basename(path))
            continue
        # ASCII, so a non-ASCII character raises at generation time.
        with io.open(path, "w", encoding="ascii", newline="\n") as f:
            f.write(text)
        print("wrote %s" % os.path.basename(path))
    if args.check:
        if stale:
            print("STALE (regenerate): " + ", ".join(stale))
            return 1
        print("all %d config(s) match the generator" % len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
