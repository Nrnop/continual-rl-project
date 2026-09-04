"""Generate the multi-environment family's config overlays — 6 environments x 4 arms.

    cd "e:/update-single task + videos"
    python -m src_continuous_control.scripts.make_multienv_configs           # write them
    python -m src_continuous_control.scripts.make_multienv_configs --check   # verify, write nothing

WHY THESE ARE GENERATED RATHER THAN HAND-WRITTEN. Twenty-four overlays that must agree with one
another in every key except the one under test is exactly the situation that produced this
project's worst config failures: an arm that read a key another ignored (failure mode #3, which
once handed one arm 3x better exploration and a spectacular fake result), and a benchmark whose
arms silently ran different environments. Emitting them from one table makes "identical except for
the agent" a property of the code rather than a claim in a comment.

WIDTHS ARE PER-ENVIRONMENT AND THAT IS NOT OPTIONAL. `pt` carries four networks where the baseline
carries two, so parity depends on the input and output dimensions, which differ in every
environment here. The shipped HalfCheetah widths [51,51]/[32,32] would land at:

    cartpole-swingup   0.931x      reacher-easy       0.939x
    ball_in_cup-catch  0.948x      walker-stand/walk  1.0155x
    cheetah-run        0.993x

so reusing them would hand `pt` a 7% capacity handicap on three environments and a 1.5% ADVANTAGE
on the walker pair — the one comparison in the study that is supposed to be clean. The table below
is derived by measuring constructed modules, and `scripts/preflight.py --multienv` asserts the
realised counts rather than trusting these numbers.

ENCODING IS EXPLICIT AND ASCII-ONLY. Twelve configs once written with the platform default were
cp1252 and unreadable on Linux, which blocked a 160-run job. Writing with encoding="ascii" makes a
smuggled non-ASCII character raise here, at generation time, instead of on the rented box.
"""
import argparse
import io
import os
import textwrap

# --- the environments, and what changes in each -------------------------------------------------
# `why` is the one-line justification for the physics parameter, and it goes into the generated
# file: a reader looking at walker's config should not have to find this script to learn why limb
# mass was chosen over joint damping.
ENVIRONMENTS = {
    "cartpole-swingup": dict(
        obs=5, act=1,
        why="the pole IS the body, and it is the parameter the 95 committed cartpole runs used",
        note="The family's ANCHOR: 95 runs already exist on this environment, and the widths "
             "below are the ones those runs used, so the anchor point is unchanged.",
    ),
    "cartpole-swingup_sparse": dict(
        obs=5, act=1,
        why="IDENTICAL to cartpole-swingup except for dm_control's `sparse` flag; the pole, the "
            "physics change and the multipliers are the same",
        note="NOT PART OF THE SIX-ENVIRONMENT FAMILY. This is the controlled test of a post-hoc "
             "observation from it: across the family, `pt` failed to beat vanilla on exactly the "
             "two environments with BINARY rewards (reacher-easy, ball_in_cup-catch) and won on "
             "three of the four with smooth shaped rewards. Dense scores four smooth factors so "
             "every step pays something; sparse pays 1 only when the cart is within +-0.25 of "
             "centre AND the pole is within 5.7 degrees of vertical, else 0 -- a random policy "
             "scores 27.6 on dense and EXACTLY 0.0 on sparse. One variable moved. If `pt`'s "
             "+10.4% advantage survives here, the reward-density explanation is wrong.",
    ),
    "reacher-easy": dict(
        obs=6, act=2,
        why="the arm IS the body; the direct analogue of the pole, with no balance and no gait",
        note="reacher stores its GOAL in the model rather than in the data, so a physics rebuild "
             "would move the target mid-episode. envs/dm_control_drift.py preserves it; "
             "tests/test_dmc_drift_env.py pins that the guard is load-bearing.",
    ),
    "ball_in_cup-catch": dict(
        obs=8, act=2,
        why="the task is entirely swung momentum, so the ball's mass is the only knob that "
            "changes what the controller must do",
        note="WATCH THE DYNAMIC RANGE HERE. DART reports this task near 800/1000, which may be "
             "too easy for three arms to separate on. It is the environment most likely to fail "
             "gate 2 and be dropped.",
    ),
    "walker-stand": dict(
        obs=24, act=6,
        why="legged locomotion; walker's joint damping is only 0.1 and would be immeasurable at "
            "0.6-1.6x, so limb mass carries the change. The torso is deliberately left alone, so "
            "the MASS DISTRIBUTION changes rather than the whole body scaling self-similarly",
        note="Half of the study's one controlled comparison. Identical to walker-walk in every "
             "key except the task name -- see that file.",
    ),
    "walker-walk": dict(
        obs=24, act=6,
        why="IDENTICAL to walker-stand by design; the pair is only meaningful if nothing else "
            "differs",
        note="THE PAIR IS THE POINT OF THE STUDY. Same robot, same 6 motors, same 24 "
             "observations, same physics change, same multipliers -- only the GOAL differs. "
             "cartpole-vs-HalfCheetah confounds body size with task type; this pair does not. "
             "If `pt` does well on stand and badly on walk the answer is task type; if it does "
             "the same on both the answer is body size.",
    ),
    "cheetah-run": dict(
        obs=17, act=6,
        why="exactly the convention the HalfCheetah studies use, which ties the family back to "
            "the ~157 existing HalfCheetah runs",
        note="obs 17 / act 6 -- the SAME dimensions as the gymnasium HalfCheetah this project "
             "has been studying all along, so this is the family's bridge to those results.",
    ),
}

# --- pt widths, per environment ------------------------------------------------------------------
# perm, trans, and the MEASURED total against vanilla's [64,64]. Derived by constructing the actual
# modules and searching for parity within 0.5% while holding the transient near the paper's
# PT-0.5x structure (transient about half the permanent's width). Re-derive with the search in
# scripts/preflight.py --multienv, which asserts these against constructed modules.
WIDTHS = {
    "cartpole-swingup": dict(perm=55, trans=30, pt_total=9215, base=9219, ratio=0.9996),
    # Same obs 5 / act 1 as dense cartpole, so the same derivation -- and it MUST be the same,
    # or the pair would differ in capacity as well as in reward density.
    "cartpole-swingup_sparse": dict(perm=55, trans=30, pt_total=9215, base=9219, ratio=0.9996),
    "reacher-easy": dict(perm=54, trans=31, pt_total=9377, base=9413, ratio=0.9962),
    "ball_in_cup-catch": dict(perm=55, trans=29, pt_total=9672, base=9669, ratio=1.0003),
    "walker-stand": dict(perm=53, trans=28, pt_total=11985, base=11981, ratio=1.0003),
    "walker-walk": dict(perm=53, trans=28, pt_total=11985, base=11981, ratio=1.0003),
    "cheetah-run": dict(perm=54, trans=28, pt_total=11110, base=11085, ratio=1.0023),
}

# --- the physics multiplier per task, per environment ---------------------------------------------
# ONE SPREAD FOR EVERY ENVIRONMENT, AND THE CALIBRATION IN MULTIENV_TASK.md SECTION 2.3 IS
# DELIBERATELY NOT DONE. Decided 2026-08-25 after the gate runs; the reasoning belongs here because
# the brief asks for the opposite.
#
# Section 2.3 wants the spreads tuned until each environment's boundary drop is of similar size, so
# that a difference between environments is not simply "one got a harder shove". Section 6 wants
# carry-over on the x-axis of the plot that justifies the study. THOSE ARE THE SAME QUANTITY
# MEASURED TWO WAYS -- disruption is what carry-over is low BECAUSE of -- so equalising the first
# collapses the second. Measured on the gate runs, carry-over across the six environments spans
# -0.22 to 1.17; calibrated, every point would sit at one x value and the headline plot would have
# nothing left to show.
#
# The instrument does not survive contact either. `boundary/return_drop` reads an EMA whose time
# constant is ~205k steps -- a whole task -- so it reports 0.5-2.5% everywhere regardless of the
# environment; and measured from raw episode returns the drop swings from -40% to +73% BETWEEN
# BOUNDARIES OF ONE RUN, which is far wider than the 4-21% differences between environments. One
# short run per environment cannot calibrate on that.
#
# So every environment carries the standard spread -- the same one the cartpole and HalfCheetah
# studies used -- and the realised disruption is REPORTED per environment as a measured covariate
# rather than tuned away. MULTIENV_RESULTS.md section 2 records the numbers.
#
# The sequence CYCLES and must revisit: tasks 1/3 and 2/4 are the same physics, which is what makes
# backward transfer well-defined. A sequence that never revisits has no backward transfer to
# measure at all.
STANDARD_SPREAD = [1.0, 1.6, 0.6, 1.6, 0.6]
MULTIPLIERS = {name: list(STANDARD_SPREAD) for name in ENVIRONMENTS}

ARMS = ("vanilla", "ewc", "pt", "pt_frozen")

# --- the three non-stationarity settings, as agreed with the supervisor ---------------------------
# "piecewise" (the boundary benchmark), "Lipschitz1" (smooth drift at a single rate) and
# "Lipschitz2" (a slow trend plus a fast fluctuation). The boundary setting is what the 180-run
# family study ran; these two are the rest of that design.
#
# THE TWO DRIFT WORLDS ARE A TEST OF pt's OWN MECHANISM, not two arbitrary speeds. `pt` carries a
# slow network and a fast one. If the world only drifts slowly the fast network has nothing to do
# and one ordinary network tracks it perfectly well; the fast ripple is what gives the transient an
# actual job. That prediction was written into the drift wrapper's docstring BEFORE the first drift
# runs, and DRIFT_RESULTS.md section 6 tests it.
#
# THE NUMBERS ARE COPIED FROM DRIFT_RESULTS.md AND MUST NOT BE "IMPROVED". cartpole-swingup appears
# in both that study and this family, so identical amplitudes, periods and reload_tol make the
# family's cartpole cell a free replication of an existing result. Changing them would throw that
# away for nothing. Note the slow amplitude is SMALLER in Lipschitz2 (0.4 vs 0.5) so the two worlds
# cover a comparable total range -- otherwise Lipschitz2 would just be "more drift" rather than
# "the same drift plus a fast component", and the contrast would measure the wrong thing.
DRIFT_KEYS_COMMON = """
# --- NO BOUNDARIES, so no transfer matrix ---
# FWT and BWT are indexed by task number, which does not exist here. This setting measures RETURN
# ONLY: the carry-over-vs-advantage plot cannot be reproduced for drift, and neither can forward or
# backward transfer. That is a property of the setting, not an omission.
transfer_eval_episodes: 0

# ONLINE EWC CANNOT RUN HERE WITHOUT THIS. EWC normally strengthens its protection at a task
# boundary; there are none, so without a timer it never accumulates a Fisher, the penalty is
# identically zero, and the EWC arm is vanilla PPO wearing a different name.
ewc_consolidate_every: 10

# Off, and it matters more here than anywhere: anneal_lr decays every learning rate toward zero,
# which is fatal in a drift experiment whose entire point is continuous adaptation. An archived
# HalfCheetah drift study was made unusable by leaving it on.
anneal_lr: false

# How far the multiplier must move before the MJCF is recompiled. 0.005 is what DRIFT_RESULTS.md
# used -- 0.4% of the drift range -- so the drift is a fine staircase rather than a true continuum.
# It is a cost, not a confound: every arm shares the same environment.
dmc_reload_tol: 0.005
"""

SETTINGS = {
    "lipschitz1": dict(
        stem="multienv_lipschitz1",
        title="Lipschitz1 - smooth drift at a SINGLE rate",
        blurb="One sine, 2.5 cycles across the run, no fast component. The slow network's job; the "
              "transient has nothing to track that a single network could not.",
        keys=dict(drift_schedule="sin", drift_amplitude=0.5, drift_period=1228800,
                  drift_amplitude2=0.0, drift_period2=30720, drift_phase=0.0),
    ),
    "lipschitz2": dict(
        stem="multienv_lipschitz2",
        title="Lipschitz2 - a slow trend PLUS a fast fluctuation",
        blurb="The same slow tide at a slightly smaller amplitude, plus a ripple completing ~100 "
              "cycles across the run. This is the world that gives pt's fast network something to "
              "do, and the one where pt was significant on cartpole in DRIFT_RESULTS.md.",
        keys=dict(drift_schedule="sin", drift_amplitude=0.4, drift_period=1228800,
                  drift_amplitude2=0.2, drift_period2=30720, drift_phase=0.0),
    ),
}

# ball_in_cup-catch is DELIBERATELY EXCLUDED from the drift settings. In the boundary study all
# three arms finished at 839-852 of a 1000 ceiling, a span of 13 points, so it separated nothing --
# the saturated-benchmark failure this project has paid for before. It is also a rebuild
# environment, i.e. one of the expensive ones. Decided 2026-08-26; the asymmetry against the
# six-environment boundary family is intentional and is stated in MULTIENV_RESULTS.md.
DRIFT_ENVIRONMENTS = ("cartpole-swingup", "reacher-easy", "walker-stand", "walker-walk",
                      "cheetah-run")

HEADER = """\
# {arm_title} on {env} -- the multi-environment family study.
#
# GENERATED by scripts/make_multienv_configs.py. Edit that script and regenerate; editing this
# file by hand breaks the symmetry the family depends on, because 24 overlays must agree in every
# key except the one under test.
#
#   cd "e:/update-single task + videos"
#   python -m src_continuous_control.train --agent {agent} --config {stem} --seed 0
#
# ENVIRONMENT: {env}, {act} motor(s), {obs} observations.
# WHAT CHANGES AT A BOUNDARY: {targets}
#   -- {why}
#
# CEILING IS 1000 BY CONSTRUCTION. dm_control's reward is in [0,1] per step over exactly 1000 steps
# with no early termination, so every return here reads as a percentage of optimal and returns
# average across the family with no fudge factor. Report the percentage, not just the raw number.
#
# {note}
"""

SIGMA_BLOCK = """
# --- EXPLORATION: STANDARD PPO, LEARNED, AND MATCHED ACROSS ALL THREE ARMS ---
# log_std is a trainable parameter starting at sigma = 1.0, which is CleanRL's ppo_continuous_action
# and what the DART paper uses on these same environments. It is stated explicitly in every arm's
# overlay rather than inherited, so no arm can quietly run a different exploration schedule -- that
# confound once handed one arm 3x the exploration of the others and produced a fake result.
# preflight asserts the REALISED log_std off live tensors, not this key.
freeze_log_std: false
log_std_init: 0.0
"""


def _env_block(env, cfg):
    return (
        '\nenv_mode: "dmc"\n'
        'dmc_env: "%s"\n'
        '\n'
        '# Same schedule and multipliers as the rest of the family, deliberately: environments\n'
        '# should differ in the ENVIRONMENT and, where unavoidable, in which physics parameter is\n'
        '# scaled -- never in the schedule.\n'
        'drift_schedule: "step"\n'
        'drift_targets: [%s]\n'
        'task_multipliers: %s\n'
        '\n'
        '# The episode is exactly 1000 steps and never terminates early. Do not change without\n'
        '# also changing the reported ceiling.\n'
        'max_episode_steps: 1000\n'
        % (env,
           ", ".join('"%s"' % t for t in cfg["targets"]),
           # Rendered with a decimal point so YAML reads every entry as a float; "%g"
           # turns 1.0 into 1, which parses as an int and makes the sequence heterogeneous.
           "[" + ", ".join(repr(float(m)) for m in cfg["multipliers"]) + "]")
    )


def _widths_block(env, arm):
    w = WIDTHS[env]
    if arm in ("vanilla", "ewc"):
        same = ("Same width as vanilla: EWC is a penalty on the baseline network, not a different\n"
                "# architecture." if arm == "ewc" else
                "The baseline. pt's widths below are matched against this total.")
        return ("\n# --- widths ---\n# %s\n# Measured at obs %d / act %d: %s parameters in total.\n"
                "hidden_sizes: [64, 64]\n"
                % (same, ENVIRONMENTS[env]["obs"], ENVIRONMENTS[env]["act"],
                   format(w["base"], ",")))
    return (
        "\n# --- WIDTHS: RE-DERIVED FOR obs %d / act %d. HalfCheetah's DO NOT TRANSFER. ---\n"
        "#\n"
        "# `pt` has four networks where the baseline has two, so parity depends on the input and\n"
        "# output dimensions. Measured on constructed modules, not computed from a formula:\n"
        "#\n"
        "#   vanilla / ewc  [64,64]                   total %9s   1.0000x\n"
        "#   pt  perm [%d,%d] + trans [%d,%d]  total %9s   %.4fx\n"
        "#   pt  perm [51,51] + trans [32,32]   (the shipped HalfCheetah widths)  %s\n"
        "#\n"
        "# This keeps the paper's PT-0.5x construction -- a permanent near baseline width and a\n"
        "# transient at about half -- while landing within 0.5%% of the baseline's parameter count.\n"
        "#\n"
        "# ALSO REPORT THE TRAINABLE SPLIT. PPO's gradient reaches only the transient (`mu_P` is\n"
        "# detached in the training forward); the permanent moves by the consolidation regression\n"
        "# instead. Parity is matched on TOTALS, which count a network PPO never touches, and on\n"
        "# HalfCheetah that asymmetry became a documented finding. preflight prints both.\n"
        "hidden_sizes: [%d, %d]                  # permanent actor\n"
        "actor_trans_hidden_sizes: [%d, %d]      # transient actor  -- about half width\n"
        "critic_hidden_sizes: [%d, %d]           # permanent critic\n"
        "critic_trans_hidden_sizes: [%d, %d]     # transient critic -- about half width\n"
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


# What reusing HalfCheetah's widths would have cost on each environment. Measured, not asserted in
# prose: the walker entry is an ADVANTAGE rather than a handicap, which is worse, because the
# walker pair is the one comparison the study relies on being clean.
_SHIPPED = {
    "cartpole-swingup": "0.931x  <- a 7% handicap",
    "cartpole-swingup_sparse": "0.931x  <- a 7% handicap",
    "reacher-easy": "0.939x  <- a 6% handicap",
    "ball_in_cup-catch": "0.948x  <- a 5% handicap",
    "walker-stand": "1.0155x <- an ADVANTAGE, on the study's cleanest comparison",
    "walker-walk": "1.0155x <- an ADVANTAGE, on the study's cleanest comparison",
    "cheetah-run": "0.993x",
}


def _shipped_note(env):
    return _SHIPPED[env]


FROZEN_BLOCK = """
# --- THE ABLATION ITSELF: the permanent's LEARNING is off ---
#
# The figure decomposes the method into its two halves:
#   vanilla   -> this arm : what having a SPLIT AT ALL buys (transient, decay, KL anchor to a
#                           permanent that never moves)
#   this arm  -> pt       : what the permanent ACTUALLY LEARNING buys
#
# WHAT THIS DOES AND DOES NOT TURN OFF. `lr_perm = 0` freezes the permanent's LEARNING. It does
# NOT turn off the transient decay, the consolidation cadence, or the KL anchor. A control that was
# not actually off is failure mode #1 in CLAUDE.md, so do not read this key -- assert that this
# arm's realised `actor_absorbed_frac` is ~0 while the live arm's is not.
#
# AND REPORT HOW MUCH THIS ARM LEARNED, ALWAYS. On HalfCheetah the frozen arm scored BWT ~ 0 with a
# peak return of exactly 0.0: every retention metric improves when an agent simply does not learn.
# Here that is easy to state honestly -- report its return as a percentage of 1000.
lr_perm: 0.0
lr_perm_actor: 0.0

# Robbins-Monro annealing of a zero learning rate is a no-op; off, so the log does not advertise a
# schedule that is not running.
rm_power: 0.0
rm_power_actor: 0.0
"""

ARM_TITLE = {"vanilla": "vanilla PPO", "ewc": "Online EWC",
             "pt": "`pt` (split actor + split critic)",
             "pt_frozen": "ABLATION ARM: `pt` with the PERMANENT FROZEN"}
ARM_AGENT = {"vanilla": "vanilla", "ewc": "ewc", "pt": "pt", "pt_frozen": "pt"}


def stem_for(env, arm):
    """multienv_<environment>_<arm>. The environment segment is the dm_control name with '-'->'_'.

    No abbreviations and no invented short names: results/ already has sixty-odd directories called
    things like `s14reset` and `sup`, and MANIFEST.md exists only because nobody could tell them
    apart.
    """
    return "multienv_%s_%s" % (env.replace("-", "_"), arm)


def _wrap(text, width=96):
    """Wrap a note to the file's comment width; an unwrapped 400-column line is unreadable."""
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
# {arm_title} on {env} -- {title}.
#
# GENERATED by scripts/make_multienv_configs.py. Edit that script and regenerate.
#
#   cd "e:/update-single task + videos"
#   python -m src_continuous_control.train --agent {agent} --config {stem} --seed 0
#
# ENVIRONMENT: {env}, {act} motor(s), {obs} observations.
# WHAT DRIFTS: {targets}
#   -- {why}
#
# {blurb}
#
# THERE ARE NO TASK BOUNDARIES HERE. The physics move every step on a sine schedule; there is no
# task index, no switch and no reset signal. This is the setting the thesis proposal specifies.
#
# CEILING IS 1000 BY CONSTRUCTION -- reward in [0,1] over exactly 1000 steps, no early termination.
"""


def drift_stem_for(env, arm, setting):
    """multienv_<setting>_<environment>_<arm>, spelled out.

    "lipschitz1"/"lipschitz2" are the supervisor's own terms for the two drift settings, so they are
    written in full rather than shortened to l1/l2 -- results/ already holds sixty-odd directories
    named things like `s14reset` and `sup`, and MANIFEST.md exists only because nobody could tell
    them apart afterwards.
    """
    return "%s_%s_%s" % (SETTINGS[setting]["stem"], env.replace("-", "_"), arm)


def _drift_env_block(env, spec, setting):
    k = SETTINGS[setting]["keys"]
    return (
        '\nenv_mode: "dmc"\n'
        'dmc_env: "%s"\n'
        '\n'
        '# The same physics parameters this environment uses in the boundary study, so the two\n'
        '# settings can be read against each other rather than being two different experiments.\n'
        'drift_targets: [%s]\n'
        '\n'
        '# --- the drift schedule. These numbers are COPIED FROM DRIFT_RESULTS.md and must not be\n'
        '# "improved": cartpole-swingup appears in both studies, so identical settings make its\n'
        '# cell here a free replication of an existing result. ---\n'
        'drift_schedule: "%s"\n'
        'drift_amplitude: %g          # slow structural component: the permanent\'s job\n'
        'drift_period: %d       # one slow cycle in GLOBAL env steps (~2.5 cycles per run)\n'
        'drift_amplitude2: %g         # fast fluctuation: the transient\'s job\n'
        'drift_period2: %d          # ~15 PPO updates per fast cycle\n'
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


def all_cases():
    """(path stem, rendered text) for every config this generator owns."""
    out = [(stem_for(e, a), render(e, a)) for e in ENVIRONMENTS for a in ARMS]
    out += [(drift_stem_for(e, a, s), render_drift(e, a, s))
            for s in SETTINGS for e in DRIFT_ENVIRONMENTS for a in ARMS]
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
        # ASCII, explicitly. A smuggled non-ASCII character raises here rather than on the box.
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
