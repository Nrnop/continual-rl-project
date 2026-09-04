"""DmControlDrift — ONE drift wrapper for the FAMILY of dm_control benchmarks.

WHY THIS FILE EXISTS. Every conclusion in this project rests on TWO environments: `pt` wins on
cartpole-swingup and loses on HalfCheetah, reproducibly and on independent hardware. The
explanation on the table — `pt` pays off when consecutive tasks share structure — is supported by a
carry-over measurement of 0.56 against 0.23 (disjoint, p = 1.08e-5). That is a relationship claimed
from TWO POINTS, and the two environments differ in size AND in task type at once, so nothing
measured on them can separate those two readings.

This module turns two points into six. `envs/cartpole_swingup.py` was hard-wired to one domain;
this is the same machinery with the domain, the task and the physics-to-scale supplied per
environment:

    cartpole-swingup     1 motor,   5 obs   pole length + pole mass
    reacher-easy         2 motors,  6 obs   arm segment length + mass
    ball_in_cup-catch    2 motors,  8 obs   ball mass
    walker-stand         6 motors, 24 obs   limb mass + ground friction
    walker-walk          6 motors, 24 obs   IDENTICAL to walker-stand, by design
    cheetah-run          6 motors, 17 obs   joint damping + ground friction

THE WALKER PAIR IS THE POINT. `stand` and `walk` are the same robot — same 6 motors, same 24
observations, same body, same physics change — differing only in the goal. If `pt` does well on
`stand` and badly on `walk` the answer is TASK TYPE; if it performs the same on both the answer is
BODY SIZE. Cartpole-versus-HalfCheetah cannot make that separation at any sample size, because
there size and task type move together.

`cheetah-run` is obs 17 / act 6 — the same dimensions as the gymnasium HalfCheetah this project has
been studying all along, so `pt`'s shipped widths carry over unchanged and the family is tied back
to the existing results.

EVERY TASK'S CEILING IS 1000 BY CONSTRUCTION. dm_control's reward is in [0,1] per step over
exactly 1000 steps with no early termination, so returns average across the family with no fudge
factor and every result reads as a percentage of optimal. Verified for all six: with each domain's
own control timestep (0.01, 0.02 or 0.025 s — they are NOT the same, which is why the cartpole
wrapper's hard-coded "/100" could not be carried over) the time limit lands on exactly 1000 steps.

`envs/cartpole_swingup.py` IS NOT REPLACED. 95 committed runs depend on it and it stays exactly as
it is; this module reproduces its cartpole physics independently, and `tests/test_dmc_drift_env.py`
pins that the two agree task for task. A wrapper that quietly disagreed with the anchor it is being
compared against would invalidate the whole family.

----------------------------------------------------------------------------------------------
TWO MECHANISMS FOR CHANGING PHYSICS, AND WHY BOTH ARE NEEDED

`ArrayScale` writes a field of the compiled model in place. Safe ONLY where nothing is derived
from that field: `dof_damping` and the sliding-friction column are read straight out of the model
every step. This is what `LipschitzDriftHalfCheetah` has always done.

`XmlScale` edits the MJCF and lets MuJoCo's own compiler rebuild mass, inertia and centre of mass.
Required for anything geometric or inertial: changing `geom_size` alone leaves `body_mass`,
`body_inertia` and `body_ipos` describing the OLD body, and the hand-derived capsule inertia needed
to patch that up is exactly the class of silent error this project keeps paying for. Writing an
explicit `mass` onto a geom works the same way — the compiler recomputes the inertia tensor to
match, so mass and inertia can never disagree.

The mechanism is therefore declared per TARGET, not per environment: cheetah needs no rebuild at
all, cartpole/reacher/ball_in_cup are rebuild-only, and walker is both (limb mass by rebuild,
ground friction by array).

A REBUILD CAN DESTROY STATE THAT BELONGS TO THE TASK RATHER THAN TO US. Two separate hazards:

  1. `reload_from_xml_string` allocates fresh `mjData`, which would teleport the body back to its
     initial pose at every boundary — a hidden episode reset that no config key asks for, handed to
     every arm. `qpos`/`qvel` are snapshotted and written back, exactly as the cartpole wrapper
     does.

  2. Some dm_control tasks store part of the TASK in the MODEL rather than in the data.
     **`reacher` is one**: `Reacher.initialize_episode` writes the target's radius into
     `geom_size['target']` and draws a random target position into `geom_pos['target']`. A rebuild
     resets both to the XML defaults, so a boundary would silently teleport the goal to the origin
     and resize it — changing the physics AND the task at the same instant, while looking exactly
     like a working experiment. `spec.preserve` names those fields per environment and they are
     restored alongside qpos/qvel. Checked against the dm_control source for all six domains:
     reacher is the only one that does this; cartpole, ball_in_cup, walker and cheetah touch
     `data` only.

MULTIPLICATIVELY-INERT PARAMETERS, FOUND BY INSPECTING THE LIVE MODELS. `dof_armature` is exactly
0.0 on cartpole, reacher and ball_in_cup, and `dof_damping` is 0.0005 on cartpole and 0.01 on
reacher — scaling any of them by 0.6-1.6 is a no-op or immeasurable. `geom_friction` only matters
where there is contact, and reacher and ball_in_cup never touch the ground. None of those are
offered as targets here. Note also that `cheetah.xml` carries `<compiler settotalmass="14"/>`, so
scaling individual cheetah masses would be silently renormalised back to 14 kg — which is why
cheetah's targets are damping and friction and mass is not offered for that domain at all.

A multiplicatively-inert parameter looks exactly like a working experiment and cost Phase 1 a week.
Being absent from the list above is an argument, not a measurement, so `scripts/preflight.py` still
gates on the realised trajectory actually diverging.

SCHEDULES are identical in form to `LipschitzDriftHalfCheetah` and `DriftCartpoleSwingup`, so all
three benchmarks are driven by the same config keys and the same training-loop code path:

    schedule="step"          the multiplier is a function of the TASK INDEX, held constant within
                             a task and changed only in `set_task(i)` at an observable boundary.
                             `task_multipliers` cycles, so tasks REVISIT physics the agent has
                             already seen — backward transfer is undefined on a sequence that
                             never revisits. This is the setting the family study runs.
    schedule="sin"/"linear"  continuous drift, no boundaries. Rebuild-based targets quantize (see
                             `reload_tol`), because a naive implementation would recompile the
                             model on every env step. Array-based targets do not need to.

SEEDING. shimmy's `reset(seed=...)` seeds its own `np_random` but NOT the dm_control task's
`RandomState`, which is what actually draws the initial pose and, on reacher, the target position.
Left alone, every sub-env of a vectorised run would start from an unseeded stream and the run would
not be reproducible from `--seed`. `reset` below re-seeds the task directly.
"""
import functools

import gymnasium as gym
import numpy as np
from lxml import etree

# Reused verbatim from the reward-flip benchmark: neither wrapper knows anything about HalfCheetah,
# and the eval/probe envs must produce the same observation layout as the training env or the
# policy cannot be run on them at all.
from .directional_half_cheetah import TaskIDObservation, TaskIDObservationSingle


# ==============================================================================================
# The two mechanisms for changing physics
# ==============================================================================================
class ArrayScale:
    """Scale a field of the COMPILED model in place, from its nominal value.

    Always from the stored nominal, never from the current value: scaling an already-scaled array
    compounds, so a schedule that visits 1.6 twice in a row would silently end up at 2.56.

    Use only for fields nothing else is derived from. `dof_damping` and the sliding-friction column
    qualify; mass and geometry emphatically do not (see XmlScale).
    """

    kind = "array"

    def __init__(self, field, column=None):
        self.field = field          # e.g. "dof_damping", "geom_friction"
        self.column = column        # e.g. 0 -> the sliding column of geom_friction only

    def nominal(self, model):
        return np.array(getattr(model, self.field), copy=True)

    def apply(self, model, nominal, mult):
        live = getattr(model, self.field)
        if self.column is None:
            live[:] = nominal * mult
        else:
            # Only the sliding-friction column; the torsional and rolling columns stay nominal,
            # which is what LipschitzDriftHalfCheetah does and keeps the benchmarks comparable.
            live[:, self.column] = nominal[:, self.column] * mult


class XmlScale:
    """Edit the MJCF and let MuJoCo's compiler rebuild mass, inertia and COM consistently.

    `edit(root, mult)` mutates the parsed tree in place. It is handed the multiplier for THIS
    target only, so a config that scales length but not mass changes exactly one of them.
    """

    kind = "xml"

    def __init__(self, edit):
        self.edit = edit

    def apply(self, root, mult):
        self.edit(root, mult)


def _scalar(value):
    """One float out of a named-model lookup, refusing anything that is not a single number.

    dm_control's named indexing returns a 0-d array for some fields and a length-1 array for
    others (`dof_damping['bthigh']`), so a bare float() both warns and would quietly take the
    first element of a longer row if a field were ever misnamed. Assert the size instead.
    """
    arr = np.asarray(value)
    if arr.size != 1:
        raise RuntimeError(
            "expected a single physics value, got shape %s — the probe is reading the wrong "
            "field and its diagnostics would be meaningless" % (arr.shape,))
    return float(arr.ravel()[0])


def _require(root, path, what):
    """Find exactly one element or fail loudly.

    A dm_control release that restructured a model would otherwise make the edit silently do
    nothing, and the environment would run at nominal physics while the log printed a multiplier
    of 1.6 — failure mode #2, a manipulation that quietly does not happen.
    """
    el = root.find(path)
    if el is None:
        raise RuntimeError(
            "could not find %s at %r in dm_control's MJCF; the model changed and this wrapper's "
            "scaling is no longer valid" % (what, path))
    return el


# Masses below are written back with 17 significant figures because most of them are DERIVED by
# the compiler from density and volume rather than written in the XML. An explicit `mass` attribute
# at a multiplier of exactly 1.0 has to reproduce the nominal model bit for bit, or "the same
# physics" would not be the same physics across two configs. `tests/test_dmc_drift_env.py` pins
# that x1.0 is an exact no-op on body_mass AND body_inertia for every environment.
_G = "%.17g"


# ==============================================================================================
# Per-domain MJCF edits
# ==============================================================================================
# --- cartpole ---------------------------------------------------------------------------------
# Both attributes live on the `pole` default class, so one edit covers every pole in the chain.
# Identical to envs/cartpole_swingup.py, which is the anchor these must agree with.
_CARTPOLE_POLE_LENGTH = 1.0     # <geom ... fromto="0 0 0 0 0 1">
_CARTPOLE_POLE_MASS = 0.1       # <geom ... mass=".1">
_CARTPOLE_POLE_GEOM = './default/default[@class="pole"]/geom'


def _edit_cartpole_length(root, mult):
    geom = _require(root, _CARTPOLE_POLE_GEOM, "the cartpole 'pole' default geom")
    geom.set("fromto", "0 0 0 0 0 %.10g" % (_CARTPOLE_POLE_LENGTH * mult))


def _edit_cartpole_mass(root, mult):
    geom = _require(root, _CARTPOLE_POLE_GEOM, "the cartpole 'pole' default geom")
    geom.set("mass", "%.10g" % (_CARTPOLE_POLE_MASS * mult))


# --- reacher ----------------------------------------------------------------------------------
# The arm IS the body here, which is what makes it the direct analogue of the pole. Scaling its
# length means scaling the capsules AND the child body offsets: leave the offsets alone and the
# segments detach from one another, which is a different robot rather than a longer one.
#
# Note the model's own quirk, preserved deliberately: the `finger` body sits .12 from the hand's
# origin while the hand capsule is only .1 long, so the fingertip overhangs the capsule. Both are
# scaled by the same factor, so the shape of the arm is preserved exactly and only its size changes.
_REACHER_ARM_LEN = 0.12         # <geom name="arm" fromto="0 0 0 0.12 0 0">
_REACHER_HAND_LEN = 0.1         # <geom name="hand" fromto="0 0 0 0.1 0 0">
_REACHER_HAND_POS = 0.12        # <body name="hand" pos=".12 0 0">
_REACHER_FINGER_POS = 0.12      # <body name="finger" pos=".12 0 0">
_REACHER_ARM_MASS = 0.04188790204786391     # compiler-derived from density x volume
_REACHER_HAND_MASS = 0.03560471674068432


def _reacher_bodies(root):
    arm = _require(root, './worldbody/body[@name="arm"]', "the reacher 'arm' body")
    hand = _require(arm, './body[@name="hand"]', "the reacher 'hand' body")
    finger = _require(hand, './body[@name="finger"]', "the reacher 'finger' body")
    return arm, hand, finger


def _edit_reacher_length(root, mult):
    arm, hand, finger = _reacher_bodies(root)
    _require(arm, './geom[@name="arm"]', "the reacher 'arm' geom").set(
        "fromto", "0 0 0 %.10g 0 0" % (_REACHER_ARM_LEN * mult))
    hand.set("pos", "%.10g 0 0" % (_REACHER_HAND_POS * mult))
    _require(hand, './geom[@name="hand"]', "the reacher 'hand' geom").set(
        "fromto", "0 0 0 %.10g 0 0" % (_REACHER_HAND_LEN * mult))
    finger.set("pos", "%.10g 0 0" % (_REACHER_FINGER_POS * mult))


def _edit_reacher_mass(root, mult):
    """Write the segment masses explicitly, so length and mass are ORTHOGONAL targets.

    Without this, scaling the length alone would drag the mass along with it (a capsule's
    density-derived mass grows with its length) and a config asking for a length change would
    silently get a mass change too. Cartpole does not have this problem because its pole carries
    an explicit `mass` attribute in dm_control's own XML; reacher's segments do not, so we add one.
    """
    arm, hand, _ = _reacher_bodies(root)
    _require(arm, './geom[@name="arm"]', "the reacher 'arm' geom").set(
        "mass", _G % (_REACHER_ARM_MASS * mult))
    _require(hand, './geom[@name="hand"]', "the reacher 'hand' geom").set(
        "mass", _G % (_REACHER_HAND_MASS * mult))


# --- ball_in_cup ------------------------------------------------------------------------------
# The task is entirely swung momentum — the cup is dragged, the ball is thrown — so the ball's mass
# is the one knob that changes what the controller has to do. Nothing here touches the ground, so
# friction would be inert, and the string is a tendon with a fixed length range that we leave alone.
_BALL_MASS = 0.06544984694978737     # compiler-derived; sphere of radius .025


def _edit_ball_mass(root, mult):
    geom = _require(root, './worldbody/body[@name="ball"]/geom[@name="ball"]',
                    "the ball_in_cup 'ball' geom")
    geom.set("mass", _G % (_BALL_MASS * mult))


# --- walker -----------------------------------------------------------------------------------
# Limb mass, not damping: walker's `dof_damping` is 0.1, small enough that a 0.6-1.6 rescale is
# borderline immeasurable. The TORSO IS DELIBERATELY LEFT ALONE — scaling the limbs against a fixed
# torso changes the mass distribution, which is what a walker's controller actually has to cope
# with; scaling everything uniformly would leave the dynamics nearly self-similar.
_WALKER_LIMB_MASS = (
    ("right_thigh", 4.057890510886818),
    ("right_leg", 2.7813566959781637),
    ("right_foot", 2.094395102393196),
    ("left_thigh", 4.057890510886818),
    ("left_leg", 2.7813566959781637),
    ("left_foot", 2.094395102393196),
)


def _edit_walker_limb_mass(root, mult):
    for name, nominal in _WALKER_LIMB_MASS:
        geom = _require(root, './/geom[@name="%s"]' % name, "the walker %r geom" % name)
        geom.set("mass", _G % (nominal * mult))


# ==============================================================================================
# The per-environment specification
# ==============================================================================================
class DmcSpec:
    """Everything that differs between one dm_control benchmark and another.

    control_timestep is stored and ASSERTED rather than read and trusted: it sets the episode
    length, and the three domains here disagree (0.01 / 0.02 / 0.025 s). A dm_control release that
    changed one would silently change how many steps an episode runs for, and therefore the
    ceiling every return in the family is reported against.

    probes are read BACK OUT OF THE LIVE MODEL for the diagnostics, never recomputed from the
    config, so a manipulation that silently failed shows up in the logs instead of looking like a
    working experiment.

    preserve names model entries that belong to the TASK rather than to the body, and which must
    survive a rebuild — see hazard 2 in the module docstring.
    """

    def __init__(self, domain, task, control_timestep, obs_dim, act_dim,
                 targets, default_targets, probes, nominal, preserve=()):
        self.domain = domain
        self.task = task
        self.control_timestep = float(control_timestep)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.targets = dict(targets)
        self.default_targets = tuple(default_targets)
        self.probes = dict(probes)
        self.nominal = tuple(nominal)
        self.preserve = tuple(preserve)

    @property
    def name(self):
        return "%s-%s" % (self.domain, self.task)

    def model_xml(self):
        """The domain's nominal MJCF, as bytes, ready to be parsed and patched."""
        from dm_control import suite
        module = getattr(suite, self.domain)
        xml, _assets = module.get_model_and_assets()
        return xml


def _walker_spec(task):
    """walker-stand and walker-walk must be IDENTICAL apart from the task name.

    The pair is the study's one controlled comparison — same body, same 6 motors, same 24
    observations, same physics change, differing only in the goal. Building both from this single
    function is what makes "identical" a property of the code rather than a claim in a comment.
    """
    return DmcSpec(
        domain="walker", task=task, control_timestep=0.025, obs_dim=24, act_dim=6,
        targets={"limb_mass": XmlScale(_edit_walker_limb_mass),
                 "ground_friction": ArrayScale("geom_friction", column=0)},
        default_targets=("limb_mass", "ground_friction"),
        probes={"limb_mass": ("drift_limb_mass", lambda n: n.body_mass["right_thigh"]),
                "ground_friction": ("drift_ground_friction", lambda n: n.geom_friction["floor"][0])},
        nominal=(("right_thigh mass", lambda n: n.body_mass["right_thigh"], 4.057890510886818),
                 ("floor friction", lambda n: n.geom_friction["floor"][0], 0.7)),
    )


def _cartpole_spec(task):
    """cartpole-swingup and cartpole-swingup_sparse, from one definition.

    Built by a shared function for exactly the reason `_walker_spec` is: the two differ ONLY in
    dm_control's `sparse` flag, and the comparison between them is worth nothing unless everything
    else is identical. dm_control itself builds them from the same `Physics.from_xml_string` and
    the same `Balance(swing_up=True, ...)` task, so the difference really is one boolean.

    WHAT THE FLAG DOES. Dense scores `upright * small_control * small_velocity * centered` — four
    smooth factors, so every step pays something proportional to how well the pole is doing. Sparse
    scores `cart_in_bounds * angle_in_bounds`, both hard 0/1 tests: 1 only when the cart is within
    +-0.25 of centre AND the pole is within 5.7 degrees of vertical, and 0 otherwise. Measured over
    1000 steps, a random policy scores 27.6 on dense and EXACTLY 0.0 on sparse.

    WHY THE SPARSE VARIANT IS HERE. Across the six-environment family, `pt` failed to beat vanilla
    on exactly the two environments whose rewards are binary (reacher-easy and ball_in_cup-catch)
    and won on three of the four with smooth shaped rewards. That is a POST-HOC observation on six
    points and this pair is the controlled test of it: same body, same physics change, same
    multipliers, same widths, one variable moved. If `pt`'s advantage survives here, the
    reward-density explanation is wrong.

    Both variants are tagged `benchmarking` in dm_control's own suite, so neither is a custom task.
    """
    return DmcSpec(
        domain="cartpole", task=task, control_timestep=0.01, obs_dim=5, act_dim=1,
        targets={"pole_length": XmlScale(_edit_cartpole_length),
                 "pole_mass": XmlScale(_edit_cartpole_mass)},
        default_targets=("pole_length", "pole_mass"),
        probes={"pole_length": ("drift_pole_length", lambda n: n.geom_size["pole_1"][1] * 2.0),
                "pole_mass": ("drift_pole_mass", lambda n: n.body_mass["pole_1"])},
        nominal=(("pole mass", lambda n: n.body_mass["pole_1"], _CARTPOLE_POLE_MASS),
                 ("pole half-length", lambda n: n.geom_size["pole_1"][1],
                  _CARTPOLE_POLE_LENGTH / 2)),
    )


SPECS = {
    "cartpole-swingup": _cartpole_spec("swingup"),
    "cartpole-swingup_sparse": _cartpole_spec("swingup_sparse"),
    "reacher-easy": DmcSpec(
        domain="reacher", task="easy", control_timestep=0.02, obs_dim=6, act_dim=2,
        targets={"arm_length": XmlScale(_edit_reacher_length),
                 "arm_mass": XmlScale(_edit_reacher_mass)},
        default_targets=("arm_length", "arm_mass"),
        probes={"arm_length": ("drift_arm_length", lambda n: n.geom_size["arm"][1] * 2.0),
                "arm_mass": ("drift_arm_mass", lambda n: n.body_mass["arm"])},
        nominal=(("arm mass", lambda n: n.body_mass["arm"], _REACHER_ARM_MASS),
                 ("hand mass", lambda n: n.body_mass["hand"], _REACHER_HAND_MASS),
                 ("arm half-length", lambda n: n.geom_size["arm"][1], _REACHER_ARM_LEN / 2)),
        # THE TARGET LIVES IN THE MODEL, NOT IN THE DATA. Reacher draws a random target position
        # and radius into geom_pos/geom_size at every episode start; a rebuild would reset both to
        # the XML defaults and move the goal mid-episode. See hazard 2 in the module docstring.
        preserve=(("geom_size", "target"), ("geom_pos", "target")),
    ),
    "ball_in_cup-catch": DmcSpec(
        domain="ball_in_cup", task="catch", control_timestep=0.02, obs_dim=8, act_dim=2,
        targets={"ball_mass": XmlScale(_edit_ball_mass)},
        default_targets=("ball_mass",),
        probes={"ball_mass": ("drift_ball_mass", lambda n: n.body_mass["ball"])},
        nominal=(("ball mass", lambda n: n.body_mass["ball"], _BALL_MASS),),
    ),
    "walker-stand": _walker_spec("stand"),
    "walker-walk": _walker_spec("walk"),
    "cheetah-run": DmcSpec(
        domain="cheetah", task="run", control_timestep=0.01, obs_dim=17, act_dim=6,
        # Exactly the convention the HalfCheetah studies use, so the family ties back to them.
        # No rebuild is needed at all here: nothing is derived from damping or friction.
        targets={"joint_damping": ArrayScale("dof_damping"),
                 "ground_friction": ArrayScale("geom_friction", column=0)},
        default_targets=("joint_damping", "ground_friction"),
        probes={"joint_damping": ("drift_joint_damping", lambda n: n.dof_damping["bthigh"]),
                "ground_friction": ("drift_ground_friction",
                                    lambda n: n.geom_friction["ground"][0])},
        nominal=(("bthigh damping", lambda n: n.dof_damping["bthigh"], 6.0),
                 ("bthigh friction", lambda n: n.geom_friction["bthigh"][0], 0.4)),
    ),
}

ENV_NAMES = tuple(SPECS)


# ==============================================================================================
# The wrapper
# ==============================================================================================
class DmControlDrift(gym.Wrapper):
    """A dm_control task whose physics change per task, with a fixed, bounded reward.

    Parameters
    ----------
    env_name : str
        One of `ENV_NAMES`, i.e. dm_control's own "<domain>-<task>".
    drift_targets : sequence[str] or None
        Which physics to scale. None takes the environment's documented defaults; anything not in
        that environment's `targets` is refused rather than ignored.
    task_multipliers : sequence[float]
        schedule="step": the multiplier held during each task. Cycles, so physics revisit.
    amplitude, period, schedule, phase, amplitude2, period2, phase2, clock_scale
        Exactly as in `LipschitzDriftHalfCheetah` and `DriftCartpoleSwingup`.
    reload_tol : float
        Continuous schedules only: rebuild the model when the multiplier has moved by more than
        this. Ignored by schedule="step", which rebuilds on every boundary regardless, and
        irrelevant to environments whose targets are all array-based.
    """

    def __init__(self, env_name="cartpole-swingup", drift_targets=None,
                 amplitude=0.5, period=1228800, schedule="step", phase=0.0,
                 amplitude2=0.0, period2=30720, phase2=0.0,
                 task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6),
                 clock_scale=1, max_episode_steps=1000, render_mode=None,
                 reload_tol=0.005):
        from dm_control import suite
        from shimmy import DmControlCompatibilityV0

        if env_name not in SPECS:
            raise ValueError("unknown env_name %r; valid: %s" % (env_name, (ENV_NAMES,)))
        spec = SPECS[env_name]

        drift_targets = spec.default_targets if drift_targets is None else tuple(drift_targets)
        bad = [t for t in drift_targets if t not in spec.targets]
        if bad:
            raise ValueError("unknown drift target(s) %s for %s; valid: %s"
                             % (bad, env_name, (tuple(spec.targets),)))
        if not drift_targets:
            raise ValueError("drift_targets is empty; the physics would never change")
        if schedule not in ("sin", "linear", "step"):
            raise ValueError("unknown schedule %r; valid: 'sin', 'linear', 'step'" % (schedule,))

        self.task_multipliers = tuple(float(x) for x in task_multipliers)
        if schedule == "step":
            if not self.task_multipliers:
                raise ValueError("schedule='step' needs a non-empty task_multipliers")
            # A single repeated value means the physics never actually change at a boundary. A
            # silently-constant env looks exactly like a working experiment (Phase 1 lost a week to
            # this), so refuse it rather than run it.
            if len(set(self.task_multipliers)) == 1:
                raise ValueError(
                    "schedule='step' with a constant task_multipliers %s would never change the "
                    "physics" % (self.task_multipliers,))

        self.env_name = env_name
        self.spec_ = spec                 # `spec` is taken by gym.Env for its EnvSpec
        self.drift_targets = tuple(drift_targets)
        self.amplitude = float(amplitude)
        self.period = int(period)
        self.schedule = schedule
        self.phase = float(phase)
        self.amplitude2 = float(amplitude2)
        self.period2 = max(int(period2), 1)
        self.phase2 = float(phase2)
        self.clock_scale = int(clock_scale)
        self.reload_tol = float(reload_tol)
        self.task_idx = 0
        self.t = 0                        # virtual clock in GLOBAL env steps; never reset

        # dm_control owns the episode length. The control timestep differs per domain, so the time
        # limit is DERIVED from it rather than hard-coded — at 0.025 s, cartpole's "/100" would
        # have given walker 2500-step episodes and a ceiling of 2500 instead of 1000.
        self._max_episode_steps = int(max_episode_steps)
        dm_env = suite.load(
            spec.domain, spec.task,
            task_kwargs={"random": 0,
                         "time_limit": self._max_episode_steps * spec.control_timestep},
        )
        self._dm_env = dm_env
        self._nominal_check(dm_env)

        # The nominal MJCF, parsed fresh on every rebuild so edits can never accumulate.
        self._base_xml = spec.model_xml()
        self._xml_targets = [t for t in self.drift_targets if spec.targets[t].kind == "xml"]
        self._array_targets = [t for t in self.drift_targets if spec.targets[t].kind == "array"]

        env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
        env = gym.wrappers.FlattenObservation(env)
        super().__init__(env)

        # Captured from the NOMINAL model, before anything is scaled. Every later scaling is
        # relative to these, so re-visiting a multiplier gives the same physics rather than
        # compounding.
        model = dm_env.physics.model
        self._nominal_arrays = {t: spec.targets[t].nominal(model) for t in self._array_targets}

        self._applied_mult = None
        self._apply(self.multiplier())

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
    def _nominal_check(self, dm_env):
        """Fail loudly if dm_control's model is not the one this wrapper scales.

        The multipliers are relative to hard-coded nominal values, so a dm_control release that
        changed a body would silently rescale every task in the sequence at once — a whole
        benchmark quietly redefined, with nothing in the logs to show it.
        """
        spec = self.spec_
        dt = float(dm_env.control_timestep())
        if not np.isclose(dt, spec.control_timestep):
            raise RuntimeError(
                "%s control timestep is %g, not the %g this wrapper derives the episode length "
                "from; episodes would not be %d steps and the ceiling would move"
                % (spec.name, dt, spec.control_timestep, self._max_episode_steps))
        named = dm_env.physics.named.model
        for label, getter, expected in spec.nominal:
            actual = _scalar(getter(named))
            if not np.isclose(actual, expected):
                raise RuntimeError(
                    "%s nominal %s is %r, not the %r this wrapper's multipliers are relative to; "
                    "the benchmark would mean something different"
                    % (spec.name, label, actual, expected))

    # ------------------------------------------------------------------
    # Schedule  (identical in form to LipschitzDriftHalfCheetah)
    # ------------------------------------------------------------------
    def multiplier(self, t=None):
        """Scaling factor applied to the nominal physics at virtual time t.

        For "step" this is a function of the TASK INDEX, not of the clock, so `t` is ignored.
        """
        if self.schedule == "step":
            return self.task_multipliers[self.task_idx % len(self.task_multipliers)]
        t = self.t if t is None else t
        if self.schedule == "sin":
            s = np.sin(2.0 * np.pi * t / self.period + self.phase)
        else:                                    # linear ramp
            s = t / self.period
        m = 1.0 + self.amplitude * s
        if self.amplitude2:
            m += self.amplitude2 * np.sin(2.0 * np.pi * t / self.period2 + self.phase2)
        return float(m)

    def set_task(self, i):
        """Switch to task `i` (semi-continual boundary). Only "step" changes physics here."""
        self.task_idx = int(i)
        if self.schedule == "step":
            self._apply(self.multiplier())
        return self.task_idx

    def lipschitz_constant(self):
        """Max |change in the multiplier| per GLOBAL env step.

        0 for "step": the physics are constant within a task, and the deliberate discontinuity at a
        boundary is not covered by the bound (its size is max |m[i+1] - m[i]|).
        """
        if self.schedule == "step":
            return 0.0
        slow = (abs(self.amplitude) * 2.0 * np.pi / self.period) if self.schedule == "sin" \
            else (abs(self.amplitude) / self.period)
        fast = abs(self.amplitude2) * 2.0 * np.pi / self.period2
        return slow + fast

    # ------------------------------------------------------------------
    # Applying the physics change
    # ------------------------------------------------------------------
    def _apply(self, mult):
        """Move the physics to multiplier `mult`, preserving everything else about the state.

        Re-applying the same multiplier is a no-op. That guard is not an optimisation: on the
        HalfCheetah wrapper, re-applying a mass change every step perturbed the simulation even
        when the value written back was identical, and made a "harder" variant secretly easier.
        """
        tol = 0.0 if self.schedule == "step" else self.reload_tol
        if self._applied_mult is not None and abs(mult - self._applied_mult) <= tol:
            return
        self._applied_mult = mult
        physics = self._dm_env.physics

        if self._xml_targets:
            root = etree.fromstring(self._base_xml)
            for name in self._xml_targets:
                self.spec_.targets[name].apply(root, mult)

            # Snapshot everything a rebuild would otherwise discard, so a boundary changes the
            # physics and NOTHING else: the simulation state, and any model entry the TASK owns
            # (reacher's target position and radius — see hazard 2 in the module docstring).
            qpos = np.array(physics.data.qpos, copy=True)
            qvel = np.array(physics.data.qvel, copy=True)
            preserved = [(field, row, np.array(getattr(physics.named.model, field)[row], copy=True))
                         for field, row in self.spec_.preserve]

            from dm_control.suite import common
            physics.reload_from_xml_string(etree.tostring(root), assets=common.ASSETS)

            for field, row, value in preserved:
                getattr(physics.named.model, field)[row] = value
            physics.data.qpos[:] = qpos
            physics.data.qvel[:] = qvel
            physics.forward()

        # AFTER any rebuild, never before: a rebuild resets the compiled arrays to their XML
        # nominal, so array targets applied first would be silently undone.
        for name in self._array_targets:
            self.spec_.targets[name].apply(
                physics.model, self._nominal_arrays[name], mult)

    def current_params(self):
        """Diagnostics: the multiplier and the REALISED physics read out of the live model.

        Read from the model rather than recomputed from the config, so a manipulation that
        silently failed shows up here instead of looking like a working experiment.
        """
        named = self._dm_env.physics.named.model
        out = {"drift_multiplier": float(self.multiplier())}
        if self.schedule == "step":
            out["drift_task"] = float(self.task_idx)
        for name in self.drift_targets:
            key, getter = self.spec_.probes[name]
            out[key] = _scalar(getter(named))
        return out

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += self.clock_scale
        if self.schedule != "step":          # nothing to re-apply within a task
            self._apply(self.multiplier())
        # Same key the other benchmarks use, so the honest-return logging path is identical.
        info["directional_reward"] = reward
        info.update(self.current_params())
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # The physics clock deliberately survives episode resets — it is a property of the world,
        # not of the episode.
        #
        # Seeding is ours to do: shimmy seeds its own np_random and never touches the dm_control
        # task's RandomState, which is what actually draws the initial pose and, on reacher, the
        # target position.
        seed = kwargs.get("seed")
        if seed is not None:
            self._dm_env.task._random = np.random.RandomState(int(seed))
        return self.env.reset(**kwargs)


# ==============================================================================================
# Factories — wrapper order matches make_drift_env / make_cartpole_env exactly
# ==============================================================================================
def _make_single_dmc(env_name, drift_targets, amplitude, period, schedule, phase,
                     clock_scale, max_episode_steps, render_mode=None,
                     amplitude2=0.0, period2=30720, phase2=0.0,
                     task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6), reload_tol=0.005):
    """Module-level factory so functools.partial stays picklable for AsyncVectorEnv on Windows."""
    return DmControlDrift(
        env_name=env_name, drift_targets=drift_targets, amplitude=amplitude, period=period,
        schedule=schedule, phase=phase, amplitude2=amplitude2, period2=period2, phase2=phase2,
        task_multipliers=task_multipliers, clock_scale=clock_scale,
        max_episode_steps=max_episode_steps, render_mode=render_mode, reload_tol=reload_tol,
    )


def make_dmc_env(env_name="cartpole-swingup", drift_targets=None,
                 amplitude=0.5, period=1228800, schedule="step", phase=0.0,
                 amplitude2=0.0, period2=30720, phase2=0.0,
                 task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6), clock_scale=1,
                 max_episode_steps=1000, render_mode=None, normalize_obs=False,
                 normalize_reward=False, gamma=0.99, clip_obs=10.0, clip_reward=10.0,
                 reload_tol=0.005, task_id_obs=False, n_task_ids=5):
    """Single dm_control env, wrapper order matching make_drift_env / make_cartpole_env."""
    env = _make_single_dmc(env_name, drift_targets, amplitude, period, schedule, phase,
                           clock_scale, max_episode_steps, render_mode,
                           amplitude2, period2, phase2, tuple(task_multipliers), reload_tol)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda o: np.clip(o, -clip_obs, clip_obs), env.observation_space)
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.ClipReward(env, -clip_reward, clip_reward)
    if task_id_obs:                      # outermost, so the label is never normalized
        env = TaskIDObservationSingle(env, n_task_ids)
    return env


def make_dmc_vector_env(env_name="cartpole-swingup", num_envs=1, drift_targets=None,
                        amplitude=0.5, period=1228800, schedule="step", phase=0.0,
                        amplitude2=0.0, period2=30720, phase2=0.0,
                        task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6),
                        max_episode_steps=1000, gamma=0.99, normalize_obs=False,
                        normalize_reward=False, clip_obs=10.0, clip_reward=10.0,
                        asynchronous=True, reload_tol=0.005,
                        task_id_obs=False, n_task_ids=5):
    """Vectorised dm_control env. Wrapper order matches make_cartpole_vector_env exactly.

    clock_scale = num_envs so every sub-env's virtual clock counts GLOBAL env steps and all
    sub-envs stay on an identical schedule.
    """
    fns = [
        functools.partial(_make_single_dmc, env_name, drift_targets, amplitude, period,
                          schedule, phase, num_envs, max_episode_steps, None,
                          amplitude2, period2, phase2, tuple(task_multipliers), reload_tol)
        for _ in range(num_envs)
    ]
    base = gym.vector.AsyncVectorEnv(fns) if (asynchronous and num_envs > 1) \
        else gym.vector.SyncVectorEnv(fns)

    envs = gym.wrappers.vector.RecordEpisodeStatistics(base)
    if normalize_obs:
        envs = gym.wrappers.vector.NormalizeObservation(envs)
        envs = gym.wrappers.vector.TransformObservation(
            envs, functools.partial(np.clip, a_min=-clip_obs, a_max=clip_obs))
    if normalize_reward:
        envs = gym.wrappers.vector.NormalizeReward(envs, gamma=gamma)
        envs = gym.wrappers.vector.ClipReward(envs, -clip_reward, clip_reward)
    if task_id_obs:
        envs = TaskIDObservation(envs, n_task_ids)
    return envs
