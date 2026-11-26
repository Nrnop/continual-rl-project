"""DmControlDrift: one drift wrapper for a family of dm_control tasks.

Supported: cartpole-swingup (and swingup_sparse), reacher-easy, ball_in_cup-catch, walker-stand,
walker-walk and cheetah-run. Each task's reward is in [0, 1] per step over exactly 1000 steps with
no early termination, so the maximum return is 1000. Physics change per task through one of two
mechanisms, declared per target: `XmlScale` patches the MJCF and recompiles (for anything mass- or
geometry-derived) and `ArrayScale` writes the compiled model in place (for damping and friction).
A rebuild preserves qpos/qvel and, for reacher, the target position and radius.
"""
import functools

import gymnasium as gym
import numpy as np
from lxml import etree

# Shared with the HalfCheetah envs so eval/probe envs produce the same observation layout.
from .directional_half_cheetah import TaskIDObservation, TaskIDObservationSingle


# The two mechanisms for changing physics

class ArrayScale:
    """Scale a field of the compiled model in place, from its stored nominal value.

    Scaling from the nominal (not the current value) keeps repeated multipliers from compounding.
    Use only for fields nothing else is derived from, such as `dof_damping` and the sliding
    friction column.
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
            # Only the sliding-friction column, as in LipschitzDriftHalfCheetah.
            live[:, self.column] = nominal[:, self.column] * mult


class XmlScale:
    """Edit the MJCF so MuJoCo's compiler rebuilds mass, inertia and COM consistently.

    `edit(root, mult)` mutates the parsed tree in place, for this target only.
    """

    kind = "xml"

    def __init__(self, edit):
        self.edit = edit

    def apply(self, root, mult):
        self.edit(root, mult)


def _scalar(value):
    """Return one float from a named-model lookup; raise if it is not a single value.

    dm_control's named indexing returns 0-d arrays for some fields and length-1 arrays for others.
    """
    arr = np.asarray(value)
    if arr.size != 1:
        raise RuntimeError(
            "expected a single physics value, got shape %s - the probe is reading the wrong "
            "field and its diagnostics would be meaningless" % (arr.shape,))
    return float(arr.ravel()[0])


def _require(root, path, what):
    """Find exactly one element, or raise so a restructured model cannot make an edit a no-op."""
    el = root.find(path)
    if el is None:
        raise RuntimeError(
            "could not find %s at %r in dm_control's MJCF; the model changed and this wrapper's "
            "scaling is no longer valid" % (what, path))
    return el


# Masses are written with 17 significant figures because most are compiler-derived from density
# and volume; at a multiplier of 1.0 an explicit `mass` must reproduce the nominal model exactly.
_G = "%.17g"


# Per-domain MJCF edits

# cartpole: both attributes live on the `pole` default class, so one edit covers every pole.
# Must agree with envs/cartpole_swingup.py.
_CARTPOLE_POLE_LENGTH = 1.0     # <geom ... fromto="0 0 0 0 0 1">
_CARTPOLE_POLE_MASS = 0.1       # <geom ... mass=".1">
_CARTPOLE_POLE_GEOM = './default/default[@class="pole"]/geom'


def _edit_cartpole_length(root, mult):
    geom = _require(root, _CARTPOLE_POLE_GEOM, "the cartpole 'pole' default geom")
    geom.set("fromto", "0 0 0 0 0 %.10g" % (_CARTPOLE_POLE_LENGTH * mult))


def _edit_cartpole_mass(root, mult):
    geom = _require(root, _CARTPOLE_POLE_GEOM, "the cartpole 'pole' default geom")
    geom.set("mass", "%.10g" % (_CARTPOLE_POLE_MASS * mult))


# reacher: scaling arm length scales the capsules and the child body offsets together, so the
# segments stay attached. The fingertip overhang in the nominal model (finger at .12, hand capsule
# .1 long) is scaled by the same factor, so only the arm's size changes.
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
    """Write the segment masses explicitly, so length and mass are independent targets.

    Reacher's segments have density-derived mass, which would otherwise grow with length.
    """
    arm, hand, _ = _reacher_bodies(root)
    _require(arm, './geom[@name="arm"]', "the reacher 'arm' geom").set(
        "mass", _G % (_REACHER_ARM_MASS * mult))
    _require(hand, './geom[@name="hand"]', "the reacher 'hand' geom").set(
        "mass", _G % (_REACHER_HAND_MASS * mult))


# ball_in_cup: only the ball's mass is scaled. Nothing touches the ground, so friction is inert.
_BALL_MASS = 0.06544984694978737     # compiler-derived; sphere of radius .025


def _edit_ball_mass(root, mult):
    geom = _require(root, './worldbody/body[@name="ball"]/geom[@name="ball"]',
                    "the ball_in_cup 'ball' geom")
    geom.set("mass", _G % (_BALL_MASS * mult))


# walker: limb mass, since dof_damping (0.1) is too small to matter. The torso is left alone so
# the mass distribution changes; uniform scaling would leave the dynamics nearly self-similar.
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


# Per-environment specification

class DmcSpec:
    """Everything that differs between one dm_control task and another.

    control_timestep is asserted against dm_control because it sets the episode length (it is
    0.01, 0.02 or 0.025 s depending on the domain). probes read the realised physics from the live
    model for logging. preserve names model entries owned by the task that must survive a rebuild.
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
        """The domain's nominal MJCF, as bytes."""
        from dm_control import suite
        module = getattr(suite, self.domain)
        xml, _assets = module.get_model_and_assets()
        return xml


def _walker_spec(task):
    """walker-stand and walker-walk, built from one definition so they differ only in the task."""
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
    """cartpole-swingup and cartpole-swingup_sparse, built from one definition.

    The two differ only in dm_control's `sparse` flag. Dense reward is
    `upright * small_control * small_velocity * centered`; sparse is 1 only when the cart is within
    +-0.25 of centre and the pole within 5.7 degrees of vertical, else 0.
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
        # Reacher draws the target position and radius into the model at each episode start;
        # a rebuild would reset both to the XML defaults.
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
        # Same targets as the HalfCheetah env; no rebuild needed. Mass is not offered because
        # cheetah.xml sets settotalmass="14", which would renormalise any mass change.
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


# The wrapper

class DmControlDrift(gym.Wrapper):
    """A dm_control task whose physics change per task, with a fixed, bounded reward.

    Parameters
    ----------
    env_name : str
        One of `ENV_NAMES`, i.e. dm_control's "<domain>-<task>".
    drift_targets : sequence[str] or None
        Which physics to scale. None uses the environment's defaults; unknown targets raise.
    task_multipliers : sequence[float]
        schedule="step": the multiplier held during each task. Cycles, so physics revisit.
    amplitude, period, schedule, phase, amplitude2, period2, phase2, clock_scale
        As in `LipschitzDriftHalfCheetah` and `DriftCartpoleSwingup`.
    reload_tol : float
        Continuous schedules only: rebuild the model when the multiplier has moved by more than
        this, since recompiling every step is too slow. Ignored by schedule="step" and by
        array-only targets.
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
            # A constant sequence would never change the physics at a boundary.
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
        self.t = 0                        # virtual clock in global env steps; never reset

        # The time limit is derived from the domain's control timestep so every episode is
        # exactly max_episode_steps long.
        self._max_episode_steps = int(max_episode_steps)
        dm_env = suite.load(
            spec.domain, spec.task,
            task_kwargs={"random": 0,
                         "time_limit": self._max_episode_steps * spec.control_timestep},
        )
        self._dm_env = dm_env
        self._nominal_check(dm_env)

        # Nominal MJCF, parsed fresh on every rebuild so edits never accumulate.
        self._base_xml = spec.model_xml()
        self._xml_targets = [t for t in self.drift_targets if spec.targets[t].kind == "xml"]
        self._array_targets = [t for t in self.drift_targets if spec.targets[t].kind == "array"]

        env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
        env = gym.wrappers.FlattenObservation(env)
        super().__init__(env)

        # Nominal arrays, captured before any scaling, so multipliers never compound.
        model = dm_env.physics.model
        self._nominal_arrays = {t: spec.targets[t].nominal(model) for t in self._array_targets}

        self._applied_mult = None
        self._apply(self.multiplier())

    # Guards

    def _nominal_check(self, dm_env):
        """Raise if dm_control's model or control timestep differ from the values assumed here."""
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

    # Schedule (same form as LipschitzDriftHalfCheetah)

    def multiplier(self, t=None):
        """Scaling factor applied to the nominal physics at virtual time t.

        For "step" this depends on the task index only, so `t` is ignored.
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
        """Switch to task `i` (observable boundary). Only "step" changes physics here."""
        self.task_idx = int(i)
        if self.schedule == "step":
            self._apply(self.multiplier())
        return self.task_idx

    def lipschitz_constant(self):
        """Max |change in the multiplier| per global env step.

        0 for "step": physics are constant within a task, and the jump at a boundary is not
        covered by the bound.
        """
        if self.schedule == "step":
            return 0.0
        slow = (abs(self.amplitude) * 2.0 * np.pi / self.period) if self.schedule == "sin" \
            else (abs(self.amplitude) / self.period)
        fast = abs(self.amplitude2) * 2.0 * np.pi / self.period2
        return slow + fast

    # Applying the physics change

    def _apply(self, mult):
        """Move the physics to multiplier `mult`, preserving the simulation state.

        Re-applying the same multiplier is a no-op.
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

            # The rebuild allocates fresh mjData and resets the model. Restore qpos/qvel and any
            # task-owned model entries (reacher's target) so a boundary changes only the physics.
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

        # After any rebuild, since a rebuild resets the compiled arrays to their XML values.
        for name in self._array_targets:
            self.spec_.targets[name].apply(
                physics.model, self._nominal_arrays[name], mult)

    def current_params(self):
        """Return the multiplier and the realised physics, read from the live model."""
        named = self._dm_env.physics.named.model
        out = {"drift_multiplier": float(self.multiplier())}
        if self.schedule == "step":
            out["drift_task"] = float(self.task_idx)
        for name in self.drift_targets:
            key, getter = self.spec_.probes[name]
            out[key] = _scalar(getter(named))
        return out

    # Gym API

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += self.clock_scale
        if self.schedule != "step":          # nothing to re-apply within a task
            self._apply(self.multiplier())
        # Same key as the other envs, so return logging is identical.
        info["directional_reward"] = reward
        info.update(self.current_params())
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # The physics clock is not reset with the episode.
        # shimmy seeds its own np_random but not the dm_control task's RandomState, which draws
        # the initial pose and, on reacher, the target position, so seed it here.
        seed = kwargs.get("seed")
        if seed is not None:
            self._dm_env.task._random = np.random.RandomState(int(seed))
        return self.env.reset(**kwargs)


# Factories; wrapper order matches make_drift_env / make_cartpole_env

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
    """Vectorised dm_control env, wrapper order matching make_cartpole_vector_env.

    clock_scale = num_envs so every sub-env's clock counts global env steps and all sub-envs
    follow the same schedule.
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
