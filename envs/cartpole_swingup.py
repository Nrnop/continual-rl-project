"""DriftCartpoleSwingup — the SECOND environment, and the counterpart to `drift_half_cheetah.py`.

WHY THIS FILE EXISTS. Every result in this project (~154 runs) is on HalfCheetah, so nothing in
the data can distinguish "a property of the PT method" from "a property of HalfCheetah". This is
the maximally-different second task that separates those two readings:

    HalfCheetah          6 actuators, 17 observations, gait (a limit cycle), unbounded reward
    cartpole-swingup     1 actuator,   5 observations, stabilization (a point attractor), r in [0,1]

THE CEILING IS KNOWN BY CONSTRUCTION, which is the deciding reason for this env. DM Control's
reward is in [0,1] per step over exactly 1000 steps and the episode NEVER terminates early, so the
maximum return is always **1000** and a result reads as "87% of optimal" rather than "1689 — is
that good?". This project has repeatedly been damaged by not knowing its own ceiling; here that
class of failure cannot occur.

Measured reference points on the nominal model, so the numbers above are not a claim:
    random uniform policy   ~48 / 1000
    hanging at rest         ~0  / 1000   (upright = (cos(pi)+1)/2 = 0)

THE REWARD DOES NOT DEPEND ON THE PHYSICS WE CHANGE. `Balance._get_reward` is a product of
`upright`, `centered`, `small_control` and `small_velocity` — pole geometry appears in none of
them. So unlike HalfCheetah, where scaling damping changes the achievable velocity and therefore
the achievable return, here every task in the sequence has the SAME ceiling of 1000 and returns are
directly comparable across the sequence. (One honest caveat: `small_velocity` scores the pole's
angular velocity against a fixed margin of 5 rad/s, and a longer pole rotates more slowly, so a
long pole is very slightly favoured on that one factor. It is a second-order effect on one of four
multiplicands, not a rescaling of the task.)

NON-STATIONARITY. Identical in structure to `LipschitzDriftHalfCheetah`, so the two benchmarks are
driven by the same config keys and the same training-loop code path:

    schedule="step"      Phase 2a. The multiplier is a function of the TASK INDEX, held constant
                         within a task and changed only in `set_task(i)` at an observable boundary.
    schedule="sin"/"linear"
                         Phase 2b. Continuous drift, no boundaries. See RECOMPILE COST below —
                         these are quantized here, which the HalfCheetah wrapper does not need to
                         do.

`task_multipliers` cycles, so tasks REVISIT physics the agent has already seen. Backward transfer
is undefined on a sequence that never revisits.

WHICH PARAMETERS, AND WHY WE RECOMPILE THE MODEL RATHER THAN PATCH ARRAYS.

`pole_length` and `pole_mass`, which is what the task brief asks for: the permanent component's job
can then be stated in one sentence — hold *swing up, then balance*, while the transient corrects
for this particular pole. Nobody on this project has ever been able to say what the permanent is
supposed to store on HalfCheetah, and being able to state the hypothesis is a precondition for
testing it.

The HalfCheetah wrapper scales `model.dof_damping` and `model.geom_friction` in place, which is
safe because those fields are read straight out of the model every step and nothing is derived
from them. **Pole length is not like that.** Changing `geom_size` alone leaves `body_mass`,
`body_inertia` and `body_ipos` describing the OLD pole, so the simulation would be internally
inconsistent — and the hand-derived capsule inertia needed to fix it up is exactly the class of
silent error this project keeps paying for (failure mode #2 in CLAUDE.md: a manipulation that
quietly does not do what it says).

So instead we patch the MJCF and let MuJoCo's own compiler recompute mass, inertia and COM from
the geometry. Verified when this file was written:

    multiplier   pole mass   half-length   body_inertia (transverse)
    x1.0         0.10        0.50          0.00942459     (identical to nominal — an exact no-op)
    x1.6         0.16        0.80          0.03683970
    x0.6         0.06        0.30          0.00221453

and, driving the SAME 300 random actions from the SAME seed, the trajectories genuinely diverge
(max |qpos difference| 0.23 at x1.6 and 0.55 at x0.6; return 1.68 / 0.66 / 2.59). A
multiplicatively-inert parameter looks exactly like a working experiment and cost Phase 1 a week —
`tests/test_cartpole_env.py` pins both halves of this.

RECOMPILE COST. `reload_from_xml_string` rebuilds the model, so it is far more expensive than
writing a float into an array. Under schedule="step" it runs FIVE times in a 3M-step run, which is
free. Under "sin"/"linear" a naive implementation would recompile on every env step, so those
schedules quantize: the model is rebuilt only when the multiplier has moved by more than
`reload_tol`. The drift is then a fine staircase rather than a true continuum. That is a real
(if small) departure from the HalfCheetah wrapper's behaviour and is stated here rather than
buried; Phase 2a does not use it.

STATE IS PRESERVED ACROSS A REBUILD. `reload_from_xml_string` allocates fresh `mjData`, which
would teleport the cart back to the origin at every boundary — a hidden episode reset that no
config key asks for. We snapshot `qpos`/`qvel`, rebuild, write them back and re-run `forward()`,
so a boundary changes the physics and nothing else, exactly as the HalfCheetah wrapper does.

SEEDING. shimmy's `reset(seed=...)` seeds its own `np_random` but NOT the dm_control task's
`RandomState`, which is what actually draws the initial pole angle. Left alone, every sub-env of a
vectorised run would start from an unseeded stream and the run would not be reproducible from
`--seed`. `reset` below re-seeds the task directly.
"""
import functools

import gymnasium as gym
import numpy as np
from lxml import etree

# Reused verbatim from the reward-flip benchmark: neither wrapper knows anything about HalfCheetah,
# and the eval/probe envs must produce the same observation layout as the training env or the
# policy cannot be run on them at all.
from .directional_half_cheetah import TaskIDObservation, TaskIDObservationSingle

# What the multiplier is allowed to scale. Both are verified non-inert in tests/test_cartpole_env.py.
_TARGETS = ("pole_length", "pole_mass")

# The nominal values in dm_control's cartpole.xml, which the multiplier scales. Asserted against
# the live model in `_nominal_check` so a dm_control upgrade that changes the model cannot silently
# rescale the whole benchmark.
_NOMINAL_LENGTH = 1.0     # <geom ... fromto="0 0 0 0 0 1">  -> a pole of length 1.0
_NOMINAL_MASS = 0.1       # <geom ... mass=".1">

# Indices into the compiled model. Fixed by dm_control's cartpole.xml and re-checked by
# `_nominal_check`, which fails loudly if the model is ever not the one we think it is.
_POLE_BODY = 2
_POLE_GEOM = 4


def _patched_model_xml(length_mult=1.0, mass_mult=1.0, num_poles=1):
    """dm_control's cartpole MJCF with the pole's length and mass scaled.

    Both attributes live on the `pole` default class, so a single edit covers every pole in the
    chain and the two-pole fallback variant needs no special case.
    """
    from dm_control.suite import cartpole

    root = etree.fromstring(cartpole._make_model(num_poles))
    geom = root.find('./default/default[@class="pole"]/geom')
    if geom is None:                      # dm_control restructured the model
        raise RuntimeError(
            "could not find the 'pole' default geom in dm_control's cartpole.xml; "
            "the model changed and this wrapper's scaling is no longer valid")
    geom.set("fromto", "0 0 0 0 0 %.10g" % (_NOMINAL_LENGTH * length_mult))
    geom.set("mass", "%.10g" % (_NOMINAL_MASS * mass_mult))
    return etree.tostring(root)


class DriftCartpoleSwingup(gym.Wrapper):
    """dm_control cartpole with per-task pole physics and a fixed, bounded reward.

    Parameters
    ----------
    task_name : str
        A dm_control cartpole task: "swingup" (the default), "swingup_sparse" or "two_poles".
        The sparse variants are the documented fallback if plain swingup saturates.
    drift_targets : tuple[str]
        Any of "pole_length", "pole_mass". Both by default: a longer, heavier pole is harder to
        swing up, so the two compound into a clearly-separated task sequence.
    task_multipliers : sequence[float]
        schedule="step": the multiplier held during each task. Cycles, so physics revisit.
    amplitude, period, schedule, phase, amplitude2, period2, phase2, clock_scale
        Exactly as in `LipschitzDriftHalfCheetah`; see that module and RECOMPILE COST above.
    reload_tol : float
        Continuous schedules only: rebuild the model when the multiplier has moved by more than
        this. Ignored by schedule="step", which rebuilds on every boundary regardless.
    """

    def __init__(self, task_name="swingup", drift_targets=("pole_length", "pole_mass"),
                 amplitude=0.5, period=1228800, schedule="step", phase=0.0,
                 amplitude2=0.0, period2=30720, phase2=0.0,
                 task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6),
                 clock_scale=1, max_episode_steps=1000, render_mode=None,
                 reload_tol=0.005):
        from dm_control import suite
        from shimmy import DmControlCompatibilityV0

        bad = [t for t in drift_targets if t not in _TARGETS]
        if bad:
            raise ValueError("unknown drift target(s) %s; valid: %s" % (bad, (_TARGETS,)))
        if schedule not in ("sin", "linear", "step"):
            raise ValueError("unknown schedule %r; valid: 'sin', 'linear', 'step'" % (schedule,))
        if not drift_targets:
            raise ValueError("drift_targets is empty; the physics would never change")

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

        self.task_name = str(task_name)
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
        self.t = 0                      # virtual clock in GLOBAL env steps; never reset

        # dm_control owns the episode length: control_timestep is 0.01 s, so a time limit of
        # max_episode_steps/100 seconds truncates at exactly max_episode_steps steps. Doing it here
        # rather than with a gym TimeLimit keeps the dm_control TimeStep's own truncation flag
        # authoritative, so there is only one place an episode can end.
        self._max_episode_steps = int(max_episode_steps)
        dm_env = suite.load(
            "cartpole", self.task_name,
            task_kwargs={"random": 0, "time_limit": self._max_episode_steps / 100.0},
        )
        self._dm_env = dm_env
        self._nominal_check(dm_env.physics)

        env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
        env = gym.wrappers.FlattenObservation(env)   # Dict(position(3), velocity(2)) -> Box(5,)
        super().__init__(env)

        self._applied_mult = None
        self._apply(self.multiplier())

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
    @staticmethod
    def _nominal_check(physics):
        """Fail loudly if dm_control's cartpole model is not the one this wrapper scales.

        The multipliers are relative to hard-coded nominal values, so a dm_control release that
        changed the pole would silently rescale every task in the sequence at once — a whole
        benchmark quietly redefined, with nothing in the logs to show it.
        """
        pole_mass = float(np.asarray(physics.model.body_mass)[_POLE_BODY])
        half_len = float(np.asarray(physics.model.geom_size)[_POLE_GEOM][1])
        if not (np.isclose(pole_mass, _NOMINAL_MASS)
                and np.isclose(half_len, _NOMINAL_LENGTH / 2)):
            raise RuntimeError(
                "dm_control's nominal cartpole differs from this wrapper's assumptions "
                "(pole mass %s vs %s, half-length %s vs %s); the multipliers would mean "
                "something different" % (pole_mass, _NOMINAL_MASS, half_len, _NOMINAL_LENGTH / 2))

    # ------------------------------------------------------------------
    # Schedule  (identical in form to LipschitzDriftHalfCheetah)
    # ------------------------------------------------------------------
    def multiplier(self, t=None):
        """Scaling factor applied to the nominal pole at virtual time t.

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
        """Rebuild the model at multiplier `mult`, preserving the simulation state.

        Re-applying the same multiplier is a no-op, for the same reason as in the HalfCheetah
        wrapper but with more force: a rebuild here would otherwise discard `mjData` every step.
        """
        tol = 0.0 if self.schedule == "step" else self.reload_tol
        if self._applied_mult is not None and abs(mult - self._applied_mult) <= tol:
            return
        self._applied_mult = mult

        length_mult = mult if "pole_length" in self.drift_targets else 1.0
        mass_mult = mult if "pole_mass" in self.drift_targets else 1.0

        from dm_control.suite import common
        physics = self._dm_env.physics
        # Snapshot the state so a boundary changes the physics and NOTHING else. Without this the
        # rebuild's fresh mjData teleports the cart to the origin and zeroes the pole's velocity —
        # a free episode reset that no config asks for, handed to every arm at every boundary.
        qpos = np.array(physics.data.qpos, copy=True)
        qvel = np.array(physics.data.qvel, copy=True)
        physics.reload_from_xml_string(
            _patched_model_xml(length_mult, mass_mult,
                               num_poles=2 if self.task_name == "two_poles" else 1),
            assets=common.ASSETS)
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
        physics.forward()

    def current_params(self):
        """Diagnostics: the multiplier and the REALISED pole parameters read out of the model.

        Read from the live model rather than recomputed from the config, so a manipulation that
        silently failed shows up here instead of looking like a working experiment.
        """
        m = self._dm_env.physics.model
        out = {"drift_multiplier": float(self.multiplier())}
        if self.schedule == "step":
            out["drift_task"] = float(self.task_idx)
        if "pole_length" in self.drift_targets:
            out["drift_pole_length"] = float(np.asarray(m.geom_size)[_POLE_GEOM][1] * 2.0)
        if "pole_mass" in self.drift_targets:
            out["drift_pole_mass"] = float(np.asarray(m.body_mass)[_POLE_BODY])
        out["drift_pole_inertia"] = float(np.asarray(m.body_inertia)[_POLE_BODY][0])
        return out

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += self.clock_scale
        if self.schedule != "step":          # nothing to re-apply within a task
            self._apply(self.multiplier())
        # Same key the other two benchmarks use, so the honest-return logging path is identical.
        info["directional_reward"] = reward
        info.update(self.current_params())
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # The physics clock deliberately survives episode resets — it is a property of the world,
        # not of the episode.
        #
        # Seeding is ours to do: shimmy seeds its own np_random and never touches the dm_control
        # task's RandomState, which is what actually draws the initial pole angle and cart offset.
        seed = kwargs.get("seed")
        if seed is not None:
            self._dm_env.task._random = np.random.RandomState(int(seed))
        return self.env.reset(**kwargs)


def _make_single_cartpole(task_name, drift_targets, amplitude, period, schedule, phase,
                          clock_scale, max_episode_steps, render_mode=None,
                          amplitude2=0.0, period2=30720, phase2=0.0,
                          task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6), reload_tol=0.005):
    """Module-level factory so functools.partial stays picklable for AsyncVectorEnv on Windows."""
    return DriftCartpoleSwingup(
        task_name=task_name, drift_targets=drift_targets, amplitude=amplitude, period=period,
        schedule=schedule, phase=phase, amplitude2=amplitude2, period2=period2, phase2=phase2,
        task_multipliers=task_multipliers, clock_scale=clock_scale,
        max_episode_steps=max_episode_steps, render_mode=render_mode, reload_tol=reload_tol,
    )


def make_cartpole_env(task_name="swingup", drift_targets=("pole_length", "pole_mass"),
                      amplitude=0.5, period=1228800, schedule="step", phase=0.0,
                      amplitude2=0.0, period2=30720, phase2=0.0,
                      task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6), clock_scale=1,
                      max_episode_steps=1000, render_mode=None, normalize_obs=False,
                      normalize_reward=False, gamma=0.99, clip_obs=10.0, clip_reward=10.0,
                      reload_tol=0.005, task_id_obs=False, n_task_ids=5):
    """Single cartpole env, wrapper order matching make_drift_env / make_directional_env."""
    env = _make_single_cartpole(task_name, tuple(drift_targets), amplitude, period, schedule,
                                phase, clock_scale, max_episode_steps, render_mode,
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


def make_cartpole_vector_env(task_name="swingup", num_envs=1,
                             drift_targets=("pole_length", "pole_mass"), amplitude=0.5,
                             period=1228800, schedule="step", phase=0.0,
                             amplitude2=0.0, period2=30720, phase2=0.0,
                             task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6),
                             max_episode_steps=1000, gamma=0.99, normalize_obs=False,
                             normalize_reward=False, clip_obs=10.0, clip_reward=10.0,
                             asynchronous=True, reload_tol=0.005,
                             task_id_obs=False, n_task_ids=5):
    """Vectorised cartpole. Wrapper order matches make_drift_vector_env exactly.

    clock_scale = num_envs so every sub-env's virtual clock counts GLOBAL env steps and all
    sub-envs stay on an identical schedule.
    """
    fns = [
        functools.partial(_make_single_cartpole, task_name, tuple(drift_targets), amplitude,
                          period, schedule, phase, num_envs, max_episode_steps, None,
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
