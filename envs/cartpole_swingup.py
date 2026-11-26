"""DriftCartpoleSwingup: dm_control cartpole-swingup with per-task pole length and mass.

The reward is in [0, 1] per step over exactly 1000 steps with no early termination, so the maximum
return is 1000. The reward does not depend on pole geometry, so every task has the same ceiling.
Pole physics change by patching the MJCF and recompiling, so MuJoCo recomputes mass, inertia and
COM; qpos/qvel are restored across the rebuild so a boundary changes only the physics.
"""
import functools

import gymnasium as gym
import numpy as np
from lxml import etree

# Shared with the HalfCheetah envs so eval/probe envs produce the same observation layout.
from .directional_half_cheetah import TaskIDObservation, TaskIDObservationSingle

_TARGETS = ("pole_length", "pole_mass")

# Nominal values in dm_control's cartpole.xml, checked against the live model in `_nominal_check`.
_NOMINAL_LENGTH = 1.0     # <geom ... fromto="0 0 0 0 0 1">  -> a pole of length 1.0
_NOMINAL_MASS = 0.1       # <geom ... mass=".1">

# Indices into the compiled model, fixed by cartpole.xml and checked by `_nominal_check`.
_POLE_BODY = 2
_POLE_GEOM = 4


def _patched_model_xml(length_mult=1.0, mass_mult=1.0, num_poles=1):
    """Return dm_control's cartpole MJCF with the pole's length and mass scaled.

    Both attributes live on the `pole` default class, so one edit covers every pole in the chain.
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
        A dm_control cartpole task: "swingup" (default), "swingup_sparse" or "two_poles".
    drift_targets : tuple[str]
        Any of "pole_length", "pole_mass".
    task_multipliers : sequence[float]
        schedule="step": the multiplier held during each task. Cycles, so physics revisit.
    amplitude, period, schedule, phase, amplitude2, period2, phase2, clock_scale
        As in `LipschitzDriftHalfCheetah`.
    reload_tol : float
        Continuous schedules only: rebuild the model when the multiplier has moved by more than
        this, since recompiling every step is too slow. The drift is then a fine staircase.
        Ignored by schedule="step", which rebuilds at every boundary.
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
            # A constant sequence would never change the physics at a boundary.
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
        self.t = 0                      # virtual clock in global env steps; never reset

        # control_timestep is 0.01 s, so a time limit of max_episode_steps/100 seconds truncates
        # at exactly max_episode_steps. dm_control's own truncation is the only episode end.
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

    # Guards

    @staticmethod
    def _nominal_check(physics):
        """Raise if dm_control's nominal cartpole differs from the values the multipliers scale."""
        pole_mass = float(np.asarray(physics.model.body_mass)[_POLE_BODY])
        half_len = float(np.asarray(physics.model.geom_size)[_POLE_GEOM][1])
        if not (np.isclose(pole_mass, _NOMINAL_MASS)
                and np.isclose(half_len, _NOMINAL_LENGTH / 2)):
            raise RuntimeError(
                "dm_control's nominal cartpole differs from this wrapper's assumptions "
                "(pole mass %s vs %s, half-length %s vs %s); the multipliers would mean "
                "something different" % (pole_mass, _NOMINAL_MASS, half_len, _NOMINAL_LENGTH / 2))

    # Schedule (same form as LipschitzDriftHalfCheetah)

    def multiplier(self, t=None):
        """Scaling factor applied to the nominal pole at virtual time t.

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
        """Rebuild the model at multiplier `mult`, preserving qpos/qvel.

        Re-applying the same multiplier is a no-op, so the model is not rebuilt every step.
        """
        tol = 0.0 if self.schedule == "step" else self.reload_tol
        if self._applied_mult is not None and abs(mult - self._applied_mult) <= tol:
            return
        self._applied_mult = mult

        length_mult = mult if "pole_length" in self.drift_targets else 1.0
        mass_mult = mult if "pole_mass" in self.drift_targets else 1.0

        from dm_control.suite import common
        physics = self._dm_env.physics
        # The rebuild allocates fresh mjData, which would reset the cart to the origin. Restore
        # the state so a boundary changes the physics and nothing else.
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
        """Return the multiplier and the realised pole parameters, read from the live model."""
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

    # Gym API

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += self.clock_scale
        if self.schedule != "step":          # nothing to re-apply within a task
            self._apply(self.multiplier())
        # Same key as the HalfCheetah envs, so return logging is identical.
        info["directional_reward"] = reward
        info.update(self.current_params())
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # The physics clock is not reset with the episode.
        # shimmy seeds its own np_random but not the dm_control task's RandomState, which draws
        # the initial pole angle and cart offset, so seed it here.
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
    """Vectorised cartpole, wrapper order matching make_drift_vector_env.

    clock_scale = num_envs so every sub-env's clock counts global env steps and all sub-envs
    follow the same schedule.
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
