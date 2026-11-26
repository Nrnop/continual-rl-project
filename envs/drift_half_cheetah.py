"""LipschitzDriftHalfCheetah: HalfCheetah with a fixed reward and changing physics.

MuJoCo parameters are scaled from their nominal values by a multiplier m. schedule="step" holds m
constant within a task and changes it in `set_task(i)`: m = task_multipliers[i % len]. The list
cycles so physics revisit. "sin"/"linear" drift continuously:
m(t) = 1 + amplitude * s(t) [+ amplitude2 * sin(2*pi*t/period2 + phase2)], with
s = sin(2*pi*t/period + phase) or t/period. The clock t counts global env steps and is not reset
between episodes.
"""
import functools

import gymnasium as gym
import numpy as np

from .directional_half_cheetah import make_base_env

# dof_frictionloss is all zeros on HalfCheetah-v5, so it cannot be scaled multiplicatively.
_TARGETS = ("damping", "friction", "mass", "armature")


class LipschitzDriftHalfCheetah(gym.Wrapper):
    """HalfCheetah with per-task or drifting physics and a fixed reward.

    Parameters
    ----------
    drift_targets : tuple[str]
        Any of "damping", "friction", "mass", "armature".
    amplitude : float
        Peak relative deviation. 0.5 means parameters swing over [0.5x, 1.5x] their nominal value.
    period : int
        Steps for one full cycle, in global env steps. Unused by "step".
    schedule : {"sin", "linear", "step"}
        "step" is piecewise constant per task; the others drift continuously.
    phase : float
        Radians, for "sin".
    task_multipliers : sequence[float]
        For schedule="step": the multiplier held during each task. Cycles, so the sequence
        revisits earlier physics and backward transfer is defined.
    clock_scale : int
        Virtual steps advanced per env.step(); set to num_envs by the vectorised factory.
    """

    def __init__(self, env_id="HalfCheetah-v5", drift_targets=("damping", "friction"),
                 amplitude=0.5, period=1228800, schedule="sin", phase=0.0,
                 amplitude2=0.0, period2=30720, phase2=0.0,
                 task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6),
                 clock_scale=1, max_episode_steps=1000, render_mode=None):
        env = make_base_env(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
        super().__init__(env)

        bad = [t for t in drift_targets if t not in _TARGETS]
        if bad:
            raise ValueError(f"unknown drift target(s) {bad}; valid: {_TARGETS}")
        if schedule not in ("sin", "linear", "step"):
            raise ValueError(
                f"unknown schedule {schedule!r}; valid: 'sin', 'linear', 'step'")

        self.task_multipliers = tuple(float(x) for x in task_multipliers)
        if schedule == "step":
            if not self.task_multipliers:
                raise ValueError("schedule='step' needs a non-empty task_multipliers")
            # A constant sequence would never change the physics at a boundary.
            if len(set(self.task_multipliers)) == 1:
                raise ValueError(
                    "schedule='step' with a constant task_multipliers "
                    f"{self.task_multipliers} would never change the physics")
        self.task_idx = 0               # which entry of task_multipliers is currently applied

        self.drift_targets = tuple(drift_targets)
        self.amplitude = float(amplitude)
        self.period = int(period)
        self.schedule = schedule
        self.phase = float(phase)
        # Optional fast component; amplitude2 = 0 gives a single-timescale drift.
        self.amplitude2 = float(amplitude2)
        self.period2 = max(int(period2), 1)
        self.phase2 = float(phase2)
        self.clock_scale = int(clock_scale)
        self.t = 0                      # virtual clock in global env steps; never reset

        # Nominal parameters. Every update rescales these, so the multiplier never compounds.
        m = self.unwrapped.model
        self._base = {
            "damping": np.array(m.dof_damping, copy=True),
            "friction": np.array(m.geom_friction, copy=True),
            "mass": np.array(m.body_mass, copy=True),
            "armature": np.array(m.dof_armature, copy=True),
        }
        self._needs_setconst = "mass" in self.drift_targets
        self._applied_mult = None       # last multiplier written to the model; see _apply
        self._apply(self.multiplier())

    # Schedule

    def multiplier(self, t=None):
        """Scaling factor applied to the nominal parameters at virtual time t.

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
        return m

    def set_task(self, i):
        """Switch to task `i` (observable boundary). Only "step" changes physics here.

        For "sin"/"linear" the index is only recorded.
        """
        self.task_idx = int(i)
        if self.schedule == "step":
            self._apply(self.multiplier())
        return self.task_idx

    def lipschitz_constant(self):
        """Max |change in the multiplier| per global env step (the eps in ||P_{t+1}-P_t|| <= eps).

        Multiply by a parameter's nominal magnitude for its absolute per-step bound. 0 for "step":
        physics are constant within a task, and the jump at a boundary is not covered by the bound.
        """
        if self.schedule == "step":
            return 0.0
        slow = (abs(self.amplitude) * 2.0 * np.pi / self.period) if self.schedule == "sin"             else (abs(self.amplitude) / self.period)
        fast = abs(self.amplitude2) * 2.0 * np.pi / self.period2
        return slow + fast

    # Applying the drift

    def _apply(self, mult):
        # Re-applying the same multiplier must be a no-op: `mj_setConst` perturbs the simulation
        # even when body_mass is unchanged, so calling it every step changes the physics.
        if self._applied_mult is not None and mult == self._applied_mult:
            return
        self._applied_mult = mult

        m = self.unwrapped.model
        if "damping" in self.drift_targets:
            m.dof_damping[:] = self._base["damping"] * mult
        if "armature" in self.drift_targets:
            m.dof_armature[:] = self._base["armature"] * mult
        if "friction" in self.drift_targets:
            # Only the sliding-friction column; the torsional/rolling columns stay nominal.
            m.geom_friction[:, 0] = self._base["friction"][:, 0] * mult
        if "mass" in self.drift_targets:
            new_mass = self._base["mass"] * mult
            # Skip when the values are unchanged (e.g. multiplier 1.0 at construction), so adding
            # "mass" to drift_targets does not perturb nominal physics.
            if not np.array_equal(np.asarray(m.body_mass), new_mass):
                m.body_mass[:] = new_mass
                # MuJoCo caches inertia-derived constants; refresh them after a mass change.
                try:
                    import mujoco
                    mujoco.mj_setConst(m, self.unwrapped.data)
                except Exception:
                    pass

    def current_params(self):
        """Diagnostics: the drift multiplier and a representative scaled value per target."""
        m = self.unwrapped.model
        out = {"drift_multiplier": float(self.multiplier())}
        if self.schedule == "step":
            out["drift_task"] = float(self.task_idx)
        if "damping" in self.drift_targets:
            out["drift_damping"] = float(np.asarray(m.dof_damping)[3])
        if "friction" in self.drift_targets:
            out["drift_friction"] = float(np.asarray(m.geom_friction)[0, 0])
        if "mass" in self.drift_targets:
            out["drift_mass"] = float(np.asarray(m.body_mass)[1])
        if "armature" in self.drift_targets:
            out["drift_armature"] = float(np.asarray(m.dof_armature)[3])
        return out

    # Gym API

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # The reward is unchanged; only the dynamics change.
        self.t += self.clock_scale
        self._apply(self.multiplier())
        info["directional_reward"] = reward      # same key as the directional env, for logging
        info.update(self.current_params())
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # The drift clock is not reset with the episode.
        return self.env.reset(**kwargs)


def _make_single_drift(env_id, drift_targets, amplitude, period, schedule, phase,
                       clock_scale, max_episode_steps, render_mode=None,
                       amplitude2=0.0, period2=30720, phase2=0.0,
                       task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6)):
    """Module-level factory so functools.partial stays picklable for AsyncVectorEnv on Windows."""
    return LipschitzDriftHalfCheetah(
        env_id=env_id, drift_targets=drift_targets, amplitude=amplitude, period=period,
        schedule=schedule, phase=phase, amplitude2=amplitude2, period2=period2, phase2=phase2,
        task_multipliers=task_multipliers,
        clock_scale=clock_scale, max_episode_steps=max_episode_steps, render_mode=render_mode,
    )


def make_drift_env(env_id="HalfCheetah-v5", drift_targets=("damping", "friction"),
                   amplitude=0.5, period=1228800, schedule="sin", phase=0.0,
                   amplitude2=0.0, period2=30720, phase2=0.0,
                   task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6), clock_scale=1,
                   max_episode_steps=1000, render_mode=None, normalize_obs=False,
                   normalize_reward=False, gamma=0.99, clip_obs=10.0, clip_reward=10.0):
    """Single drifting env, optionally with the CleanRL-style normalisation wrappers."""
    env = _make_single_drift(env_id, drift_targets, amplitude, period, schedule, phase,
                             clock_scale, max_episode_steps, render_mode,
                             amplitude2, period2, phase2, tuple(task_multipliers))
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda o: np.clip(o, -clip_obs, clip_obs), env.observation_space)
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.ClipReward(env, -clip_reward, clip_reward)
    return env


def make_drift_vector_env(env_id="HalfCheetah-v5", num_envs=1,
                          drift_targets=("damping", "friction"), amplitude=0.5,
                          period=1228800, schedule="sin", phase=0.0,
                          amplitude2=0.0, period2=30720, phase2=0.0,
                          task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6),
                          max_episode_steps=1000, gamma=0.99, normalize_obs=False,
                          normalize_reward=False, clip_obs=10.0, clip_reward=10.0,
                          asynchronous=True):
    """Vectorised drifting env. Wrapper order matches make_vector_env in the directional module.

    clock_scale = num_envs so every sub-env's clock counts global env steps and all sub-envs
    follow the same schedule.
    """
    fns = [
        functools.partial(_make_single_drift, env_id, tuple(drift_targets), amplitude, period,
                          schedule, phase, num_envs, max_episode_steps, None,
                          amplitude2, period2, phase2, tuple(task_multipliers))
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
    return envs
