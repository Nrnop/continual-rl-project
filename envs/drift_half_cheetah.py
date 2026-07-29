"""LipschitzDriftHalfCheetah — smooth, boundary-free non-stationarity in the DYNAMICS.

This is the setting the thesis proposal actually specifies, and the counterpart to
`directional_half_cheetah.py`:

    DirectionalHalfCheetah : the REWARD flips sign at discrete task boundaries.
    LipschitzDriftHalfCheetah : the REWARD is fixed; the PHYSICS drift continuously, with no
                                boundaries, no task index and no reset signal.

Formally the transition kernel satisfies ||P_{t+1} - P_t|| <= eps: physical parameters move by a
bounded amount each step. We implement that by scaling MuJoCo model parameters with a smooth
multiplier m(t) applied to their ORIGINAL values (never compounding):

    m(t) = 1 + amplitude * s(t),    s = sin(2*pi*t/period + phase)   [default]
                                    s = t/period                     [schedule="linear"]

`lipschitz_constant()` reports the implied bound (max |dm/dt| times the base magnitude), so the
eps in the proposal's formulation is a measured property of the config rather than a claim.

WHY SINUSOIDAL BY DEFAULT. A monotone ramp never revisits earlier dynamics, so "did the agent
retain anything?" is unanswerable. A cyclic schedule returns to physics the agent has already seen,
which is exactly what makes retention measurable — the drift analogue of the directional task
returning to +1 in phases 3 and 5. Use schedule="linear" for a pure never-repeating ramp.

WHICH PARAMETERS. Probed on HalfCheetah-v5:
    dof_damping      [0,0,0, 6, 4.5, 3, 4.5, 3, 1.5]  -> 6 actuated joints, scales cleanly
    geom_friction    (9,3), sliding column ~0.4        -> ground contact friction
    body_mass        7 non-zero bodies                 -> the proposal's "armature mass"
    dof_frictionloss all zeros                         -> NOT usable multiplicatively (no-op)
Defaults are damping + friction: both are read straight out of the model each step, so no derived
quantities go stale. Mass is supported but triggers a `mj_setConst` refresh, since MuJoCo caches
inertia-derived constants.

CLOCK. The drift is a function of GLOBAL env steps and must NOT reset between episodes — it is a
property of the world, not of the episode. With a vectorised env each sub-env only steps once per
global step, so the factory passes `clock_scale=num_envs`; the wrapper's virtual clock then counts
global env steps and `period` stays directly comparable to `switch` in the directional experiment.
"""
import functools

import gymnasium as gym
import numpy as np

from .directional_half_cheetah import make_base_env

# Which model fields each drift target maps to. Values are (attribute, index-or-slice-builder).
_TARGETS = ("damping", "friction", "mass", "armature")


class LipschitzDriftHalfCheetah(gym.Wrapper):
    """HalfCheetah with smoothly drifting physics and a fixed reward.

    Parameters
    ----------
    drift_targets : tuple[str]
        Any of "damping", "friction", "mass", "armature".
    amplitude : float
        Peak relative deviation. 0.5 means parameters swing over [0.5x, 1.5x] their nominal value.
    period : int
        Steps for one full cycle, in GLOBAL env steps (see CLOCK above).
    schedule : {"sin", "linear"}
    phase : float
        Radians, for "sin". Lets different runs start at different points of the cycle.
    clock_scale : int
        Virtual steps advanced per env.step() — set to num_envs by the vectorised factory.
    """

    def __init__(self, env_id="HalfCheetah-v5", drift_targets=("damping", "friction"),
                 amplitude=0.5, period=1228800, schedule="sin", phase=0.0,
                 clock_scale=1, max_episode_steps=1000, render_mode=None):
        env = make_base_env(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
        super().__init__(env)

        bad = [t for t in drift_targets if t not in _TARGETS]
        if bad:
            raise ValueError(f"unknown drift target(s) {bad}; valid: {_TARGETS}")
        if schedule not in ("sin", "linear"):
            raise ValueError(f"unknown schedule {schedule!r}; valid: 'sin', 'linear'")

        self.drift_targets = tuple(drift_targets)
        self.amplitude = float(amplitude)
        self.period = int(period)
        self.schedule = schedule
        self.phase = float(phase)
        self.clock_scale = int(clock_scale)
        self.t = 0                      # virtual clock in GLOBAL env steps; never reset

        # Snapshot the nominal parameters ONCE. Every update rescales these originals, so the
        # multiplier never compounds across steps.
        m = self.unwrapped.model
        self._base = {
            "damping": np.array(m.dof_damping, copy=True),
            "friction": np.array(m.geom_friction, copy=True),
            "mass": np.array(m.body_mass, copy=True),
            "armature": np.array(m.dof_armature, copy=True),
        }
        self._needs_setconst = "mass" in self.drift_targets
        self._apply(self.multiplier())

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------
    def multiplier(self, t=None):
        """Smooth scaling factor applied to the nominal parameters at virtual time t."""
        t = self.t if t is None else t
        if self.schedule == "sin":
            s = np.sin(2.0 * np.pi * t / self.period + self.phase)
        else:                                    # linear ramp
            s = t / self.period
        return 1.0 + self.amplitude * s

    def lipschitz_constant(self):
        """Max |change in the multiplier| per GLOBAL env step — the eps of ||P_{t+1}-P_t|| <= eps.

        Multiply by a parameter's nominal magnitude for that parameter's absolute per-step bound.
        """
        if self.schedule == "sin":
            return abs(self.amplitude) * 2.0 * np.pi / self.period
        return abs(self.amplitude) / self.period

    # ------------------------------------------------------------------
    # Applying the drift
    # ------------------------------------------------------------------
    def _apply(self, mult):
        m = self.unwrapped.model
        if "damping" in self.drift_targets:
            m.dof_damping[:] = self._base["damping"] * mult
        if "armature" in self.drift_targets:
            m.dof_armature[:] = self._base["armature"] * mult
        if "friction" in self.drift_targets:
            # Only the sliding-friction column; the torsional/rolling columns stay nominal.
            m.geom_friction[:, 0] = self._base["friction"][:, 0] * mult
        if "mass" in self.drift_targets:
            m.body_mass[:] = self._base["mass"] * mult
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
        if "damping" in self.drift_targets:
            out["drift_damping"] = float(np.asarray(m.dof_damping)[3])
        if "friction" in self.drift_targets:
            out["drift_friction"] = float(np.asarray(m.geom_friction)[0, 0])
        if "mass" in self.drift_targets:
            out["drift_mass"] = float(np.asarray(m.body_mass)[1])
        if "armature" in self.drift_targets:
            out["drift_armature"] = float(np.asarray(m.dof_armature)[3])
        return out

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Advance the world clock and re-apply the drift. The reward is UNCHANGED: unlike the
        # directional task, non-stationarity here lives entirely in the transition dynamics.
        self.t += self.clock_scale
        self._apply(self.multiplier())
        info["directional_reward"] = reward      # keep the honest-return logging path identical
        info.update(self.current_params())
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # NOTE: the drift clock deliberately survives episode resets — it is a property of the
        # world, not of the episode.
        return self.env.reset(**kwargs)


def _make_single_drift(env_id, drift_targets, amplitude, period, schedule, phase,
                       clock_scale, max_episode_steps, render_mode=None):
    """Module-level factory so functools.partial stays picklable for AsyncVectorEnv."""
    return LipschitzDriftHalfCheetah(
        env_id=env_id, drift_targets=drift_targets, amplitude=amplitude, period=period,
        schedule=schedule, phase=phase, clock_scale=clock_scale,
        max_episode_steps=max_episode_steps, render_mode=render_mode,
    )


def make_drift_env(env_id="HalfCheetah-v5", drift_targets=("damping", "friction"),
                   amplitude=0.5, period=1228800, schedule="sin", phase=0.0, clock_scale=1,
                   max_episode_steps=1000, render_mode=None, normalize_obs=False,
                   normalize_reward=False, gamma=0.99, clip_obs=10.0, clip_reward=10.0):
    """Single drifting env, optionally with the CleanRL-style normalisation wrappers."""
    env = _make_single_drift(env_id, drift_targets, amplitude, period, schedule, phase,
                             clock_scale, max_episode_steps, render_mode)
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
                          max_episode_steps=1000, gamma=0.99, normalize_obs=False,
                          normalize_reward=False, clip_obs=10.0, clip_reward=10.0,
                          asynchronous=True):
    """Vectorised drifting env. Wrapper order matches make_vector_env in the directional module.

    clock_scale = num_envs so every sub-env's virtual clock counts GLOBAL env steps and all
    sub-envs stay on an identical drift schedule.
    """
    fns = [
        functools.partial(_make_single_drift, env_id, tuple(drift_targets), amplitude, period,
                          schedule, phase, num_envs, max_episode_steps, None)
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
