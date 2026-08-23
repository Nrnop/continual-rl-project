"""LipschitzDriftHalfCheetah — smooth, boundary-free non-stationarity in the DYNAMICS.

This is the setting the thesis proposal actually specifies, and the counterpart to
`directional_half_cheetah.py`:

    DirectionalHalfCheetah : the REWARD flips sign at discrete task boundaries.
    LipschitzDriftHalfCheetah : the REWARD is fixed; the PHYSICS drift continuously, with no
                                boundaries, no task index and no reset signal.

Formally the transition kernel satisfies ||P_{t+1} - P_t|| <= eps: physical parameters move by a
bounded amount each step. We implement that by scaling MuJoCo model parameters with a smooth
multiplier m(t) applied to their ORIGINAL values (never compounding):

    m(t) = 1 + amplitude * s(t) [+ amplitude2 * sin(2*pi*t/period2 + phase2)]

with s = sin(2*pi*t/period + phase) by default, or t/period for schedule="linear".

PHASE 2a USES schedule="step" INSTEAD, which is NOT a drift at all: the multiplier is held
constant within a task and changed only when `set_task(i)` is called at an observable boundary
(the paper's *semi-continual* setting). The multiplier is then a function of the TASK INDEX, not
of the clock:

    m = task_multipliers[i % len(task_multipliers)]

The list cycles deliberately, so tasks revisit physics the agent has already seen — with
`[1.0, 1.6, 0.6, 1.6, 0.6]`, tasks 2/4 and 3/5 are the same physics. Backward transfer is
undefined on a sequence that never revisits. `sin`/`linear` stay intact for Phase 2b, which
removes boundaries again.

The OPTIONAL SECOND COMPONENT is what makes this a real test of a dual-timescale method. With one
slow component the physics move ~0.5% per PPO update at the default period — far slower than a
single critic tracks, so a permanent/transient split has nothing to do and PT necessarily ties the
baseline. Adding a fast component gives the regime PT is actually designed for: a slow structural
trend the PERMANENT part should capture, plus a fast fluctuation the TRANSIENT part should absorb
without contaminating the permanent ("filtering out temporary noise").

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
        Steps for one full cycle, in GLOBAL env steps (see CLOCK above). Unused by "step".
    schedule : {"sin", "linear", "step"}
        "step" is the Phase 2a piecewise-constant mode; see the module docstring.
    phase : float
        Radians, for "sin". Lets different runs start at different points of the cycle.
    task_multipliers : sequence[float]
        For schedule="step": the multiplier held during each task. Cycles, so the sequence
        revisits earlier physics and backward transfer stays well-defined.
    clock_scale : int
        Virtual steps advanced per env.step() — set to num_envs by the vectorised factory.
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
            # A single repeated value means the physics never actually change at a boundary —
            # a silently-constant env looks exactly like a working experiment (Phase 1 lost a
            # week to this), so refuse it rather than run it.
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
        # Optional fast component (0 amplitude = single-timescale drift, the original behaviour).
        self.amplitude2 = float(amplitude2)
        self.period2 = max(int(period2), 1)
        self.phase2 = float(phase2)
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
        self._applied_mult = None       # last multiplier written to the model; see _apply
        self._apply(self.multiplier())

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------
    def multiplier(self, t=None):
        """Scaling factor applied to the nominal parameters at virtual time t.

        For "step" the multiplier is a function of the TASK INDEX, not of the clock, so `t` is
        ignored — the physics are constant within a task and change only in `set_task`.
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
        """Switch to task `i` (semi-continual boundary). Only "step" changes physics here.

        For "sin"/"linear" the multiplier is driven by the clock, so there is nothing for a
        boundary to do; the index is still recorded, and `current_params()` reports it, so a
        misconfigured run shows up in the logs rather than looking like a working experiment.
        """
        self.task_idx = int(i)
        if self.schedule == "step":
            self._apply(self.multiplier())
        return self.task_idx

    def lipschitz_constant(self):
        """Max |change in the multiplier| per GLOBAL env step — the eps of ||P_{t+1}-P_t|| <= eps.

        Multiply by a parameter's nominal magnitude for that parameter's absolute per-step bound.

        For "step" this is 0: the physics are constant *within* a task. The discontinuity at a
        boundary is deliberate and is not covered by the bound — its size is
        max |task_multipliers[i+1] - task_multipliers[i]|.
        """
        if self.schedule == "step":
            return 0.0
        slow = (abs(self.amplitude) * 2.0 * np.pi / self.period) if self.schedule == "sin"             else (abs(self.amplitude) / self.period)
        fast = abs(self.amplitude2) * 2.0 * np.pi / self.period2
        return slow + fast

    # ------------------------------------------------------------------
    # Applying the drift
    # ------------------------------------------------------------------
    def _apply(self, mult):
        # RE-APPLYING THE SAME MULTIPLIER MUST BE A NO-OP, and for "mass" it was not.
        #
        # `mj_setConst` recomputes MuJoCo's inertia-derived constants. Calling it every step —
        # which is what happened, because `step` re-applied unconditionally — perturbs the
        # simulation even when body_mass is written back unchanged. Measured: with the multiplier
        # pinned at 1.0, so that the physics are nominal by construction, adding "mass" to
        # drift_targets changed the trajectory outright (mean x_velocity -0.93 vs -0.12 over the
        # same 150 actions from the same seed) and made the task LOOK EASIER — vanilla was scoring
        # ~3500 at 80k steps against ~-200 without it. A "harder" variant that is really a
        # different, easier physics is exactly the kind of silent confound this project keeps
        # hitting, so the guard below is the fix: skip the work entirely when nothing changed.
        #
        # It is also free performance: under schedule="step" the multiplier is constant within a
        # task, so the physics are now written once per boundary instead of once per env step.
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
            # Second guard, on the VALUES rather than the multiplier: a multiplier of exactly 1.0
            # leaves body_mass identical, and there is then nothing to refresh. Without this,
            # constructing the env with mass in drift_targets perturbs it before the run even
            # starts, so "same physics" would not be the same physics across two configs.
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
                       clock_scale, max_episode_steps, render_mode=None,
                       amplitude2=0.0, period2=30720, phase2=0.0,
                       task_multipliers=(1.0, 1.6, 0.6, 1.6, 0.6)):
    """Module-level factory so functools.partial stays picklable for AsyncVectorEnv."""
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

    clock_scale = num_envs so every sub-env's virtual clock counts GLOBAL env steps and all
    sub-envs stay on an identical drift schedule.
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
