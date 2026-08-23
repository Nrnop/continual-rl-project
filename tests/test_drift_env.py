"""LipschitzDriftHalfCheetah: the drift must be smooth, bounded, boundary-free and reward-preserving."""
import numpy as np
import pytest

from src_continuous_control.envs.drift_half_cheetah import (
    LipschitzDriftHalfCheetah,
    make_drift_vector_env,
)

mujoco = pytest.importorskip("mujoco", reason="drift env needs the mujoco physics model")


def _env(**kw):
    kw.setdefault("period", 1000)
    kw.setdefault("amplitude", 0.5)
    kw.setdefault("max_episode_steps", 50)
    return LipschitzDriftHalfCheetah(**kw)


def test_multiplier_is_smooth_and_respects_the_lipschitz_bound():
    """No single step may move the multiplier by more than the reported constant."""
    env = _env(schedule="sin")
    eps = env.lipschitz_constant()
    ts = np.arange(0, 4 * env.period)
    m = np.array([env.multiplier(t) for t in ts])
    steps = np.abs(np.diff(m))
    assert steps.max() <= eps + 1e-9, f"max step {steps.max():.3e} exceeds bound {eps:.3e}"
    # amplitude is a real deviation, not a no-op
    assert m.max() == pytest.approx(1.5, abs=1e-3)
    assert m.min() == pytest.approx(0.5, abs=1e-3)


def test_sin_schedule_revisits_earlier_dynamics():
    """Cyclic drift must return to previously-seen physics — that is what makes retention testable."""
    env = _env(schedule="sin")
    assert env.multiplier(0) == pytest.approx(env.multiplier(env.period), abs=1e-6)
    # a linear ramp must NOT come back
    ramp = _env(schedule="linear")
    assert ramp.multiplier(ramp.period) > ramp.multiplier(0) + 0.1


def test_physics_actually_change_and_track_the_multiplier():
    env = _env(drift_targets=("damping", "friction"), schedule="sin")
    model = env.unwrapped.model
    base_damping = float(np.asarray(model.dof_damping)[3])
    env.reset(seed=0)
    seen = []
    for _ in range(120):
        env.step(env.action_space.sample())
        seen.append(float(np.asarray(model.dof_damping)[3]))
    seen = np.array(seen)
    assert seen.std() > 0, "damping never changed — the drift is not being applied"
    # values must equal base * multiplier, i.e. rescale the ORIGINALS rather than compounding
    expected = base_damping * env.multiplier()
    assert seen[-1] == pytest.approx(expected, rel=1e-6)


def test_drift_does_not_compound_across_many_steps():
    """After a full cycle the parameters must return to nominal, not drift away."""
    env = _env(schedule="sin", period=200, max_episode_steps=10_000)
    model = env.unwrapped.model
    nominal = float(np.asarray(model.dof_damping)[3]) / env.multiplier(0)
    env.reset(seed=0)
    for _ in range(200):                      # exactly one period
        env.step(env.action_space.sample())
    assert float(np.asarray(model.dof_damping)[3]) == pytest.approx(nominal * env.multiplier(),
                                                                    rel=1e-6)


def test_clock_survives_episode_reset():
    """Drift is a property of the world, not the episode: reset must not rewind it."""
    env = _env(max_episode_steps=5)
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    t_before = env.t
    env.reset()
    assert env.t == t_before, "episode reset rewound the drift clock"


def test_reward_is_untouched_by_the_drift_wrapper():
    """Non-stationarity lives in the dynamics; the reward function must be unchanged."""
    env = _env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert info["directional_reward"] == pytest.approx(reward)
    assert "drift_multiplier" in info


def test_vector_env_clock_counts_global_steps():
    """With num_envs sub-envs, each local step must advance the clock by num_envs."""
    num_envs = 4
    envs = make_drift_vector_env(num_envs=num_envs, period=4000, amplitude=0.3,
                                 max_episode_steps=50, asynchronous=False)
    envs.reset(seed=0)
    for _ in range(3):
        envs.step(np.stack([envs.single_action_space.sample() for _ in range(num_envs)]))
    ts = envs.get_attr("t") if hasattr(envs, "get_attr") else None
    if ts is not None:
        assert all(t == 3 * num_envs for t in ts), f"clocks out of sync with global steps: {ts}"
    envs.close()


def test_rejects_unknown_target_or_schedule():
    with pytest.raises(ValueError):
        _env(drift_targets=("not_a_parameter",))
    with pytest.raises(ValueError):
        _env(schedule="zigzag")


# ----------------------------------------------------------------------
# schedule="step" — the Phase 2a semi-continual mode: physics are piecewise
# constant, changing only at an observable boundary via set_task(i).
# ----------------------------------------------------------------------
PHASE2_TASKS = (1.0, 1.6, 0.6, 1.6, 0.6)


def _step_env(**kw):
    kw.setdefault("schedule", "step")
    kw.setdefault("task_multipliers", PHASE2_TASKS)
    return _env(**kw)


def _damping(env):
    return float(np.asarray(env.unwrapped.model.dof_damping)[3])


def test_step_physics_are_constant_within_a_task_and_change_at_the_boundary():
    """The failure that cost a week in Phase 1: an env that never actually changes."""
    env = _step_env()
    env.reset(seed=0)
    within = []
    for _ in range(60):
        env.step(env.action_space.sample())
        within.append(_damping(env))
    assert np.std(within) == 0.0, "damping moved inside a task — this is not a step schedule"

    before = _damping(env)
    env.set_task(1)
    after = _damping(env)
    assert after != before, "set_task did not change the physics"
    assert after == pytest.approx(before / PHASE2_TASKS[0] * PHASE2_TASKS[1], rel=1e-9)

    # ...and it stays put again after the boundary.
    for _ in range(30):
        env.step(env.action_space.sample())
    assert _damping(env) == pytest.approx(after, rel=1e-12)


def test_step_multiplier_ignores_the_clock():
    """Physics are a function of the task index, not of t."""
    env = _step_env()
    assert env.multiplier(0) == env.multiplier(10_000) == PHASE2_TASKS[0]
    env.set_task(2)
    assert env.multiplier(0) == env.multiplier(10_000) == PHASE2_TASKS[2]


def test_step_task_sequence_revisits_earlier_physics():
    """BWT is undefined unless tasks repeat: 2/4 and 3/5 must be the same physics."""
    env = _step_env()
    seen = []
    for i in range(len(PHASE2_TASKS)):
        env.set_task(i)
        seen.append(_damping(env))
    assert seen[1] == pytest.approx(seen[3], rel=1e-12)
    assert seen[2] == pytest.approx(seen[4], rel=1e-12)
    assert len(set(np.round(seen, 9))) > 1, "the sequence never changes the physics"


def test_step_task_index_cycles():
    """Indices past the end wrap, so a longer run keeps revisiting the same set."""
    env = _step_env()
    env.set_task(0)
    first = _damping(env)
    env.set_task(len(PHASE2_TASKS))
    assert _damping(env) == pytest.approx(first, rel=1e-12)


def test_step_reports_zero_lipschitz_and_the_task_index():
    env = _step_env()
    assert env.lipschitz_constant() == 0.0        # constant within a task, by construction
    env.set_task(3)
    params = env.current_params()
    assert params["drift_task"] == 3
    assert params["drift_multiplier"] == pytest.approx(PHASE2_TASKS[3])


def test_step_rejects_a_constant_task_sequence():
    """A single repeated multiplier means no non-stationarity at all — refuse it loudly."""
    with pytest.raises(ValueError):
        _step_env(task_multipliers=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        _step_env(task_multipliers=())


def _reward_trace(drift_targets, n=120, task=0):
    """Rewards from a fixed action sequence, from a fixed seed — a physics fingerprint."""
    env = LipschitzDriftHalfCheetah(drift_targets=drift_targets, schedule="step",
                                    task_multipliers=PHASE2_TASKS, max_episode_steps=1000)
    env.set_task(task)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    rewards = [env.step(rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32))[1]
               for _ in range(n)]
    env.close()
    return np.asarray(rewards)


def test_a_multiplier_of_one_is_a_true_no_op_for_every_target():
    """Task 0 is x1.0, so ALL drift_targets must give byte-identical physics there.

    REGRESSION TEST. `mass` used to fail this: `_apply` ran on every env step, and its
    `mj_setConst` inertia refresh perturbed the simulation even when body_mass was written back
    unchanged. The trajectory diverged outright (mean x_velocity -0.93 vs -0.12 over the same
    actions), and it made the "harder" four-parameter variant score ~3500 where the two-parameter
    one scored ~-200 — an easier task wearing a harder task's name. A physics change that happens
    when nothing changed is the same class of defect as a physics change that never happens.
    """
    baseline = _reward_trace(("damping", "friction"))
    for extra in (("armature",), ("mass",), ("mass", "armature")):
        trace = _reward_trace(("damping", "friction") + extra)
        assert np.allclose(baseline, trace), (
            f"adding {extra} changed the physics at multiplier x1.0, where it must be a no-op "
            f"(total reward {trace.sum():.2f} vs {baseline.sum():.2f})")


def test_re_applying_the_same_multiplier_changes_nothing():
    """`_apply` must be idempotent — it runs once per env step under the continuous schedules."""
    env = _step_env(drift_targets=("damping", "friction", "mass", "armature"))
    env.set_task(1)
    before = (np.array(env.unwrapped.model.dof_damping, copy=True),
              np.array(env.unwrapped.model.body_mass, copy=True))
    for _ in range(5):
        env._apply(env.multiplier())
    assert np.array_equal(np.asarray(env.unwrapped.model.dof_damping), before[0])
    assert np.array_equal(np.asarray(env.unwrapped.model.body_mass), before[1])
    env.close()


def test_mass_and_armature_really_do_move_at_a_boundary():
    """The other half of the guard: a real multiplier change must still reach every target."""
    env = _step_env(drift_targets=("damping", "friction", "mass", "armature"))
    env.set_task(0)
    nominal = env.current_params()
    env.set_task(1)                              # x1.6
    scaled = env.current_params()
    for key in ("drift_damping", "drift_friction", "drift_mass", "drift_armature"):
        assert scaled[key] == pytest.approx(nominal[key] * 1.6, rel=1e-9), \
            f"{key} did not scale at the boundary"
    env.close()


def test_step_set_task_propagates_through_the_vector_env():
    """train.py switches tasks with env.unwrapped.call('set_task', i) — every sub-env must move."""
    num_envs = 3
    envs = make_drift_vector_env(num_envs=num_envs, schedule="step",
                                 task_multipliers=PHASE2_TASKS, max_episode_steps=50,
                                 asynchronous=False)
    envs.reset(seed=0)
    envs.unwrapped.call("set_task", 1)
    mults = [p["drift_multiplier"] for p in envs.unwrapped.call("current_params")]
    assert mults == [pytest.approx(PHASE2_TASKS[1])] * num_envs
    envs.close()
