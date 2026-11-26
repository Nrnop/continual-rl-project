"""Tests for DriftCartpoleSwingup.

Each task must change the pole's physics and nothing else, and the return ceiling must be exactly
1000 in every task.
"""
import numpy as np
import pytest

pytest.importorskip("dm_control", reason="the cartpole benchmark needs dm_control")
pytest.importorskip("shimmy", reason="the cartpole benchmark needs shimmy")

from src_continuous_control.envs.cartpole_swingup import (   # noqa: E402
    DriftCartpoleSwingup,
    make_cartpole_vector_env,
)

PHASE2_TASKS = (1.0, 1.6, 0.6, 1.6, 0.6)


def _env(**kw):
    kw.setdefault("max_episode_steps", 100)
    kw.setdefault("task_multipliers", PHASE2_TASKS)
    return DriftCartpoleSwingup(**kw)


# Physics change at a boundary and stay constant within a task.
def test_physics_differ_between_tasks():
    """Distinct multipliers give distinct physics, and revisited tasks give identical physics."""
    env = _env()
    seen = {}
    for i in range(len(PHASE2_TASKS)):
        env.set_task(i)
        p = env.current_params()
        seen[i] = (p["drift_pole_length"], p["drift_pole_mass"], p["drift_pole_inertia"])
    # Tasks 1/3 and 2/4 revisit the same physics, which backward transfer relies on.
    assert seen[1] == seen[3]
    assert seen[2] == seen[4]
    assert len({seen[0], seen[1], seen[2]}) == 3
    env.close()


def test_physics_are_constant_within_a_task():
    """With schedule='step' the physics do not change between boundaries."""
    env = _env()
    env.set_task(1)
    env.reset(seed=0)
    fixed = env.current_params()
    for _ in range(80):
        env.step(env.action_space.sample())
        assert env.current_params() == fixed, "physics moved WITHIN a task"
    env.close()


def test_each_target_is_non_inert():
    """Scaling pole_length or pole_mass alone changes the trajectory under identical actions."""
    def rollout(targets, mult):
        # The second multiplier only satisfies the constant-sequence check; task 0 is under test.
        e = DriftCartpoleSwingup(drift_targets=targets, max_episode_steps=200,
                                 task_multipliers=(mult, mult + 0.5))
        e.reset(seed=4)
        e.set_task(0)
        rng = np.random.RandomState(7)
        qs = []
        for _ in range(150):
            e.step(rng.uniform(-1, 1, size=(1,)))
            qs.append(np.array(e._dm_env.physics.data.qpos, copy=True))
        e.close()
        return np.array(qs)

    for target in ("pole_length", "pole_mass"):
        base = rollout((target,), 1.0)
        moved = rollout((target,), 1.6)
        assert np.abs(base - moved).max() > 1e-3, \
            f"{target} scaled but the simulation did not change - the target is inert"


def test_multiplier_one_is_an_exact_no_op():
    """A multiplier of 1.0 reproduces dm_control's nominal cartpole model exactly."""
    from dm_control import suite
    nominal = suite.load("cartpole", "swingup", task_kwargs={"random": 0})
    env = _env()
    env.set_task(0)
    m, n = env._dm_env.physics.model, nominal.physics.model
    assert np.array_equal(np.asarray(m.body_mass), np.asarray(n.body_mass))
    assert np.array_equal(np.asarray(m.geom_size), np.asarray(n.geom_size))
    assert np.allclose(np.asarray(m.body_inertia), np.asarray(n.body_inertia), rtol=0, atol=0)
    env.close()


def test_constant_multipliers_are_refused():
    """A multiplier sequence that never changes the physics raises ValueError."""
    with pytest.raises(ValueError, match="never change the physics"):
        DriftCartpoleSwingup(schedule="step", task_multipliers=(1.0, 1.0, 1.0))


def test_unknown_target_is_refused():
    """An unknown drift target raises ValueError."""
    with pytest.raises(ValueError, match="unknown drift target"):
        DriftCartpoleSwingup(drift_targets=("damping",))     # a HalfCheetah target


# A boundary changes the physics and nothing else.
def test_boundary_does_not_reset_the_state():
    """set_task keeps qpos and qvel across the model rebuild.

    `reload_from_xml_string` allocates fresh mjData, so without the snapshot the cart would return
    to the origin with zero velocity at every boundary.
    """
    env = _env()
    env.reset(seed=1)
    for _ in range(60):
        env.step(np.array([0.9]))
    before_q = np.array(env._dm_env.physics.data.qpos, copy=True)
    before_v = np.array(env._dm_env.physics.data.qvel, copy=True)
    assert np.abs(before_q).max() > 1e-3, "test is vacuous: the state never left the origin"

    env.set_task(1)

    assert np.allclose(np.array(env._dm_env.physics.data.qpos), before_q)
    assert np.allclose(np.array(env._dm_env.physics.data.qvel), before_v)
    env.close()


def test_model_survives_an_episode_reset():
    """An episode reset keeps the current task's pole parameters."""
    env = _env()
    env.set_task(1)
    before = env.current_params()
    env.reset(seed=9)
    assert env.current_params() == before
    env.close()


# A known return ceiling that is the same in every task.
def test_episode_is_exactly_1000_steps_and_never_terminates_early():
    """Episodes truncate at exactly 1000 steps and each reward is in [0, 1]."""
    env = DriftCartpoleSwingup(max_episode_steps=1000, task_multipliers=PHASE2_TASKS)
    env.reset(seed=0)
    n = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, r, terminated, truncated, _ = env.step(env.action_space.sample())
        n += 1
        assert 0.0 <= r <= 1.0, f"reward {r} is outside [0,1]; the ceiling claim is wrong"
        assert n <= 1000, "episode ran past its limit"
    assert n == 1000
    assert truncated and not terminated, "cartpole must truncate at the limit, never terminate"
    env.close()


def test_reward_does_not_depend_on_the_pole_we_change():
    """At a fixed state and control, the reward is the same for every pole multiplier.

    The cartpole reward reads only cart position, pole angle, angular velocity and control, so the
    ceiling is the same in every task.
    """
    state_q = np.array([0.13, 2.1])
    state_v = np.array([-0.4, 0.7])
    ctrl = np.array([0.3])

    rewards = []
    for mult in (0.6, 1.0, 1.6):
        env = DriftCartpoleSwingup(max_episode_steps=100,
                                   task_multipliers=(mult, mult + 0.5))
        env.set_task(0)
        ph = env._dm_env.physics
        ph.data.qpos[:] = state_q
        ph.data.qvel[:] = state_v
        ph.data.ctrl[:] = ctrl
        ph.forward()
        rewards.append(float(env._dm_env.task.get_reward(ph)))
        env.close()
    assert rewards[0] == pytest.approx(rewards[1]) == pytest.approx(rewards[2]), \
        f"the reward function changed with the pole: {rewards}"


# Behaviour the training loop depends on.
def test_seeding_is_reproducible():
    """reset(seed) is reproducible; the wrapper seeds the dm_control task since shimmy does not."""
    env = _env()
    a, _ = env.reset(seed=11)
    b, _ = env.reset(seed=11)
    c, _ = env.reset(seed=12)
    assert np.allclose(a, b), "same seed gave different initial states"
    assert not np.allclose(a, c), "different seeds gave the same initial state"
    env.close()


def test_set_task_propagates_through_the_vector_env():
    """env.unwrapped.call('set_task', i), as used in train.py, reaches every sub-env."""
    num_envs = 3
    envs = make_cartpole_vector_env(num_envs=num_envs, schedule="step",
                                   task_multipliers=PHASE2_TASKS, max_episode_steps=50,
                                   asynchronous=False)
    envs.reset(seed=0)
    envs.unwrapped.call("set_task", 1)
    params = envs.unwrapped.call("current_params")
    assert [p["drift_multiplier"] for p in params] == [pytest.approx(1.6)] * num_envs
    assert [p["drift_pole_length"] for p in params] == [pytest.approx(1.6)] * num_envs
    envs.close()


def test_observation_and_action_dimensions():
    """Observations have 5 dimensions and actions have 1."""
    env = _env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5,)
    assert env.action_space.shape == (1,)
    env.close()


def test_step_schedule_reports_a_zero_lipschitz_constant():
    """The step schedule reports a per-step Lipschitz constant of 0."""
    assert _env(schedule="step").lipschitz_constant() == 0.0
