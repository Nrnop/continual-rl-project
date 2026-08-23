"""DriftCartpoleSwingup: the physics must really change per task, and nothing else may.

The env exists to answer whether the HalfCheetah findings about `pt` are a property of the METHOD
or of HalfCheetah, so its own correctness has to be pinned rather than assumed. Two of these tests
correspond directly to failures this project has already paid for:

  * a multiplicatively-inert parameter that looked exactly like a working experiment (a week lost
    in Phase 1) -> `test_each_target_is_non_inert`, `test_physics_differ_between_tasks`
  * a manipulation that silently did more than it said -> `test_boundary_does_not_reset_the_state`

and the rest pin the property this benchmark was chosen FOR: a return ceiling of exactly 1000 that
is identical across every task in the sequence.
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


# ---------------------------------------------------------------------------
# The physics change at a boundary, and hold still within a task
# ---------------------------------------------------------------------------
def test_physics_differ_between_tasks():
    """Required by CARTPOLE_TASK.md §5. Every distinct multiplier must give distinct physics."""
    env = _env()
    seen = {}
    for i in range(len(PHASE2_TASKS)):
        env.set_task(i)
        p = env.current_params()
        seen[i] = (p["drift_pole_length"], p["drift_pole_mass"], p["drift_pole_inertia"])
    # tasks 1/3 and 2/4 are REVISITS of the same physics — that is what makes backward transfer
    # well-defined, so they must match exactly rather than merely be close
    assert seen[1] == seen[3]
    assert seen[2] == seen[4]
    # ...and the three distinct tasks must genuinely differ
    assert len({seen[0], seen[1], seen[2]}) == 3
    env.close()


def test_physics_are_constant_within_a_task():
    """The other half of §5: schedule='step' must not move the physics between boundaries."""
    env = _env()
    env.set_task(1)
    env.reset(seed=0)
    fixed = env.current_params()
    for _ in range(80):
        env.step(env.action_space.sample())
        assert env.current_params() == fixed, "physics moved WITHIN a task"
    env.close()


def test_each_target_is_non_inert():
    """Each drift target, ON ITS OWN, must change the simulation.

    A parameter that scales but does not alter the trajectory looks exactly like a working
    experiment. Driving identical actions from an identical state, the resulting states must
    diverge — for pole_length and pole_mass separately, not merely for the two together.
    """
    def rollout(targets, mult):
        # The second entry only exists to satisfy the constant-sequence guard; task 0 is the one
        # under test, so it must be the multiplier we want applied.
        e = DriftCartpoleSwingup(drift_targets=targets, max_episode_steps=200,
                                 task_multipliers=(mult, mult + 0.5))
        e.reset(seed=4)
        e.set_task(0)                       # apply `mult`
        rng = np.random.RandomState(7)
        qs = []
        for _ in range(150):
            e.step(rng.uniform(-1, 1, size=(1,)))
            qs.append(np.array(e._dm_env.physics.data.qpos, copy=True))
        e.close()
        return np.array(qs)

    for target in ("pole_length", "pole_mass"):
        base = rollout((target,), 1.0)      # multiplier 1.0 -> nominal
        moved = rollout((target,), 1.6)
        assert np.abs(base - moved).max() > 1e-3, \
            f"{target} scaled but the simulation did not change — the target is inert"


def test_multiplier_one_is_an_exact_no_op():
    """x1.0 must reproduce dm_control's nominal model bit for bit.

    Otherwise 'the same physics' is not the same physics across two configs, and task 0 of this
    benchmark would silently not be stock cartpole-swingup.
    """
    from dm_control import suite
    nominal = suite.load("cartpole", "swingup", task_kwargs={"random": 0})
    env = _env()
    env.set_task(0)                                         # multiplier 1.0
    m, n = env._dm_env.physics.model, nominal.physics.model
    assert np.array_equal(np.asarray(m.body_mass), np.asarray(n.body_mass))
    assert np.array_equal(np.asarray(m.geom_size), np.asarray(n.geom_size))
    assert np.allclose(np.asarray(m.body_inertia), np.asarray(n.body_inertia), rtol=0, atol=0)
    env.close()


def test_constant_multipliers_are_refused():
    """A sequence that never changes the physics is a silently-broken experiment, not a config."""
    with pytest.raises(ValueError, match="never change the physics"):
        DriftCartpoleSwingup(schedule="step", task_multipliers=(1.0, 1.0, 1.0))


def test_unknown_target_is_refused():
    with pytest.raises(ValueError, match="unknown drift target"):
        DriftCartpoleSwingup(drift_targets=("damping",))     # a HalfCheetah target


# ---------------------------------------------------------------------------
# A boundary changes the physics and NOTHING else
# ---------------------------------------------------------------------------
def test_boundary_does_not_reset_the_state():
    """Rebuilding the model must not hand every arm a free episode reset at every boundary.

    `reload_from_xml_string` allocates fresh mjData, which would teleport the cart to the origin
    and zero the pole's velocity. That would be a hidden manipulation no config key asks for, and
    it would flatter whichever method benefits most from restarting.
    """
    env = _env()
    env.reset(seed=1)
    for _ in range(60):                     # move away from the initial state
        env.step(np.array([0.9]))
    before_q = np.array(env._dm_env.physics.data.qpos, copy=True)
    before_v = np.array(env._dm_env.physics.data.qvel, copy=True)
    assert np.abs(before_q).max() > 1e-3, "test is vacuous: the state never left the origin"

    env.set_task(1)                         # a real physics change

    assert np.allclose(np.array(env._dm_env.physics.data.qpos), before_q)
    assert np.allclose(np.array(env._dm_env.physics.data.qvel), before_v)
    env.close()


def test_model_survives_an_episode_reset():
    """dm_control's reset restores mjData, not the compiled model — the task's pole must persist."""
    env = _env()
    env.set_task(1)
    before = env.current_params()
    env.reset(seed=9)
    assert env.current_params() == before
    env.close()


# ---------------------------------------------------------------------------
# The reason this benchmark was chosen: a known, task-invariant ceiling
# ---------------------------------------------------------------------------
def test_episode_is_exactly_1000_steps_and_never_terminates_early():
    """The return ceiling is `max_episode_steps` only if the episode always runs to the limit."""
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
    """Every task must have the SAME ceiling, or returns are not comparable across the sequence.

    Unlike HalfCheetah — where scaling damping changes the achievable velocity and therefore the
    achievable return — the cartpole reward reads only cart position, pole angle, angular velocity
    and control. Evaluated at an IDENTICAL state with an identical control, it must be identical
    across poles. (This holds the state fixed on purpose: the dynamics of course differ, which is
    the manipulation. What must not differ is the scoring function.)
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


# ---------------------------------------------------------------------------
# Plumbing the training loop depends on
# ---------------------------------------------------------------------------
def test_seeding_is_reproducible():
    """shimmy does not seed the dm_control task, so the wrapper must — or runs are not seed-fixed."""
    env = _env()
    a, _ = env.reset(seed=11)
    b, _ = env.reset(seed=11)
    c, _ = env.reset(seed=12)
    assert np.allclose(a, b), "same seed gave different initial states"
    assert not np.allclose(a, c), "different seeds gave the same initial state"
    env.close()


def test_set_task_propagates_through_the_vector_env():
    """train.py switches tasks with env.unwrapped.call('set_task', i) — every sub-env must move."""
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
    """obs 5 / act 1 — the dimensions parameter parity was recomputed against."""
    env = _env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5,)
    assert env.action_space.shape == (1,)
    env.close()


def test_step_schedule_reports_a_zero_lipschitz_constant():
    """Physics are constant within a task, so the per-step bound is 0; the jump is at the boundary."""
    assert _env(schedule="step").lipschitz_constant() == 0.0
