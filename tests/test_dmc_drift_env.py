"""Tests for DmControlDrift, the drift wrapper for the six dm_control environments.

Each environment must change its physics at a boundary and nothing else, be non-inert, match stock
dm_control at multiplier 1.0, and run fixed-length episodes with a ceiling of 1000.
"""
import numpy as np
import pytest

pytest.importorskip("dm_control", reason="the dm_control family needs dm_control")
pytest.importorskip("shimmy", reason="the dm_control family needs shimmy")

from dm_control import suite                                          # noqa: E402
from src_continuous_control.envs.cartpole_swingup import (            # noqa: E402
    DriftCartpoleSwingup,
)
from src_continuous_control.envs.dm_control_drift import (            # noqa: E402
    ENV_NAMES,
    SPECS,
    DmControlDrift,
    make_dmc_vector_env,
)

PHASE2_TASKS = (1.0, 1.6, 0.6, 1.6, 0.6)

# Parametrise on ENV_NAMES so a newly added environment is covered automatically.
ALL_ENVS = list(ENV_NAMES)


def _env(name, **kw):
    kw.setdefault("max_episode_steps", 100)
    kw.setdefault("task_multipliers", PHASE2_TASKS)
    return DmControlDrift(env_name=name, **kw)


def _rollout(name, mult, n_steps=200, seed=7, action_seed=11):
    """Return the qpos trace of a fixed action sequence at multiplier `mult`.

    `999.0` is a second multiplier that is never selected; it only satisfies the wrapper's
    rejection of a constant `task_multipliers`.
    """
    env = DmControlDrift(env_name=name, task_multipliers=[mult, 999.0],
                         max_episode_steps=n_steps + 10)
    env.set_task(0)
    env.reset(seed=seed)
    rng = np.random.RandomState(action_seed)
    trace = []
    for _ in range(n_steps):
        env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
        trace.append(np.array(env._dm_env.physics.data.qpos, copy=True))
    env.close()
    return np.asarray(trace)


# Physics change at a boundary and stay constant within a task.
@pytest.mark.parametrize("name", ALL_ENVS)
def test_physics_differ_between_tasks(name):
    """Distinct multipliers give distinct physics, and revisited tasks give identical physics."""
    env = _env(name)
    seen = {}
    for i in range(len(PHASE2_TASKS)):
        env.set_task(i)
        params = env.current_params()
        seen[i] = tuple(v for k, v in sorted(params.items()) if k != "drift_task")
    # Tasks 1/3 and 2/4 revisit the same physics, which backward transfer relies on.
    assert seen[1] == seen[3]
    assert seen[2] == seen[4]
    assert len({seen[0], seen[1], seen[2]}) == 3
    env.close()


@pytest.mark.parametrize("name", ALL_ENVS)
def test_physics_hold_still_within_a_task(name):
    """With schedule='step' the physics do not change between boundaries."""
    env = _env(name)
    env.set_task(1)
    env.reset(seed=0)
    before = env.current_params()
    rng = np.random.RandomState(0)
    for _ in range(50):
        env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
    after = env.current_params()
    assert before == after
    env.close()


@pytest.mark.parametrize("name", ALL_ENVS)
def test_revisiting_a_multiplier_does_not_compound(name):
    """1.6 -> 0.6 -> 1.6 returns exactly to the physics of the first 1.6.

    Array targets scale from a stored nominal; scaling the current array would compound.
    """
    env = _env(name)
    env.set_task(1)
    first = np.array(env._dm_env.physics.model.body_mass, copy=True)
    first_damp = np.array(env._dm_env.physics.model.dof_damping, copy=True)
    first_fric = np.array(env._dm_env.physics.model.geom_friction, copy=True)
    env.set_task(2)
    env.set_task(3)                      # PHASE2_TASKS[3] == 1.6 == PHASE2_TASKS[1]
    assert np.array_equal(np.asarray(env._dm_env.physics.model.body_mass), first)
    assert np.array_equal(np.asarray(env._dm_env.physics.model.dof_damping), first_damp)
    assert np.array_equal(np.asarray(env._dm_env.physics.model.geom_friction), first_fric)
    env.close()


# A multiplier of 1.0 is an exact no-op.
@pytest.mark.parametrize("name", ALL_ENVS)
def test_unit_multiplier_is_an_exact_no_op(name):
    """At 1.0 the model is bit-identical to stock dm_control.

    Four of the six environments have no explicit `mass` in their MJCF, so the wrapper writes the
    compiler-derived mass back as an attribute; this round trip must be exact.
    """
    spec = SPECS[name]
    reference = suite.load(spec.domain, spec.task, task_kwargs={"random": 0})
    ref_mass = np.array(reference.physics.model.body_mass, copy=True)
    ref_inertia = np.array(reference.physics.model.body_inertia, copy=True)
    ref_size = np.array(reference.physics.model.geom_size, copy=True)
    reference.close()

    env = _env(name)
    env.set_task(0)                      # PHASE2_TASKS[0] == 1.0
    model = env._dm_env.physics.model
    assert np.array_equal(np.asarray(model.body_mass), ref_mass)
    assert np.array_equal(np.asarray(model.body_inertia), ref_inertia)
    assert np.array_equal(np.asarray(model.geom_size), ref_size)
    env.close()


# Non-inert physics changes.
@pytest.mark.parametrize("name", ALL_ENVS)
def test_each_environment_is_non_inert(name):
    """Same seed and actions with scaled physics give a different trajectory.

    Guards against multiplicatively inert targets (e.g. armature at 0, or friction on a body that
    never touches the ground).
    """
    base = _rollout(name, 1.0)
    for mult in (1.6, 0.6):
        scaled = _rollout(name, mult)
        assert np.abs(scaled - base).max() > 1e-3, (
            f"{name} at x{mult} is INERT: the chosen physics parameter does not move the "
            f"trajectory, so this environment would measure nothing")


# A boundary changes the physics and nothing else.
@pytest.mark.parametrize("name", ALL_ENVS)
def test_boundary_does_not_reset_the_state(name):
    """set_task keeps qpos and qvel across the model rebuild, which allocates fresh mjData."""
    env = _env(name, max_episode_steps=200)
    env.reset(seed=3)
    rng = np.random.RandomState(0)
    for _ in range(40):
        env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
    physics = env._dm_env.physics
    qpos = np.array(physics.data.qpos, copy=True)
    qvel = np.array(physics.data.qvel, copy=True)

    env.set_task(1)

    assert np.array_equal(np.asarray(env._dm_env.physics.data.qpos), qpos)
    assert np.array_equal(np.asarray(env._dm_env.physics.data.qvel), qvel)
    env.close()


def test_reacher_target_survives_a_boundary():
    """reacher's target position and radius, stored in the model, survive a boundary.

    `Reacher.initialize_episode` writes the target radius into `geom_size['target']` and a random
    target position into `geom_pos['target']`.
    """
    env = _env("reacher-easy", max_episode_steps=200)
    env.reset(seed=3)
    named = env._dm_env.physics.named.model
    pos = np.array(named.geom_pos["target"], copy=True)
    size = np.array(named.geom_size["target"], copy=True)
    # The drawn target must differ from the XML default for the test to mean anything.
    assert np.abs(pos[:2]).max() > 1e-6

    env.set_task(1)

    named = env._dm_env.physics.named.model
    assert np.array_equal(np.asarray(named.geom_pos["target"]), pos)
    assert np.array_equal(np.asarray(named.geom_size["target"]), size)
    env.close()


def test_reacher_target_guard_is_load_bearing():
    """With `preserve` emptied the reacher target moves at a boundary, so the guard is needed."""
    env = _env("reacher-easy", max_episode_steps=200)
    env.reset(seed=3)
    before = np.array(env._dm_env.physics.named.model.geom_pos["target"], copy=True)
    env.spec_.preserve = ()              # disable the guard
    try:
        env.set_task(1)
        after = np.array(env._dm_env.physics.named.model.geom_pos["target"], copy=True)
    finally:
        env.spec_.preserve = (("geom_size", "target"), ("geom_pos", "target"))
        env.close()
    assert np.abs(after - before).max() > 1e-3, (
        "emptying `preserve` did not move reacher's target, so the guard protects nothing")


# Consistency with the standalone cartpole wrapper.
def test_cartpole_matches_the_anchor_wrapper():
    """This wrapper's cartpole has the same model as envs/cartpole_swingup.py in every task."""
    new = DmControlDrift(env_name="cartpole-swingup", task_multipliers=PHASE2_TASKS,
                         max_episode_steps=200)
    old = DriftCartpoleSwingup(task_multipliers=PHASE2_TASKS, max_episode_steps=200)
    for i in range(len(PHASE2_TASKS)):
        new.set_task(i)
        old.set_task(i)
        for field in ("body_mass", "body_inertia", "geom_size", "dof_damping", "geom_friction"):
            assert np.array_equal(np.asarray(getattr(new._dm_env.physics.model, field)),
                                  np.asarray(getattr(old._dm_env.physics.model, field))), \
                f"cartpole {field} disagrees with the anchor wrapper at task {i}"
    new.close()
    old.close()


# Fixed episode length and a return ceiling of 1000.
@pytest.mark.parametrize("name", ALL_ENVS)
def test_episode_is_exactly_max_episode_steps(name):
    """Every environment truncates at exactly max_episode_steps.

    The control timestep differs per domain (0.01 / 0.02 / 0.025 s), so the time limit is derived
    from it per environment.
    """
    n = 60
    env = _env(name, max_episode_steps=n)
    env.reset(seed=0)
    rng = np.random.RandomState(0)
    steps, ended = 0, False
    for _ in range(n + 5):
        _, _, terminated, truncated, _ = env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
        steps += 1
        if terminated or truncated:
            ended = True
            break
    assert ended and steps == n, f"{name} ended after {steps} steps, not {n}"
    env.close()


@pytest.mark.parametrize("name", ALL_ENVS)
def test_reward_is_bounded_in_unit_interval(name):
    """Per-step rewards are in [0, 1], so the 1000-step ceiling is 1000."""
    env = _env(name)
    env.reset(seed=0)
    rng = np.random.RandomState(0)
    for _ in range(100):
        _, reward, _, _, _ = env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
        assert 0.0 <= reward <= 1.0
    env.close()


@pytest.mark.parametrize("name", ALL_ENVS)
def test_observation_and_action_dimensions_match_the_spec(name):
    """Observation and action shapes match the spec, which the parameter-parity widths use."""
    spec = SPECS[name]
    env = _env(name)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (spec.obs_dim,)
    assert env.observation_space.shape == (spec.obs_dim,)
    assert env.action_space.shape == (spec.act_dim,)
    env.close()


# The walker pair.
def test_walker_pair_is_identical_apart_from_the_task():
    """walker-stand and walker-walk differ only in the task.

    Body, observations, drift targets and the realised physics per task must all match, so the
    pair separates task type from body size.
    """
    stand, walk = SPECS["walker-stand"], SPECS["walker-walk"]
    assert stand.domain == walk.domain == "walker"
    assert stand.task != walk.task
    assert (stand.obs_dim, stand.act_dim) == (walk.obs_dim, walk.act_dim)
    assert stand.default_targets == walk.default_targets
    assert stand.control_timestep == walk.control_timestep
    assert sorted(stand.targets) == sorted(walk.targets)

    # Check the realised physics, not only the declared spec.
    a = _env("walker-stand")
    b = _env("walker-walk")
    for i in range(len(PHASE2_TASKS)):
        a.set_task(i)
        b.set_task(i)
        assert np.array_equal(np.asarray(a._dm_env.physics.model.body_mass),
                              np.asarray(b._dm_env.physics.model.body_mass))
        assert np.array_equal(np.asarray(a._dm_env.physics.model.geom_friction),
                              np.asarray(b._dm_env.physics.model.geom_friction))
    a.close()
    b.close()


# Schedule bookkeeping.
@pytest.mark.parametrize("name", ALL_ENVS)
def test_step_schedule_cycles_and_has_zero_lipschitz_constant(name):
    """The step schedule cycles through the multipliers and reports a Lipschitz constant of 0."""
    env = _env(name)
    for i, expected in enumerate(PHASE2_TASKS):
        env.set_task(i)
        assert env.multiplier() == expected
    env.set_task(len(PHASE2_TASKS))      # wraps back to task 0
    assert env.multiplier() == PHASE2_TASKS[0]
    # The bound covers within-task steps only; the boundary jump is excluded.
    assert env.lipschitz_constant() == 0.0
    env.close()


def test_continuous_schedule_has_a_positive_lipschitz_constant():
    """The sin schedule reports a positive Lipschitz constant."""
    env = DmControlDrift(env_name="cheetah-run", schedule="sin", amplitude=0.5, period=1000,
                         max_episode_steps=100)
    assert env.lipschitz_constant() > 0.0
    env.close()


@pytest.mark.parametrize("name", ALL_ENVS)
def test_seeding_is_reproducible(name):
    """The same seed gives the same rollout; the wrapper seeds the dm_control task itself."""
    traces = []
    for _ in range(2):
        env = _env(name, max_episode_steps=200)
        env.reset(seed=12345)
        rng = np.random.RandomState(4)
        rollout = []
        for _ in range(30):
            obs, _, _, _, _ = env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
            rollout.append(np.array(obs, copy=True))
        env.close()
        traces.append(np.asarray(rollout))
    assert np.array_equal(traces[0], traces[1])


def test_different_seeds_give_different_episodes():
    """Different seeds give different reacher targets."""
    out = []
    for seed in (0, 1):
        env = _env("reacher-easy", max_episode_steps=200)
        env.reset(seed=seed)
        out.append(np.array(env._dm_env.physics.named.model.geom_pos["target"], copy=True))
        env.close()
    assert not np.array_equal(out[0], out[1])


# Invalid configurations raise ValueError.
def test_unknown_environment_is_refused():
    """An unknown env_name raises ValueError."""
    with pytest.raises(ValueError, match="unknown env_name"):
        DmControlDrift(env_name="humanoid-run")


def test_unknown_drift_target_is_refused():
    """A drift target from another environment raises ValueError."""
    with pytest.raises(ValueError, match="unknown drift target"):
        DmControlDrift(env_name="cheetah-run", drift_targets=["pole_length"])


def test_empty_drift_targets_is_refused():
    """An empty drift_targets list raises ValueError."""
    with pytest.raises(ValueError, match="never change"):
        DmControlDrift(env_name="cheetah-run", drift_targets=[])


def test_constant_multipliers_are_refused():
    """A multiplier sequence that never changes the physics raises ValueError."""
    with pytest.raises(ValueError, match="would never change the physics"):
        DmControlDrift(env_name="cheetah-run", task_multipliers=[1.0, 1.0, 1.0])


def test_unknown_schedule_is_refused():
    """An unknown schedule raises ValueError."""
    with pytest.raises(ValueError, match="unknown schedule"):
        DmControlDrift(env_name="cheetah-run", schedule="sawtooth")


# The vector-env factory used by training.
@pytest.mark.parametrize("name", ["cartpole-swingup", "walker-walk"])
def test_vector_env_builds_and_switches_tasks(name):
    """The synchronous vector env builds, steps, and reports the drift multiplier in info."""
    envs = make_dmc_vector_env(env_name=name, num_envs=2, task_multipliers=PHASE2_TASKS,
                               max_episode_steps=100, asynchronous=False)
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (2, SPECS[name].obs_dim)
    actions = np.zeros((2,) + envs.single_action_space.shape, dtype=np.float32)
    _, _, _, _, infos = envs.step(actions)
    assert "drift_multiplier" in infos
    envs.close()
