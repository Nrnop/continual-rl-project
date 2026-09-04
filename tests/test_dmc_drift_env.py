"""DmControlDrift: six environments, one wrapper, and none of them may lie about their physics.

The family study exists because every conclusion in this project rests on two environments that
differ in size AND task type at once. Six points only buy anything if each one is a benchmark that
actually does what its config says, so the tests below pin the properties the study depends on —
and three of them correspond directly to failures this project has already paid for:

  * a multiplicatively-inert parameter that looked exactly like a working experiment (a week lost
    in Phase 1)                        -> `test_each_environment_is_non_inert`
  * a manipulation that silently did MORE than it said
                                       -> `test_boundary_does_not_reset_the_state`,
                                          `test_reacher_target_survives_a_boundary`
  * a control that was not actually off / a guard that was not actually load-bearing
                                       -> `test_reacher_target_guard_is_load_bearing`

`test_cartpole_matches_the_anchor_wrapper` is the one that protects the existing data: 95 committed
runs were produced by `envs/cartpole_swingup.py`, and if this wrapper's cartpole disagreed with it
the anchor of the whole family would be a different environment from the one already measured.
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

# Every environment in the family must be exercised by every structural test. Parametrising on
# ENV_NAMES rather than a hand-written list means adding a seventh environment cannot silently
# skip its own gates.
ALL_ENVS = list(ENV_NAMES)


def _env(name, **kw):
    kw.setdefault("max_episode_steps", 100)
    kw.setdefault("task_multipliers", PHASE2_TASKS)
    return DmControlDrift(env_name=name, **kw)


def _rollout(name, mult, n_steps=200, seed=7, action_seed=11):
    """Drive one env with a FIXED action sequence at a fixed multiplier; return its qpos trace.

    The actions do not depend on the physics, so any divergence between two multipliers is caused
    by the physics and nothing else. `999.0` is a second multiplier that is never selected — the
    wrapper refuses a constant `task_multipliers`, and a value that is obviously wrong is safer
    here than a plausible one that might quietly be used.
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


# ---------------------------------------------------------------------------
# The physics change at a boundary, and hold still within a task
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_ENVS)
def test_physics_differ_between_tasks(name):
    """Required by MULTIENV_TASK.md §9. Distinct multipliers must give distinct physics."""
    env = _env(name)
    seen = {}
    for i in range(len(PHASE2_TASKS)):
        env.set_task(i)
        params = env.current_params()
        seen[i] = tuple(v for k, v in sorted(params.items()) if k != "drift_task")
    # Tasks 1/3 and 2/4 are REVISITS of the same physics — that is what makes backward transfer
    # well-defined, so they must match exactly rather than merely be close.
    assert seen[1] == seen[3]
    assert seen[2] == seen[4]
    # ...and the three distinct tasks must genuinely differ.
    assert len({seen[0], seen[1], seen[2]}) == 3
    env.close()


@pytest.mark.parametrize("name", ALL_ENVS)
def test_physics_hold_still_within_a_task(name):
    """schedule='step' means the physics are a function of the TASK INDEX, not of the clock."""
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
    """1.6 -> 0.6 -> 1.6 must land on exactly the physics of the first 1.6.

    Array-based targets scale from a stored nominal for precisely this reason: scaling an
    already-scaled array compounds, and the sequence above would silently end at 2.56x.
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


# ---------------------------------------------------------------------------
# A multiplier of 1.0 must be an EXACT no-op
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_ENVS)
def test_unit_multiplier_is_an_exact_no_op(name):
    """At 1.0 the model must be bit-identical to stock dm_control.

    This is not pedantry. Four of the six environments have no explicit `mass` in their MJCF, so
    the wrapper writes the COMPILER-DERIVED mass back as an attribute in order to keep length and
    mass independent. If that round-trip were merely close rather than exact, "the same physics"
    would not be the same physics across two configs, and task 0 of the family would differ from
    the stock benchmark everyone else publishes against.
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


# ---------------------------------------------------------------------------
# Non-inert: the failure that cost Phase 1 a week
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_ENVS)
def test_each_environment_is_non_inert(name):
    """Same seed, same actions, scaled physics -> the trajectories must genuinely diverge.

    A parameter that is multiplicatively inert (armature at exactly 0, damping at 0.0005, friction
    on a body that never touches the ground) produces an experiment that runs, logs a multiplier,
    and measures nothing at all.
    """
    base = _rollout(name, 1.0)
    for mult in (1.6, 0.6):
        scaled = _rollout(name, mult)
        assert np.abs(scaled - base).max() > 1e-3, (
            f"{name} at x{mult} is INERT: the chosen physics parameter does not move the "
            f"trajectory, so this environment would measure nothing")


# ---------------------------------------------------------------------------
# A boundary changes the physics and NOTHING else
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_ENVS)
def test_boundary_does_not_reset_the_state(name):
    """A rebuild allocates fresh mjData; without the snapshot that is a free episode reset."""
    env = _env(name, max_episode_steps=200)
    env.reset(seed=3)
    rng = np.random.RandomState(0)
    for _ in range(40):                  # walk away from the initial pose
        env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
    physics = env._dm_env.physics
    qpos = np.array(physics.data.qpos, copy=True)
    qvel = np.array(physics.data.qvel, copy=True)

    env.set_task(1)

    assert np.array_equal(np.asarray(env._dm_env.physics.data.qpos), qpos)
    assert np.array_equal(np.asarray(env._dm_env.physics.data.qvel), qvel)
    env.close()


def test_reacher_target_survives_a_boundary():
    """reacher stores its GOAL in the model, so a rebuild would move it mid-episode.

    `Reacher.initialize_episode` writes the target radius into `geom_size['target']` and draws a
    random target position into `geom_pos['target']`. A boundary that reset them would change the
    physics and the task at the same instant — and would look exactly like a working experiment.
    """
    env = _env("reacher-easy", max_episode_steps=200)
    env.reset(seed=3)
    named = env._dm_env.physics.named.model
    pos = np.array(named.geom_pos["target"], copy=True)
    size = np.array(named.geom_size["target"], copy=True)
    # The randomly-drawn target must not be at the XML default, or this test proves nothing.
    assert np.abs(pos[:2]).max() > 1e-6

    env.set_task(1)

    named = env._dm_env.physics.named.model
    assert np.array_equal(np.asarray(named.geom_pos["target"]), pos)
    assert np.array_equal(np.asarray(named.geom_size["target"]), size)
    env.close()


def test_reacher_target_guard_is_load_bearing():
    """The guard above must actually be doing something.

    Failure mode #2 in CLAUDE.md is a manipulation that silently never fires; the mirror image is
    a GUARD that silently never fires, which is just as invisible and just as damaging. With
    `preserve` emptied, the target must move — if it does not, the test above is passing for the
    wrong reason and would keep passing if the guard were deleted.
    """
    env = _env("reacher-easy", max_episode_steps=200)
    env.reset(seed=3)
    before = np.array(env._dm_env.physics.named.model.geom_pos["target"], copy=True)
    env.spec_.preserve = ()              # simulate the guard not existing
    try:
        env.set_task(1)
        after = np.array(env._dm_env.physics.named.model.geom_pos["target"], copy=True)
    finally:
        env.spec_.preserve = (("geom_size", "target"), ("geom_pos", "target"))
        env.close()
    assert np.abs(after - before).max() > 1e-3, (
        "emptying `preserve` did not move reacher's target, so the guard protects nothing")


# ---------------------------------------------------------------------------
# The cartpole anchor: 95 committed runs depend on the original wrapper
# ---------------------------------------------------------------------------
def test_cartpole_matches_the_anchor_wrapper():
    """This wrapper's cartpole must be the SAME environment as envs/cartpole_swingup.py.

    cartpole-swingup is the family's anchor precisely because it already has 95 runs behind it. If
    the two wrappers disagreed by so much as a gram, the anchor point on the carry-over plot would
    come from a different benchmark than the one it is being compared against.
    """
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


# ---------------------------------------------------------------------------
# The property the whole family was chosen FOR: a ceiling of exactly 1000
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_ENVS)
def test_episode_is_exactly_max_episode_steps(name):
    """Returns average across the family only because every task runs for the same 1000 steps.

    The control timestep differs per domain (0.01 / 0.02 / 0.025 s), so the time limit is derived
    from it. Hard-coding cartpole's "/100" would have given walker 2500-step episodes and a
    ceiling of 2500 rather than 1000 — with nothing in the logs to say so.
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
    """dm_control rewards are in [0,1] per step, which is what makes the ceiling 1000."""
    env = _env(name)
    env.reset(seed=0)
    rng = np.random.RandomState(0)
    for _ in range(100):
        _, reward, _, _, _ = env.step(rng.uniform(-1.0, 1.0, env.action_space.shape))
        assert 0.0 <= reward <= 1.0
    env.close()


@pytest.mark.parametrize("name", ALL_ENVS)
def test_observation_and_action_dimensions_match_the_spec(name):
    """The spec's dimensions drive the per-environment parameter-parity derivation.

    If a spec claimed 24 observations and the env produced 25, `pt`'s widths would be derived for
    the wrong network and the capacity handicap would be invisible — which is exactly what
    happened on cartpole at 0.931x.
    """
    spec = SPECS[name]
    env = _env(name)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (spec.obs_dim,)
    assert env.observation_space.shape == (spec.obs_dim,)
    assert env.action_space.shape == (spec.act_dim,)
    env.close()


# ---------------------------------------------------------------------------
# The walker pair is the study's one controlled comparison
# ---------------------------------------------------------------------------
def test_walker_pair_is_identical_apart_from_the_task():
    """walker-stand and walker-walk must differ in the GOAL and in nothing else.

    The pair is the only comparison in the family that separates body size from task type, and it
    does that only if the body, the observations, the physics change and the multipliers are all
    identical. Anything else that differed would reopen exactly the confound the pair exists to
    close.
    """
    stand, walk = SPECS["walker-stand"], SPECS["walker-walk"]
    assert stand.domain == walk.domain == "walker"
    assert stand.task != walk.task
    assert (stand.obs_dim, stand.act_dim) == (walk.obs_dim, walk.act_dim)
    assert stand.default_targets == walk.default_targets
    assert stand.control_timestep == walk.control_timestep
    assert sorted(stand.targets) == sorted(walk.targets)

    # ...and the realised physics must match task for task, not merely the declared config.
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


# ---------------------------------------------------------------------------
# Schedule bookkeeping, shared with the other two benchmarks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_ENVS)
def test_step_schedule_cycles_and_has_zero_lipschitz_constant(name):
    env = _env(name)
    for i, expected in enumerate(PHASE2_TASKS):
        env.set_task(i)
        assert env.multiplier() == expected
    env.set_task(len(PHASE2_TASKS))      # wraps back to task 0
    assert env.multiplier() == PHASE2_TASKS[0]
    # Under "step" the physics are constant within a task; the boundary discontinuity is
    # deliberate and is not covered by the bound.
    assert env.lipschitz_constant() == 0.0
    env.close()


def test_continuous_schedule_has_a_positive_lipschitz_constant():
    env = DmControlDrift(env_name="cheetah-run", schedule="sin", amplitude=0.5, period=1000,
                         max_episode_steps=100)
    assert env.lipschitz_constant() > 0.0
    env.close()


@pytest.mark.parametrize("name", ALL_ENVS)
def test_seeding_is_reproducible(name):
    """shimmy does not seed the dm_control task, so the wrapper must — or --seed means nothing."""
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
    """The counterpart to the test above: seeding must not have collapsed to a constant."""
    out = []
    for seed in (0, 1):
        env = _env("reacher-easy", max_episode_steps=200)
        env.reset(seed=seed)
        out.append(np.array(env._dm_env.physics.named.model.geom_pos["target"], copy=True))
        env.close()
    assert not np.array_equal(out[0], out[1])


# ---------------------------------------------------------------------------
# Refusals: a misconfigured benchmark must not run quietly
# ---------------------------------------------------------------------------
def test_unknown_environment_is_refused():
    with pytest.raises(ValueError, match="unknown env_name"):
        DmControlDrift(env_name="humanoid-run")


def test_unknown_drift_target_is_refused():
    with pytest.raises(ValueError, match="unknown drift target"):
        DmControlDrift(env_name="cheetah-run", drift_targets=["pole_length"])


def test_empty_drift_targets_is_refused():
    with pytest.raises(ValueError, match="never change"):
        DmControlDrift(env_name="cheetah-run", drift_targets=[])


def test_constant_multipliers_are_refused():
    """A sequence that never changes the physics looks exactly like a working experiment."""
    with pytest.raises(ValueError, match="would never change the physics"):
        DmControlDrift(env_name="cheetah-run", task_multipliers=[1.0, 1.0, 1.0])


def test_unknown_schedule_is_refused():
    with pytest.raises(ValueError, match="unknown schedule"):
        DmControlDrift(env_name="cheetah-run", schedule="sawtooth")


# ---------------------------------------------------------------------------
# The vectorised factory, which is what training actually uses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["cartpole-swingup", "walker-walk"])
def test_vector_env_builds_and_switches_tasks(name):
    """Sync only: the point is the wrapper order and set_task plumbing, not parallelism."""
    envs = make_dmc_vector_env(env_name=name, num_envs=2, task_multipliers=PHASE2_TASKS,
                               max_episode_steps=100, asynchronous=False)
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (2, SPECS[name].obs_dim)
    actions = np.zeros((2,) + envs.single_action_space.shape, dtype=np.float32)
    _, _, _, _, infos = envs.step(actions)
    assert "drift_multiplier" in infos
    envs.close()
