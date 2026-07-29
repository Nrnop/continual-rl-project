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
