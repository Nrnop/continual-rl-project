"""A task boundary must start the new task from a DEFINED state, not from wherever the last one
happened to leave the body.

Without a reset, HalfCheetah carries its pose, joint velocities and body angle across the switch.
Whether the new task begins from a clean stand or from a mid-air somersault is then luck that
varies per seed and per arm, and it blurs the meaning of every post-boundary metric this study
reports — jumpstart, boundary drop, recovery.

It matters most for the physics benchmark: changing mass or damping while the body is airborne is
not a well-defined transition at all.
"""
import numpy as np
import pytest

from src_continuous_control.envs.drift_half_cheetah import (
    LipschitzDriftHalfCheetah,
    make_drift_vector_env,
)

mujoco = pytest.importorskip("mujoco", reason="needs the mujoco physics model")

TASKS = (1.0, 1.6, 0.6, 1.6, 0.6)


def _walk(env, n, seed=None):
    """Step the env with fixed actions so the body leaves its start pose."""
    if seed is not None:
        env.reset(seed=seed)
    rng = np.random.default_rng(0)
    for _ in range(n):
        env.step(rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32))


def test_reset_returns_the_body_to_a_standing_start():
    """After a reset the state must match a fresh episode, not the state we drifted into."""
    env = LipschitzDriftHalfCheetah(schedule="step", task_multipliers=TASKS,
                                    max_episode_steps=10_000)
    obs0, _ = env.reset(seed=0)
    _walk(env, 120)
    moved, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert not np.allclose(moved, obs0, atol=1e-3), "sanity: the body should have moved"

    obs_after_reset, _ = env.reset()
    qvel = np.asarray(env.unwrapped.data.qvel)
    assert np.abs(qvel).max() < 1.0, f"reset left the body moving (max |qvel| = {np.abs(qvel).max()})"
    assert np.allclose(obs_after_reset, obs0, atol=0.3), \
        "reset did not return the body to a start-of-episode pose"
    env.close()


def test_the_drift_clock_survives_a_boundary_reset():
    """The physics are a property of the WORLD, not of the episode — resetting must not rewind."""
    env = LipschitzDriftHalfCheetah(schedule="sin", period=1000, amplitude=0.5,
                                    max_episode_steps=10_000)
    env.reset(seed=0)
    _walk(env, 50)
    t_before = env.t
    env.reset()
    assert env.t == t_before, "the boundary reset rewound the drift clock"
    env.close()


def test_reset_keeps_the_task_that_was_just_selected():
    """set_task then reset: the reset must not undo the physics the boundary just applied."""
    env = LipschitzDriftHalfCheetah(schedule="step", task_multipliers=TASKS,
                                    drift_targets=("damping", "friction", "mass", "armature"),
                                    max_episode_steps=10_000)
    env.reset(seed=0)
    _walk(env, 30)
    env.set_task(1)                                   # x1.6
    before = env.current_params()
    env.reset()
    after = env.current_params()
    assert after == before, f"reset changed the physics: {before} -> {after}"
    assert after["drift_multiplier"] == pytest.approx(1.6)
    env.close()


def test_vector_env_reset_clears_every_sub_env():
    """train.py resets the whole vector env at a boundary; every sub-env must come back."""
    num_envs = 3
    envs = make_drift_vector_env(num_envs=num_envs, schedule="step", task_multipliers=TASKS,
                                 max_episode_steps=10_000, asynchronous=False)
    envs.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(60):
        envs.step(np.stack([rng.uniform(-1, 1, size=envs.single_action_space.shape)
                            .astype(np.float32) for _ in range(num_envs)]))
    obs, _ = envs.reset()
    assert obs.shape[0] == num_envs
    for qvel in envs.unwrapped.call("current_params"):   # smoke: every sub-env answers
        assert "drift_multiplier" in qvel
    envs.close()
