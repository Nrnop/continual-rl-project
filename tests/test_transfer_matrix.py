"""Forward/backward transfer: R[i, j], BWT and FWT (Lopez-Paz & Ranzato 2017).

Phase 2's second metric. The definitions are simple enough that the risk is not the arithmetic but
the wiring: evaluating on the wrong task, on a policy that is still exploring, or on the training
env (whose physics would then be left on the wrong task). These pin the contract.
"""
import numpy as np
import pytest

from src_continuous_control.utils.metrics import (
    TransferMatrix,
    evaluate_policy_on_tasks,
    evaluate_transfer_matrix,
)


class FakeTaskEnv:
    """Two-task env whose return is a known function of (task, policy) — no physics involved.

    Each episode is `episode_len` steps of reward `payoff[task][policy_id]`, so a cell of the
    transfer matrix has an exactly predictable value and any indexing error is visible.
    """

    def __init__(self, payoff, episode_len=3):
        self.payoff = payoff
        self.episode_len = episode_len
        self.task = 0
        self.resets = 0
        self._t = 0

    def set_task(self, i):
        self.task = int(i)

    def reset(self):
        self.resets += 1
        self._t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        reward = float(self.payoff[self.task][int(action[0])])
        terminated = False
        truncated = self._t >= self.episode_len
        return np.zeros(1, dtype=np.float32), reward, terminated, truncated, {}


def _policy(policy_id):
    return lambda obs: np.array([policy_id])


# payoff[task][policy]: policy 0 is good at task 0, policy 1 is good at task 1 and has
# forgotten task 0.
PAYOFF = [[10.0, 2.0],
          [1.0, 8.0]]


def test_row_evaluates_every_task_and_restores_the_env():
    env = FakeTaskEnv(PAYOFF)
    env.set_task(1)
    row = evaluate_policy_on_tasks(_policy(0), env, env.set_task, n_tasks=2, n_episodes=4,
                                   restore_task=1)
    # 3 steps per episode, averaged over 4 identical episodes.
    assert row == pytest.approx([30.0, 3.0])
    assert env.resets == 8, "n_tasks * n_episodes episodes must actually be run"
    assert env.task == 1, "restore_task must put the env back where it was found"


def test_bwt_and_fwt_on_a_two_task_case():
    env = FakeTaskEnv(PAYOFF)
    tm = evaluate_transfer_matrix([_policy(0), _policy(1)], env, env.set_task, n_episodes=2,
                                  baseline_policy_fn=_policy(0))
    expected = np.array([[30.0, 3.0],       # after task 0: good at 0, poor at 1
                         [6.0, 24.0]])      # after task 1: forgot 0, good at 1
    assert np.allclose(tm.matrix, expected)
    assert tm.is_complete()

    # BWT = R[1,0] - R[0,0] = 6 - 30 = -24: the classic forgetting signature.
    assert tm.bwt() == pytest.approx(-24.0)
    # FWT = R[0,1] - b[1] = 3 - 3 = 0 with policy 0 as the "random init" baseline.
    assert tm.fwt() == pytest.approx(0.0)


def test_incomplete_matrix_reports_none_rather_than_a_number():
    """A half-filled matrix must not silently produce a plausible-looking BWT."""
    tm = TransferMatrix(3)
    assert tm.bwt() is None and tm.fwt() is None and not tm.is_complete()
    tm.add_row(0, [1.0, 2.0, 3.0])
    assert tm.bwt() is None
    tm.add_row(1, [4.0, 5.0, 6.0])
    tm.add_row(2, [7.0, 8.0, 9.0])
    assert tm.is_complete()
    # BWT = mean(R[2,0]-R[0,0], R[2,1]-R[1,1]) = mean(7-1, 8-5) = 4.5
    assert tm.bwt() == pytest.approx(4.5)
    # FWT still needs the baselines.
    assert tm.fwt() is None
    tm.set_baselines([0.0, 0.0, 0.0])
    # FWT = mean(R[0,1], R[1,2]) = mean(2, 6) = 4.0
    assert tm.fwt() == pytest.approx(4.0)


def test_shape_and_index_errors_are_loud():
    tm = TransferMatrix(2)
    with pytest.raises(ValueError):
        tm.add_row(0, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        tm.add_row(5, [1.0, 2.0])
    with pytest.raises(ValueError):
        tm.set_baselines([1.0])
    with pytest.raises(ValueError):
        TransferMatrix(1)


def test_the_matrix_uses_the_untouched_reward_when_the_env_reports_one():
    """Drift envs put the honest, pre-normalization reward in info['directional_reward']."""
    class NormalizingEnv(FakeTaskEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            info["directional_reward"] = reward
            return obs, reward * 1000.0, term, trunc, info      # a "normalized" reward

    env = NormalizingEnv(PAYOFF)
    row = evaluate_policy_on_tasks(_policy(0), env, env.set_task, n_tasks=2, n_episodes=1)
    assert row == pytest.approx([30.0, 3.0]), "the normalized reward leaked into the return"
