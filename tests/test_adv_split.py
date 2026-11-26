"""The advantage-split diagnostic behind Table 4.1 (`log_adv_split`).

The three parts must sum to the advantage PPO uses, and turning the diagnostic on must not change
training.
"""
import numpy as np
import torch

from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.utils.buffers import RolloutBuffer


def _filled_buffer(seed=0, n_steps=16, num_envs=3):
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(n_steps, num_envs, obs_dim=2, act_dim=1, device="cpu")
    buf.rewards[:] = rng.normal(size=(n_steps, num_envs))
    buf.v_perm[:] = rng.normal(size=(n_steps, num_envs))
    buf.v_trans[:] = rng.normal(size=(n_steps, num_envs))
    buf.dones[:] = (rng.random((n_steps, num_envs)) < 0.15).astype(np.float32)
    return buf, rng


def test_parts_sum_to_the_gae_advantage():
    """A_R + A_P + A_T equals the GAE advantage."""
    buf, rng = _filled_buffer()
    last_vp, last_vt = rng.normal(size=3), rng.normal(size=3)
    last_done = np.array([0.0, 1.0, 0.0])
    adv, _ = buf.compute_gae(last_vp + last_vt, last_done, 0.99, 0.95)
    a_r, a_p, a_t = buf.advantage_split(last_vp, last_vt, last_done, 0.99, 0.95)
    np.testing.assert_allclose(a_r + a_p + a_t, adv, atol=1e-5)


def test_exact_cancellation_reads_as_minus_one():
    """With v_trans = -v_perm, corr(A_P, A_T) reads as -1."""
    buf, rng = _filled_buffer(seed=1)
    buf.v_trans[:] = -buf.v_perm
    last_vp = rng.normal(size=3)
    a_r, a_p, a_t = buf.advantage_split(last_vp, -last_vp, np.zeros(3), 0.99, 0.95)
    assert np.corrcoef(a_p, a_t)[0, 1] < -0.999


def _cfg(log):
    cfg = {
        "lr_actor": 1e-3, "n_steps": 8, "num_envs": 2, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_coef": 0.2, "epochs": 2, "minibatch_size": 4, "ent_coef": 0.0,
        "max_grad_norm": 0.5, "normalize_advantage": True, "rho": 0.5, "kl_prior_coef": 0.01,
        "k": 1, "consolidation_epochs": 1, "critic_hidden_sizes": [8, 8], "hidden_sizes": [8, 8],
        "actor_trans_hidden_sizes": [4, 4], "critic_trans_hidden_sizes": [4, 4],
        "lr_trans": 1e-3, "lr_perm": 1e-3, "perm_optimizer": "adam", "rm_power": 0.5,
        "consolidation_buffer_size": 32,
    }
    if log:
        cfg["log_adv_split"] = True
    return cfg


def _run_update(log):
    torch.manual_seed(0)
    np.random.seed(0)          # the update shuffles minibatches with the global numpy generator
    agent = PPOPT(3, 2, _cfg(log), "cpu")
    rng = np.random.default_rng(0)
    for _ in range(8):
        obs = rng.normal(size=(2, 3)).astype(np.float32)
        vp, vt = agent.get_value(obs)
        act = rng.normal(size=(2, 2)).astype(np.float32)
        agent.buffer.add(obs, act, np.zeros(2, np.float32), rng.normal(size=2).astype(np.float32),
                         np.zeros(2, np.float32), vp, vt)
    metrics = agent.update(rng.normal(size=(2, 3)).astype(np.float32), np.zeros(2, np.float32), 1)
    return agent, metrics


def test_the_harness_itself_is_deterministic():
    """Two identical update runs give identical actor weights."""
    a, _ = _run_update(log=False)
    b, _ = _run_update(log=False)
    for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
        assert torch.equal(pa, pb)


def test_logging_on_trains_bit_identically():
    """log_adv_split adds its metrics without changing any trained weight."""
    a, ma = _run_update(log=False)
    b, mb = _run_update(log=True)
    for pa, pb in zip(list(a.actor.parameters()) + list(a.critic.parameters()),
                      list(b.actor.parameters()) + list(b.critic.parameters())):
        assert torch.equal(pa, pb)
    assert "adv_corr_perm_trans" not in ma
    for key in ("adv_var_perm_ratio", "adv_var_trans_ratio", "adv_var_reward_ratio"):
        assert key in mb and np.isfinite(mb[key])
    # The transient starts at the zero function (Theorem 1), so on the first update A_T is zero:
    # its variance is 0 and the correlation is undefined, reported as NaN.
    assert mb["adv_var_trans_ratio"] == 0.0 and np.isnan(mb["adv_corr_perm_trans"])
    assert mb["adv_split_residual"] < 1e-4
