import numpy as np
import pytest
import torch
from src_continuous_control.utils.buffers import RolloutBuffer, ConsolidationBuffer


def test_rollout_buffer_soft_reset_and_overwrite():
    n_steps = 10
    num_envs = 2
    obs_dim = 4
    act_dim = 2
    device = torch.device("cpu")
    buffer = RolloutBuffer(n_steps, num_envs, obs_dim, act_dim, device)

    # Fill initial buffer with 1.0s (all args are per-env arrays with leading dim num_envs)
    for _ in range(n_steps):
        buffer.add(
            obs=np.ones((num_envs, obs_dim), dtype=np.float32),
            action=np.ones((num_envs, act_dim), dtype=np.float32),
            logprob=np.ones(num_envs, dtype=np.float32),
            reward=np.ones(num_envs, dtype=np.float32),
            done=np.zeros(num_envs, dtype=np.float32),
            v_perm=np.full(num_envs, 0.5, dtype=np.float32),
            v_trans=np.full(num_envs, 0.5, dtype=np.float32),
        )
    assert buffer.ptr == n_steps
    assert buffer.full is True

    # Soft reset
    buffer.reset()
    assert buffer.ptr == 0
    assert buffer.full is False
    # Check arrays not reallocated (still vectorized shape)
    assert buffer.obs.shape == (n_steps, num_envs, obs_dim)

    # Overwrite first 3 entries with 2.0s
    for _ in range(3):
        buffer.add(
            obs=np.full((num_envs, obs_dim), 2.0, dtype=np.float32),
            action=np.full((num_envs, act_dim), 2.0, dtype=np.float32),
            logprob=np.full(num_envs, 2.0, dtype=np.float32),
            reward=np.full(num_envs, 2.0, dtype=np.float32),
            done=np.ones(num_envs, dtype=np.float32),
            v_perm=np.ones(num_envs, dtype=np.float32),
            v_trans=np.ones(num_envs, dtype=np.float32),
        )
    assert buffer.ptr == 3
    assert np.all(buffer.obs[:buffer.ptr] == 2.0)
    # Beyond ptr, old data (1.0) remains on the underlying array; slicing up to ptr isolates active steps
    assert np.all(buffer.obs[buffer.ptr:] == 1.0)


def test_rollout_buffer_gae_terminal_boundaries():
    n_steps = 5
    num_envs = 1
    buffer = RolloutBuffer(n_steps=n_steps, num_envs=num_envs, obs_dim=2, act_dim=1,
                           device=torch.device("cpu"))
    for t in range(n_steps):
        # Step 2 is terminal (done=1.0). All args are per-env arrays (num_envs == 1 here).
        done = 1.0 if t == 2 else 0.0
        buffer.add(
            obs=np.zeros((num_envs, 2), dtype=np.float32),
            action=np.zeros((num_envs, 1), dtype=np.float32),
            logprob=np.zeros(num_envs, dtype=np.float32),
            reward=np.full(num_envs, float(t + 1), dtype=np.float32),
            done=np.full(num_envs, done, dtype=np.float32),
            v_perm=np.ones(num_envs, dtype=np.float32),
            v_trans=np.ones(num_envs, dtype=np.float32),
        )
    # values at each step t is v_perm + v_trans = 2.0
    # compute_gae returns FLATTENED (n_steps * num_envs,) == (n_steps,) for num_envs == 1.
    advantages, returns = buffer.compute_gae(last_value=2.0, last_done=0.0, gamma=0.9, gae_lambda=0.95)
    assert advantages.shape == (n_steps * num_envs,)
    assert returns.shape == (n_steps * num_envs,)
    # For t=1: next_nonterminal = 1.0 - dones[2] = 0.0 -> next_value masked, delta = 2.0 - 2.0 = 0.0.
    # returns == advantages + values holds elementwise; check it at every (flattened) step.
    values = (buffer.v_perm + buffer.v_trans).reshape(-1)
    assert np.allclose(returns, advantages + values)


def test_rollout_buffer_get_tensors():
    n_steps, num_envs, obs_dim, act_dim = 4, 2, 3, 2
    batch = n_steps * num_envs
    buffer = RolloutBuffer(n_steps=n_steps, num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim,
                           device=torch.device("cpu"))
    for _ in range(n_steps):
        buffer.add(
            obs=np.zeros((num_envs, obs_dim), dtype=np.float32),
            action=np.zeros((num_envs, act_dim), dtype=np.float32),
            logprob=np.zeros(num_envs, dtype=np.float32),
            reward=np.ones(num_envs, dtype=np.float32),
            done=np.zeros(num_envs, dtype=np.float32),
            v_perm=np.full(num_envs, 0.5, dtype=np.float32),
            v_trans=np.full(num_envs, 0.5, dtype=np.float32),
        )
    tensors = buffer.get_tensors()
    # get_tensors flattens (n_steps, num_envs, ...) -> (n_steps * num_envs, ...)
    assert tensors["obs"].shape == (batch, obs_dim)
    assert tensors["actions"].shape == (batch, act_dim)
    assert tensors["logprobs"].shape == (batch,)
    assert tensors["v_perm"].shape == (batch,)
    assert tensors["v_trans"].shape == (batch,)
    assert isinstance(tensors["obs"], torch.Tensor)


def test_consolidation_buffer_rolling_eviction_and_minibatches():
    capacity = 50
    buf = ConsolidationBuffer(capacity=capacity)
    assert len(buf) == 0

    # Add 80 items
    states = np.arange(80 * 4, dtype=np.float32).reshape(80, 4)
    old_v = np.arange(80, dtype=np.float32)
    buf.add_batch(states, old_v)

    assert len(buf) == capacity
    s_arr, v_arr = buf.as_arrays()
    assert s_arr.shape == (capacity, 4)
    assert v_arr.shape == (capacity,)
    # Check eviction: oldest 30 items evicted, so first item in buf should be index 30
    assert np.allclose(v_arr[0], 30.0)
    assert s_arr.dtype == np.float32

    # Check iter_minibatches without dropping samples
    batch_size = 16
    batches = list(buf.iter_minibatches(batch_size=batch_size, device=torch.device("cpu"), shuffle=False))
    # 50 total items / 16 = 3 full batches of 16 + 1 trailing batch of 2
    assert len(batches) == 4
    assert batches[0][0].shape == (16, 4)
    assert batches[-1][0].shape == (2, 4)

    # Check clear
    buf.clear()
    assert len(buf) == 0
