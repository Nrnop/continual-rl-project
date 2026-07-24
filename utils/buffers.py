"""Rollout buffer (with split values) + permanent consolidation buffer.

The RolloutBuffer is a standard on-policy PPO buffer extended to store the permanent and transient
value estimates separately, so GAE can bootstrap from V = V_perm + V_trans while the PT agent can
still regress each head independently.

The ConsolidationBuffer is the on-policy analogue of the baseline's `exp_replay_PM`
(control/minatar_crl/PT_DQN_half.py): it keeps (state, old_V_perm) snapshots that the slow
permanent critic regresses toward `V_trans.detach() + old_V_perm` during consolidation.
"""
from collections import deque

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-length on-policy buffer for one PPO update."""

    def __init__(self, n_steps, obs_dim, act_dim, device):
        self.n_steps = n_steps
        self.device = device
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.logprobs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.v_perm = np.zeros(n_steps, dtype=np.float32)
        self.v_trans = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, v_perm, v_trans):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = done
        self.v_perm[i] = v_perm
        self.v_trans[i] = v_trans
        self.ptr += 1

    @property
    def full(self):
        return self.ptr >= self.n_steps

    def compute_gae(self, last_value, last_done, gamma, gae_lambda):
        """GAE-lambda over the combined value V = V_perm + V_trans.

        Returns (advantages, returns) as numpy arrays of shape (n_steps,).
        """
        values = self.v_perm + self.v_trans
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_nonterminal = 1.0 - last_done
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_value = values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def get_tensors(self):
        t = lambda x: torch.as_tensor(x, device=self.device)
        return {
            "obs": t(self.obs),
            "actions": t(self.actions),
            "logprobs": t(self.logprobs),
            "v_perm": t(self.v_perm),
            "v_trans": t(self.v_trans),
        }


class ConsolidationBuffer:
    """Rolling store of (state, old_V_perm) used for the slow permanent update.

    old_V_perm is the permanent value at the time the state was visited — captured BEFORE the
    permanent critic is updated, exactly like `old_p_vals` in the baseline's train_P_Net.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.old_v_perm = deque(maxlen=capacity)

    def add_batch(self, states_np, old_v_perm_np):
        for s, v in zip(states_np, old_v_perm_np):
            self.states.append(s.astype(np.float32))
            self.old_v_perm.append(np.float32(v))

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states.clear()
        self.old_v_perm.clear()

    def as_arrays(self):
        return (
            np.asarray(self.states, dtype=np.float32),
            np.asarray(self.old_v_perm, dtype=np.float32),
        )

    def iter_minibatches(self, batch_size, device, shuffle=True):
        n = len(self)
        states, old_v = self.as_arrays()
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            yield (
                torch.as_tensor(states[mb], device=device),
                torch.as_tensor(old_v[mb], device=device),
            )
