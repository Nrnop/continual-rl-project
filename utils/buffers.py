"""PPO rollout buffer with split values, and the buffers used for permanent consolidation.

RolloutBuffer stores V_perm and V_trans separately; GAE bootstraps from their sum.
ConsolidationBuffer keeps (state, old_V_perm) pairs, the on-policy analogue of `exp_replay_PM` in
the reference code; the permanent regresses toward `V_trans.detach() + old_V_perm`.
"""
from collections import deque

import numpy as np
import torch


class RolloutBuffer:
    """On-policy buffer for one PPO update, vectorized over `num_envs`.

    Storage is (n_steps, num_envs, ...). GAE runs per env; advantages, returns and get_tensors
    are flattened to a batch of n_steps * num_envs.
    """

    def __init__(self, n_steps, num_envs, obs_dim, act_dim, device):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.batch_size = n_steps * num_envs
        self.device = device
        self.obs = np.zeros((n_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, num_envs, act_dim), dtype=np.float32)
        self.logprobs = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.v_perm = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.v_trans = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, v_perm, v_trans):
        """Store one step. All arguments are per-env arrays with leading dim num_envs.

        `done` follows the CleanRL convention: dones[t] marks obs[t] as the first state after a
        reset. `reward` is the reward for the action taken at this step.
        """
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
        """GAE-lambda over V = V_perm + V_trans, per env.

        last_value, last_done: arrays of shape (num_envs,) for the bootstrap step.
        Returns (advantages, returns), each flattened to (n_steps * num_envs,) in the same
        row-major order as get_tensors.
        """
        values = self.v_perm + self.v_trans                 # (n_steps, num_envs)
        last_value = np.asarray(last_value, dtype=np.float32).reshape(self.num_envs)
        last_done = np.asarray(last_done, dtype=np.float32).reshape(self.num_envs)
        advantages = np.zeros_like(values)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
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
        return advantages.reshape(-1), returns.reshape(-1)

    def advantage_split(self, last_v_perm, last_v_trans, last_done, gamma, gae_lambda):
        """Split the GAE advantage into the parts from the reward, V_perm and V_trans.

        GAE is linear in rewards and values, so three runs of the recursion (rewards with zero
        values, zero rewards with V_perm, zero rewards with V_trans) give A_R + A_P + A_T = A
        exactly. Under a split critic the transient is trained on R - V_perm, so A_T tends to
        -A_P (Table 4.1). Diagnostic only; does not modify the buffer.
        Returns (A_R, A_P, A_T), each flattened to (n_steps * num_envs,).
        """
        last_done = np.asarray(last_done, dtype=np.float32).reshape(self.num_envs)

        def gae(rewards, values, last_value):
            last_value = np.asarray(last_value, dtype=np.float32).reshape(self.num_envs)
            adv = np.zeros_like(values)
            last_gae = np.zeros(self.num_envs, dtype=np.float32)
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    nonterminal, next_value = 1.0 - last_done, last_value
                else:
                    nonterminal, next_value = 1.0 - self.dones[t + 1], values[t + 1]
                delta = rewards[t] + gamma * next_value * nonterminal - values[t]
                last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
                adv[t] = last_gae
            return adv.reshape(-1)

        zeros_v = np.zeros_like(self.v_perm)
        zeros_r = np.zeros_like(self.rewards)
        a_r = gae(self.rewards, zeros_v, np.zeros(self.num_envs, dtype=np.float32))
        a_p = gae(zeros_r, self.v_perm, last_v_perm)
        a_t = gae(zeros_r, self.v_trans, last_v_trans)
        return a_r, a_p, a_t

    def get_tensors(self):
        """Flattened (batch, ...) tensors, batch = n_steps * num_envs."""
        obs_dim = self.obs.shape[-1]
        act_dim = self.actions.shape[-1]
        t = lambda x: torch.as_tensor(x, device=self.device)
        return {
            "obs": t(self.obs.reshape(-1, obs_dim)),
            "actions": t(self.actions.reshape(-1, act_dim)),
            "logprobs": t(self.logprobs.reshape(-1)),
            "v_perm": t(self.v_perm.reshape(-1)),
            "v_trans": t(self.v_trans.reshape(-1)),
        }


class ConsolidationBuffer:
    """Rolling store of (state, old_V_perm) pairs for the permanent update.

    old_V_perm is captured when the state is visited, before the permanent is updated, like
    `old_p_vals` in the reference code's train_P_Net.
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


class StateConsolidationBuffer:
    """Rolling state-only buffer for actor and critic PT-PPO consolidation."""

    def __init__(self, capacity):
        if int(capacity) <= 0:
            raise ValueError("consolidation buffer capacity must be positive")
        self.capacity = int(capacity)
        self.states = deque(maxlen=self.capacity)

    def add_batch(self, states_np):
        states = np.asarray(states_np, dtype=np.float32)
        if states.ndim == 1:
            states = states[None, :]
        for state in states:
            self.states.append(np.array(state, dtype=np.float32, copy=True))

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states.clear()

    def as_array(self):
        if not self.states:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(self.states, dtype=np.float32)

    def as_arrays(self):
        return self.as_array()

    def iter_minibatches(self, batch_size, device, shuffle=False):
        states = self.as_array()
        indices = np.arange(len(states))
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start:start + int(batch_size)]
            yield torch.as_tensor(states[batch_indices], device=device)
