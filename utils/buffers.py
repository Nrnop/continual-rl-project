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
    """Fixed-length on-policy buffer for one PPO update, vectorized over `num_envs`.

    Storage is shaped (n_steps, num_envs, ...); GAE is computed per-env and the
    returned advantages/returns (and get_tensors) are flattened to a single batch
    of size n_steps * num_envs, so the PPO update loop is env-count-agnostic.
    A single synchronous env is just num_envs == 1.
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
        """All arguments are per-env arrays with leading dim num_envs.

        `done` is the done flag of `obs` at this step (CleanRL convention:
        dones[t] marks whether obs[t] is the first state after a reset), and
        `reward` is the reward received for the action taken at this step.
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
        """GAE-lambda over the combined value V = V_perm + V_trans, per env.

        last_value, last_done: per-env arrays (num_envs,) for the bootstrap step.
        Returns (advantages, returns) FLATTENED to (n_steps * num_envs,), matching
        the row-major flattening used by get_tensors.
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

    def compute_gae_components(self, last_v_perm, last_v_trans, last_done, gamma, gae_lambda):
        """Split the advantage into the parts contributed by rewards, V_perm and V_trans.

        THE DECOMPOSITION IS EXACT, not an attribution heuristic. The TD residual is affine in the
        value function,

            delta_t = r_t + gamma*(1-d)*V(s') - V(s),     V = V_perm + V_trans

        so it separates into delta_r + delta_perm + delta_trans with no remainder, and GAE is a
        linear filter over delta. Therefore

            A = A_reward + A_perm + A_trans

        holds elementwise, and `test_advantage_decomposition_is_exact` pins it.

        Why this is the measurement that answers "how much influence does the critic have on the
        policy": in PPO the critic reaches behaviour ONLY through the advantage. Whatever the
        critic knows that does not show up in A cannot affect a single action the agent takes.
        A_perm is therefore the permanent component's entire influence on decision-making.

        Note what A_perm actually contains: `gamma*V_perm(s') - V_perm(s)`, a TEMPORAL DIFFERENCE
        of the permanent, not its level. A slow, smooth V_perm has a small one — and any constant
        offset cancels here exactly, before advantage normalisation gets a chance to remove it
        again. The permanent's influence on the policy is structurally attenuated twice over.

        Returns (adv_reward, adv_perm, adv_trans), each flattened like `compute_gae`.
        """
        last_done = np.asarray(last_done, dtype=np.float32).reshape(self.num_envs)
        parts = []
        for values_c, last_c in (
                (None, None),                                                   # rewards
                (self.v_perm, last_v_perm),
                (self.v_trans, last_v_trans)):
            adv_c = np.zeros_like(self.rewards)
            last_gae = np.zeros(self.num_envs, dtype=np.float32)
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_nonterminal = 1.0 - last_done
                    next_value = (np.asarray(last_c, dtype=np.float32).reshape(self.num_envs)
                                  if values_c is not None else 0.0)
                else:
                    next_nonterminal = 1.0 - self.dones[t + 1]
                    next_value = values_c[t + 1] if values_c is not None else 0.0
                if values_c is None:
                    delta = self.rewards[t]
                else:
                    delta = gamma * next_value * next_nonterminal - values_c[t]
                last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
                adv_c[t] = last_gae
            parts.append(adv_c.reshape(-1))
        return tuple(parts)

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
