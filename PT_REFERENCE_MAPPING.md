# How we use Anand & Precup's PT algorithms and settings

**Reference paper.** N. Anand & D. Precup, *Prediction and Control in Continual Reinforcement
Learning*, NeurIPS 2023 (arXiv:2312.11669).

**Reference code.** `https://github.com/NishanthVAnand/prediction-and-control-in-continual-reinforcement-learning`

Of the repository's two halves — `prediction_semi_crl/` (Fig. 2) and `control/` (Figs. 3–5) — we
draw **only from `control/`**, and within it from the two *fully continual* control agents:

| Reference file | What we take from it |
|---|---|
| `control/minatar_crl/PT_DQN_half.py` | Algorithm 2 / Alg. 4 (PT-DQN CRL): the `train_T_Net` / `train_P_Net` split, the every-`k`-steps consolidate-then-decay cycle, and the two-separate-networks architecture |
| `control/minatar_crl/replay.py` (`expReplay_PM`) | The permanent-update buffer that stores `(state, action, old_P_value)` snapshots |
| `control/minatar_crl/run_minatar.sh` | The hyperparameter *ratios* (α_P ≪ α_T, SGD for P vs Adam for T, decay ∈ [0.5, 0.95]) |
| `control/minigrid/model.py` (`obj_net_two_heads`) | The shared-trunk / two-head architecture, which we implement as an alternative critic (`SharedTrunkSplitCritic`) |
| `control/tabular/CL_envs.py` | The reward-sign-flip style of non-stationarity, which we port to a continuous-control reward |

Everything below states what we reuse **verbatim in structure**, what we **had to change** to move
from value-based DQN with a replay buffer to on-policy PPO on MuJoCo, and why.

---

## 1. The algorithm we are porting

Their Alg. 2 (PT-Q-learning, CRL) is, per step *t*:

1. act with `Q = Q_P + Q_T`;
2. update the **transient** net `w` with a TD target that bootstraps on the *sum*;
3. every `k` steps: update the **permanent** net `θ` to absorb the transient over a buffer of
   visited states, then decay `w ← λ·w` and clear that buffer.

Our port keeps steps 1–3 exactly and changes only the *learner* the mechanism is attached to:
Q-learning/DQN → the **critic of PPO**, with continuous actions from a Gaussian actor.

**PT is critic-only in our work.** The paper decomposes the *value function*; §7.1 lists a
"permanent policy + transient correction" only as future work. So all three of our agents
(`vanilla`, `pt`, `ewc`) share one identical `GaussianActor`, and the sole difference between
`vanilla` and `pt` is the critic. That keeps the comparison attributable.

---

## 2. Component-by-component mapping

### 2.1 The additive decomposition — Eq. (2)/(3)

Their `PT_DQN_half.py` holds two *independent* CNNs and adds their outputs:

```python
# reference: PT_DQN_half.py
T_Net = CNN_half(in_channels, num_actions).to(device)
P_Net = CNN_half(in_channels, num_actions).to(device)
...
curr_Q_vals = curr_T_vals + curr_P_vals          # Q^(PT) = Q^(P) + Q^(T)
```

Ours is the state-value analogue, `models/critic.py`:

```python
class SplitCritic(nn.Module):
    """Dual-timescale state-value: V(s) = V_perm(s; theta_P) + V_trans(s; theta_T)."""

    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.perm  = mlp(obs_dim, list(hidden_sizes), 1, out_gain=1.0)
        self.trans = mlp(obs_dim, list(hidden_sizes), 1, out_gain=1.0)

    def forward(self, obs):
        return self.perm(obs).squeeze(-1), self.trans(obs).squeeze(-1)

    def value(self, obs):
        v_perm, v_trans = self.forward(obs)
        return v_perm + v_trans
```

`V = V_perm + V_trans` is what GAE bootstraps from, mirroring their acting on the sum
(`utils/buffers.py`, `RolloutBuffer.compute_gae`: `values = self.v_perm + self.v_trans`).

We also implement their **minigrid** architecture as a selectable variant. Their
`obj_net_two_heads` shares the conv trunk and splits into `permanent_layer` / `transient_layer`;
our `SharedTrunkSplitCritic` shares an MLP trunk and splits into two *linear* heads, which makes
consolidation exact weight arithmetic instead of a regression (see §4.2). Selected with
`critic_arch: "shared_trunk"`.

### 2.2 Transient update — Eq. (5)/(8), `train_T_Net`

Their transient learns the residual on top of a **frozen** permanent: `P_pred` is computed under
`torch.no_grad()` and the loss is on the *sum*, so the gradient reaches `w` only.

```python
# reference: PT_DQN_half.py
def train_T_Net():
    ...
    with torch.no_grad():
        T_next_pred = Target_net(next_states)
        P_next_pred = P_Net(next_states)
        P_pred      = P_Net(states).gather(1, actions)
    T_pred  = T_Net(states).gather(1, actions)
    targets = rewards + (1 - done) * gamma * ((P_next_pred + T_next_pred).max(1)[0]).reshape(-1, 1)
    loss    = T_criterion(T_pred + P_pred, targets)          # semi-gradient: only w moves
```

Ours, `agents/ppo_pt.py` — same structure, with the TD/max target replaced by the PPO **GAE
return** (the natural on-policy analogue) and `V_perm` detached to reproduce the semi-gradient:

```python
def critic_loss(self, batch, advantages, returns):
    """theta_T (fast): MSE(V_perm.detach() + V_trans, returns) — the transient head
    learns the residual above the *frozen* permanent baseline."""
    v_perm, v_trans = self.critic(batch["obs"])
    v_combined = v_perm.detach() + v_trans
    return 0.5 * ((v_combined - returns) ** 2).mean()
```

`theta_P` is deliberately **not** trained here — matching the reference, where `P_Net` never
appears in `T_opt`. (Regressing `V_perm` on returns every step double-counts against the acting
value and drove the divergence documented in `FINDINGS.md`; `configs/pt_fixed.yaml` is the
corrected recipe.)

### 2.3 The permanent buffer — `expReplay_PM`

They store `(state, action, V_P(state))` at *visit time*, i.e. the permanent value **before** the
permanent net is updated:

```python
# reference: replay.py
class expReplay_PM():
    def __init__(self, max_size, batch_size, device):
        self.memory = deque(maxlen=max_size)
    def store(self, obs, action, val_p): ...
```

Ours is the on-policy analogue, `utils/buffers.py` — no actions (state-value, not action-value),
and filled once per PPO rollout rather than once per env step:

```python
class ConsolidationBuffer:
    """Rolling store of (state, old_V_perm) used for the slow permanent update.

    old_V_perm is the permanent value at the time the state was visited — captured BEFORE the
    permanent critic is updated, exactly like `old_p_vals` in the baseline's train_P_Net.
    """
    def __init__(self, capacity):
        self.states     = deque(maxlen=capacity)
        self.old_v_perm = deque(maxlen=capacity)
```

Filled in `post_update`, using the rollout's own states so the buffer tracks the on-policy state
distribution (their `deque(maxlen=args.update)` does the same by construction):

```python
def post_update(self, update_idx):
    states = self.buffer.obs.reshape(-1, self.obs_dim)
    with torch.no_grad():
        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        old_v_perm, _ = self.critic(s_t)
    self.consolidation_buffer.add_batch(states, old_v_perm.cpu().numpy())

    self._updates_since_consolidation += 1
    if self._updates_since_consolidation >= self.k:
        self._consolidate()
        self._updates_since_consolidation = 0
```

### 2.4 Consolidation + decay — Eq. (4)/(7), `train_P_Net`

The heart of the method. Theirs sweeps the PM buffer in `batch_size` chunks and regresses
`P_pred → T_pred + old_p_vals`, then scales every transient parameter by `λ`:

```python
# reference: PT_DQN_half.py
def train_P_Net():
    for p_update in range(u_steps):
        states, actions, old_p_vals = ...
        with torch.no_grad():
            T_pred = T_Net(states).gather(1, actions)
        P_pred = P_Net(states).gather(1, actions)
        loss   = P_criterion(P_pred, T_pred + old_p_vals)
        P_opt.zero_grad(); loss.backward(); P_opt.step()

# main loop
if (step + 1) % args.update == 0:
    p_loss = train_P_Net()
    for params in T_Net.parameters():
        params.data *= args.decay
```

Ours, `agents/ppo_pt.py::_consolidate` — same loop, same optimizer split, one deliberate change to
the target (`keep = 1 - decay`, explained in §4.1):

```python
keep = 1.0 - self.transient_decay
for _ in range(self.consolidation_epochs):
    for s_mb, old_vp_mb in self.consolidation_buffer.iter_minibatches(
            self.minibatch_size, self.device):
        v_perm, v_trans = self.critic(s_mb)
        target = old_vp_mb + keep * v_trans.detach()      # reference uses keep = 1
        loss   = 0.5 * ((v_perm - target) ** 2).mean()
        self.perm_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.perm.parameters(), self.max_grad_norm)
        self.perm_optim.step()
...
self._decay_transient(self.transient_decay)               # theta_T <- decay * theta_T
self.consolidation_buffer.clear()                         # their `exp_replay_PM` deque rolls over
```

and the decay itself is their `params.data *= args.decay`, verbatim:

```python
@torch.no_grad()
def decay_transient(self, decay):
    """theta_T <- decay * theta_T  (decay=0 ~= reset). Mirrors `params.data *= args.decay`."""
    for p in self.trans.parameters():
        p.data.mul_(decay)
```

### 2.5 Non-stationarity — `CL_envs.py`

Their control environments flip the sign of goal rewards on a fixed step schedule (tabular:
`(1.0, -1.0) ↔ (-1.0, 1.0)`; JBW: red/blue alternate between −1 and +2 every 150k steps; MinAtar:
resample the task every 500k steps). We port the *reward-sign flip* to continuous control:
`DirectionalHalfCheetah` flips the sign of the forward-velocity term while keeping the control cost
(shared physics) task-invariant:

```python
def _directional_reward(self, info):
    """Reconstruct reward = direction * fwd_velocity_term + ctrl_cost (shared)."""
    run_term = self.forward_reward_weight * info.get("x_velocity")
    return self.direction * run_term + info.get("reward_ctrl", 0.0)
```

As in their loop (`if (step+1) % args.switch == 0: env = CL_envs_func(...)`), the *training loop*
drives the switch, not the env:

```python
# train.py
next_switch = (task_idx + 1) * switch_interval
if global_step >= next_switch:
    task_idx += 1
    direction = tasks[task_idx % len(tasks)]
    env.unwrapped.call("set_task", direction)
    agent.on_task_switch(global_step)
```

---

## 3. Settings: their flags → our config keys

Their `PT_DQN_half.py` exposes `--lr1` (permanent), `--lr2` (transient), `--update` (k),
`--decay` (λ), `--switch`, `--t-steps`. The full mapping to `configs/ppo_pt.yaml` /
`configs/pt_fixed.yaml`:

| Reference flag / choice | Our key | Our value | Note |
|---|---|---|---|
| `--lr1` (P_Net) | `lr_perm` | `1e-5` | α_P; MinAtar used `1e-8`, JBW swept `1e-4…1e-8` |
| `P_opt = optim.SGD(...)` | `perm_optimizer` | `"sgd"` | kept: paper §7 argues P must move slowly |
| `--lr2` (T_Net) | `lr_trans` | `3e-4` | α_T; MinAtar `1e-4`, JBW `1e-2…1e-5` |
| `T_opt = optim.Adam(...)` | (hard-coded) | Adam | kept |
| **α_P ≪ α_T** (Appendix C.4) | — | `1e-5` vs `3e-4` | the ordering is the point, and we preserve it |
| `--update` (P update every *n* env steps) | `k` | `10` PPO updates | `k · n_steps · num_envs = 20480` env steps |
| `maxlen=args.update` on `expReplay_PM` | `consolidation_buffer_size` | `20480` | exactly one consolidation cycle, as theirs |
| `--batch-size 64` (P sweep chunk) | `minibatch_size` | `64` | same chunk size for the P regression |
| one pass over the PM buffer | `consolidation_epochs` | `1` | theirs is `u_steps = size//batch - 1`, i.e. 1 pass |
| `--decay` (λ) | `decay` | `0.5` | MinAtar `0.75`; JBW reported `0.55 / 0.75 / 0.95` |
| `--switch` | `switch` | `614400` env steps | theirs 500k (MinAtar) / 150k (JBW) |
| `--t-steps` | `total_steps` | `3072000` (5 phases) | theirs 3.5M (MinAtar) / 2.1M (JBW) |
| `gamma` from `misc_params.cfg` | `gamma` | `0.99` | same as their MinAtar value |
| 30 seeds, 90% CI | `scripts/run_all.sh`, `plots/plot_compare.py` | 5 seeds, mean±CI | reduced for a bachelor's compute budget |

Two settings are **ours, not theirs**, because the base learner differs: PPO's own knobs
(`clip_coef`, `gae_lambda`, `epochs`, `ent_coef`, obs/reward normalization, LR anneal) come from
the validated CleanRL `ppo_continuous_action` recipe (`configs/cleanrl_match.yaml`), not from the
PT paper. Keeping those fixed across `vanilla`/`pt`/`ewc` is what makes the PT contribution
isolable.

---

## 4. Deliberate deviations, and why

### 4.1 A value-preserving consolidation target

Their target is `P ← old_P + T` (`keep = 1`) while the decay is `λ = 0.75`. For any `λ > 0` this
inflates the acting value by `λ·T` on every cycle, because the transient is counted once inside `P`
and once again in the residual `λ·T`. Ours uses `keep = 1 − decay`:

```
V_new = P_new + decay·T = old_P + (1-decay)·T + decay·T = old_P + T = V_old
```

so `V = V_perm + V_trans` is unchanged across consolidation for *any* decay. `decay = 0` reduces to
their case (hard reset, `P` absorbs all of `T`) and the two targets coincide. This matters far more
here than in DQN: over ~150 consolidations per run, a repeated multiplicative inflation of the
critic destabilises PPO's advantage estimates.

### 4.2 Consolidation is only *approximate* with two separate trunks

Their `train_P_Net` must make one MLP/CNN *learn* the function `old_P + T` by regression, and their
decay scales *parameters*, which for a nonlinear net does not scale the *output* by `λ` (only
`λ = 0` is exact). We inherited both problems, measured them, and log them: `_consolidate` reports
`last_consolidation_error` (drift on fitted states) and `last_consolidation_error_holdout` (drift on
states withheld from the regression — the operationally relevant one, since the next rollout visits
new states). This is why we also implement `SharedTrunkSplitCritic`, following their *minigrid*
`obj_net_two_heads`: with two **linear** heads on a shared trunk, `V = (w_P + w_T)·φ(s)`, so
consolidation is exact arithmetic with zero drift and no buffer at all:

```python
@torch.no_grad()
def consolidate(self, decay):
    keep = 1.0 - decay
    self.perm.weight.add_(keep * self.trans.weight)
    self.perm.bias.add_(keep * self.trans.bias)
    self.trans.weight.mul_(decay)
    self.trans.bias.mul_(decay)
```

### 4.3 Boundary handling: semi-continual on top of fully-continual

Their Alg. 2 is boundary-agnostic (`mod(t, k) == 0` only). Because our switch schedule is known, we
additionally expose the Alg. 1 / Alg. 3 *semi*-continual behaviour at the boundary — consolidate
first, *then* decay, locking the just-learned task value into `θ_P`:

```python
def on_task_switch(self, step):
    mode = self.cfg.get("on_switch") or ("consolidate" if self.cfg.get("consolidate_on_switch", True) else "decay")
    if mode == "consolidate":
        self._consolidate()               # absorbs T into P, then decays T
        self._updates_since_consolidation = 0
    elif mode == "decay":
        if self.transient_decay < 1.0:
            self._decay_transient(self.transient_decay)
    # mode == "none": pure Alg. 2, periodic k-step consolidation only
```

`on_switch: "none"` reproduces their fully-continual setting exactly; it is one of the ablations in
`configs/abl_pt_*.yaml`.

### 4.4 An Adam-state reset the reference does not need

Scaling `θ_T` touches only the parameters; Adam's `exp_avg` / `exp_avg_sq` for those parameters
survive, so the next step displaces the freshly-decayed weights with momentum from a network that no
longer exists. Their DQN runs one transient step per env step and dilutes this quickly; our PPO runs
`epochs × minibatches` steps immediately after each consolidation, so it does not. Hence
`reset_trans_optim_on_decay` (default `False`, to keep earlier runs reproducible) — see
`FINDINGS.md` §5.6.

---

## 5. What we deliberately did **not** take

- **Target network** (`Target_net`, refreshed every 1000 steps) — PPO regresses on Monte-Carlo/GAE
  returns, not a bootstrapped max, so there is no moving target to stabilise.
- **Experience replay** (`expReplay`, `maxlen=100000`) — PPO is on-policy. Only the *permanent*
  buffer survives, because Eq. (4) is inherently a replay over visited states.
- **ε-greedy exploration** (`epsilon` from `misc_params.cfg`) — replaced by the Gaussian policy's
  own entropy.
- **`Q^(P)`/`Q^(T)` over discrete actions** (`.gather(1, actions)`) — we decompose `V(s)`, not
  `Q(s,a)`, since PPO's critic is a state-value function.
- **The 0.5×-parameter budget** (`CNN_half`, "PT-DQN-0.5x") — their parameter-matching trick against
  DQN. We keep `hidden_sizes` identical across `vanilla`/`pt`/`ewc`, so PT has ~2× the *critic*
  parameters. Their Appendix C.3 ablation shows PT still wins at equal *and* at 2× capacity, so we
  accept the asymmetry and note it; the actor, which dominates behaviour, is identical.
- **The `DQN-multi-head` / `DQN-large-buffer` baselines** — we use vanilla PPO and online EWC
  instead, since EWC is the standard regularisation baseline for our continuous-control setting.

---

## 6. Where to look in our code

| Concern | File |
|---|---|
| PT agent (transient loss, consolidation, decay, boundary hooks) | `agents/ppo_pt.py` |
| Split critics (`SplitCritic`, `SharedTrunkSplitCritic`) | `models/critic.py` |
| `ConsolidationBuffer` (≙ `expReplay_PM`) | `utils/buffers.py` |
| Reward-sign-flip non-stationarity (≙ `CL_envs.py`) | `envs/directional_half_cheetah.py` |
| Switch schedule, boundary metrics | `train.py`, `utils/metrics.py` |
| Hyperparameters (≙ their shell-script flags) | `configs/ppo_pt.yaml`, `configs/pt_fixed.yaml` |
| Ablations isolating each PT ingredient | `configs/abl_pt_*.yaml` |
| Measured behaviour of the ported mechanism | `FINDINGS.md` |
