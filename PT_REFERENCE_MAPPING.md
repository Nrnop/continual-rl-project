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

**Two separate networks, no weight sharing — this is not negotiable in the port.** `P_Net` and
`T_Net` are independent `CNN_half` instances in the reference, and `SplitCritic` is independent
`perm` / `trans` MLPs here. A shared-trunk variant with two *linear* heads was implemented for a
while, because it makes consolidation exact weight arithmetic rather than a lossy regression; it has
been **removed**. Their `control/minigrid/model.py::obj_net_two_heads` does share a conv trunk, but
it splits into two full multi-layer MLPs (`permanent_layer` / `transient_layer`), not linear heads —
so our variant matched neither reference architecture, and it is the MinAtar PT-DQN agent (two
separate networks) whose algorithm we port. FINDINGS.md §7 and §8 report runs made with the removed
variant.

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
`--decay` (λ), `--switch`, `--t-steps`. **`configs/pt_paper.yaml` is the current, faithful recipe**;
`ppo_pt.yaml` / `pt_fixed.yaml` are the older ones the first sweeps used and are kept only so those
runs remain reproducible.

| Reference flag / choice | Our key | `pt_paper.yaml` | (old `pt_fixed`) | Note |
|---|---|---|---|---|
| `--lr1` (P_Net) | `lr_perm` | `1e-5` | `1e-5` | α_P; MinAtar shipped `1e-8`, JBW swept `1e-4…1e-8` |
| `P_opt = optim.SGD(...)` | `perm_optimizer` | `"sgd"` | `"sgd"` | kept |
| `--lr2` (T_Net) | `lr_trans` | `3e-4` | `3e-4` | α_T; MinAtar `1e-4`, JBW `1e-2…1e-5` |
| `T_opt = optim.Adam(...)` | (hard-coded) | Adam | Adam | kept |
| **α_P ≪ α_T** (App. C.4) | — | `1e-5` vs `3e-4` | same | the ordering is the point |
| `--update` (P update every *n* env steps) | `k` | **`24`** | `10` | `k·n_steps·num_envs = 49 152` ≈ their 50 000; also does not divide the 300 updates/phase, so `on_switch` is finally testable (FINDINGS 5.1) |
| `maxlen=args.update` | `consolidation_buffer_size` | **`49152`** | `20480` | one full cycle, as theirs |
| `--batch-size 64` | `minibatch_size` | `64` | `64` | same chunk size for the P regression |
| one pass over the PM buffer | `consolidation_epochs` | `1` | `1` | theirs is `u_steps = size//batch − 1` |
| `--decay` (λ) | `decay` | **`0.75`** | `0.5` | MinAtar's value; best row of Tables C.13/C.17 |
| Eq. (4) target `old_P + T` | `value_preserving_consolidation` | **`false`** | (was `keep=1−decay`) | §4.1 |
| `V^(T)_0 = 0` (Theorem 1) | — | zero-init `θ_T` | (was random) | §4.1b |
| **PT-DQN-0.5x** (§6.1) | `critic_hidden_sizes` | **`[43,43]`** | (none → `[64,64]`) | 5 420 params vs vanilla's 5 377 = 1.008× |
| no LR annealing | `anneal_lr` | **`false`** | `true` | §4.5 |
| `--switch` | `switch` | `614400` | same | theirs 500k (MinAtar) / 150k (JBW) |
| `--t-steps` | `total_steps` | `3072000` (5 phases) | same | theirs 3.5M / 2.1M |
| `gamma` | `gamma` | `0.99` | same | their MinAtar value |
| 30 seeds, 90% CI | `plots/plot_compare.py` | 5 seeds, mean±CI | same | reduced for a bachelor's compute budget |

PPO's own knobs (`clip_coef`, `gae_lambda`, `epochs`, `ent_coef`, obs/reward normalization) are
**ours, not theirs**, and come from the validated CleanRL `ppo_continuous_action` recipe. Keeping
them fixed across `vanilla`/`pt`/`ewc` is what makes the PT contribution isolable — which is why
`configs/vanilla_paper.yaml` exists: `pt_paper.yaml` changes `anneal_lr` and the critic width, so
PT must be compared against a baseline carrying the same changes, not against the older sweep.

---

## 4. Deliberate deviations, and why

### 4.1 ~~A value-preserving consolidation target~~ — RETRACTED, we now use theirs

An earlier version of this port used `keep = 1 − decay` in place of their `keep = 1`, on the
reasoning that `P ← old_P + T` followed by `θ_T ← λ·θ_T` leaves the acting value overshooting by
`λ·T`, and that repeated inflation would destabilise PPO's advantages.

**That change was wrong and has been reverted.** The overshoot is not an accident of their code —
it is Eq. (4), which regresses `V^(P)` onto the full acting value `V^(PT)`, and Alg. 4 line 15
(`ŷ = Q^(P)(S,A) + Q^(T)(S,A;w)`). It is what gives `θ_P` the fixed point `E_τ[v_τ]`
(**Theorem 5**), the mean value function over the task distribution, which in turn optimises the
jumpstart objective (**Theorem 6**). A `keep = 1 − decay` operator converges somewhere else, so
none of that theory transfers to it. The residual `λ·T` is corrected by the fast transient within a
few updates, by design.

The old behaviour remains reachable via `value_preserving_consolidation: true`, solely so earlier
runs can be reproduced. It is not the algorithm.

### 4.1b Initialisation — what the reference actually does

**The paper specifies nothing.** Algs. 1, 2 and 4 say only `Initialize: θ, w`; §3.3 says *"The
initialization and resets are done appropriately based on the function approximation used."* So the
reference code is the only source of truth, and a direct search of it (2026-08-04) gives:

| file | permanent | transient |
|---|---|---|
| `prediction_semi_crl/tabular_linear/PT_Mem.py` | `w_1 = np.zeros_like(...)` | `w_2 = np.zeros_like(w_1)` |
| `control/tabular/PT_q_learning_crl.py` | `w_1 = np.zeros_like(...)` | `w_2 = np.zeros_like(w_1)` |
| `prediction_semi_crl/minigrid/model.py` | `nn.init.normal_(w, 0, 0.01)` | identical |
| `control/minatar_crl/model.py` | no explicit init (torch default) | identical |

**Both components start at (or within rounding of) zero in every version.** The tabular and linear
implementations — the ones the theorems apply to, and which produce Figs. 2, 3b and 4 — use exactly
zero. Note this means the reference's own deep agents do *not* satisfy Theorem 1's `V^(T)_0 = 0`
either; the theorem is a tabular result, not an implementation prescription.

We originally used `mlp(..., out_gain=1.0)` for both — orthogonal, gain 1.0. Measured on our
`[43,43]` critic over 2048 probe states:

```
ours   orthogonal out_gain=1.0     |V_perm| = 0.4046
ref    normal_(0, 0.01)            |V_perm| = 0.000112     <- 3628x smaller
ref    tabular  w_1 = 0            |V_perm| = 0.000000
```

Because α_P is small *by design* — it is the slow timescale — `θ_P` barely moves
(`perm/drift_from_init ≈ 0.3` against value magnitudes of O(1)), so `V_perm` stays dominated by its
initialisation for the whole run. The transient's target is then `R − V_perm`: the value function
minus a fixed unstructured function it must cancel on every state.

`perm_zero_init` / `trans_zero_init` (both default `true`) now reproduce the reference. See
`REINVESTIGATION.md` §6a–§6b for the measurement and the re-run.

### 4.2 Consolidation is only *approximate* with two separate trunks — and we keep it that way

Their `train_P_Net` must make one MLP/CNN *learn* the function `old_P + T` by regression, and their
decay scales *parameters*, which for a nonlinear net does not scale the *output* by `λ` (only
`λ = 0` is exact). We inherited both problems, measured them, and log them: `_consolidate` reports
`last_consolidation_error` (drift on fitted states) and `last_consolidation_error_holdout` (drift on
states withheld from the regression — the operationally relevant one, since the next rollout visits
new states). Measured at production settings: the regression transfers **0.03 %** of what it should
(320 SGD steps at `lr_perm = 1e-5` do not descend at all), while `θ_T ← 0.5·θ_T` leaves only **20 %**
of `V_trans`'s output rather than 50 % — a net **79 %** of the acting value destroyed per
consolidation.

Both properties are inherited from the reference, not introduced here, so we keep them: their
`--lr1 = 1e-8` (vs our `1e-5`) means their permanent net moves even less than ours, and their decay
is the same `params.data *= args.decay` on a nonlinear net. Making the transfer exact requires
changing the architecture — which is a different method, not this one.

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

### 4.4 The k / λ pairing

§6: *"For small values of k, large values of λ yield better performance … For large values of k, the
transient value function receives enough updates before the permanent value function is trained, to
attain low prediction error. Therefore, the updates to the permanent value function are effective,
and the transient predictions can be decayed more aggressively."* The two are co-dependent, and the
old config paired a **small k with a small λ** — the one quadrant the paper argues against. Worse,
the ratio is far harsher here than in DQN: their `train_T_Net` runs **once per env step**, so the
transient gets 50 000 gradient steps between consolidations; PPO gives us `k × epochs ×
(batch/minibatch)` = 7 680 at `k = 24` (and only 3 200 at the old `k = 10`). `pt_paper.yaml` uses
`k = 24`, `λ = 0.75`.

### 4.5 No LR annealing

The reference does not anneal. Our CleanRL-derived recipe annealed every optimizer's LR linearly to
zero, including `lr_trans` — so PT's ability to rebuild the transient after each decay degraded
monotonically over a run, a handicap manufactured by the PPO recipe rather than by the method.
`pt_paper.yaml` sets `anneal_lr: false`, and `vanilla_paper.yaml` does the same so the change is
controlled.

### 4.4 An Adam-state reset the reference does not need

Scaling `θ_T` touches only the parameters; Adam's `exp_avg` / `exp_avg_sq` for those parameters
survive, so the next step displaces the freshly-decayed weights with momentum from a network that no
longer exists. Their DQN runs one transient step per env step and dilutes this quickly; our PPO runs
`epochs × minibatches` steps immediately after each consolidation, so it does not. Hence
`reset_trans_optim_on_decay` (default `False`, to keep earlier runs reproducible) — see
`FINDINGS.md` §5.6.

---

### 4.6 Deviations that remain, and are not corrections

The August 2026 audit (`REINVESTIGATION.md`) fixed fifteen defects, but four settings still differ
from the reference **by choice**. Each was justified against a theorem; none was validated against
the reference's own behaviour. They are listed here so the distinction is never lost again.

| | reference code | ours | our justification |
|---|---|---|---|
| α_P | `1e-8`, constant | `2e-4` + Robbins-Monro | paper tunes α_P per domain (7 orders of magnitude across its own experiments); Thm 5 requires R-M |
| decay | all parameters (`params`) | output layer (`output`) | Alg. 2 line 9 says `λw` on the *parameters*, §3.2 prose says the *value function* — ambiguous under FA. **This is an interpretation, not a bug fix.** |
| λ | 0.75 | 0.95 | tuned (Job B) |
| k | 50 000 env steps | 122 880 | tuned |

**One structural infidelity cannot be configured away.** Their `train_T_Net` runs once per env step,
so the transient gets **1.0 gradient steps per env step**; PPO's arithmetic
(`epochs / minibatch = 10 / 64`) gives ours **0.156** — a 6.4× shortfall in how much the transient
can rebuild between consolidations. In a value-based agent the value function *is* the policy; in an
actor-critic it reaches the policy only through the advantage.

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
- ~~**The 0.5×-parameter budget**~~ — **now taken.** An earlier version of this port gave PT ~2× the
  critic parameters and dismissed the asymmetry as harmless. Appendix C.3 says otherwise: with the
  baseline scaled to matched capacity, "the DQN agent … catches up with our method after seeing
  enough data … When the agent's capacity is large relative to the complexity of the environment,
  there's no additional benefit (neither there is any downside) to our method." PT is a
  *big-world / small-agent* method, so parameter parity is a precondition for the comparison to be
  about the decomposition rather than about capacity. `critic_hidden_sizes: [43, 43]` in
  `pt_paper.yaml` gives 5 420 critic params against vanilla's 5 377. The actor is untouched and
  stays identical across all three agents.
- **The `DQN-multi-head` / `DQN-large-buffer` baselines** — we use vanilla PPO and online EWC
  instead, since EWC is the standard regularisation baseline for our continuous-control setting.

---

## 6. Where to look in our code

| Concern | File |
|---|---|
| PT agent (transient loss, consolidation, decay, boundary hooks) | `agents/ppo_pt.py` |
| Split critic (`SplitCritic` — two separate networks) | `models/critic.py` |
| `ConsolidationBuffer` (≙ `expReplay_PM`) | `utils/buffers.py` |
| Reward-sign-flip non-stationarity (≙ `CL_envs.py`) | `envs/directional_half_cheetah.py` |
| Switch schedule, boundary metrics | `train.py`, `utils/metrics.py` |
| Hyperparameters (≙ their shell-script flags) | `configs/ppo_pt.yaml`, `configs/pt_fixed.yaml` |
| Ablations isolating each PT ingredient | `configs/abl_pt_*.yaml` |
| Measured behaviour of the ported mechanism | `FINDINGS.md` |
