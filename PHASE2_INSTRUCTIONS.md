# Phase 2 — instructions

Decisions from the supervisor meeting, turned into a work plan, with the follow-up clarifications
folded in. This supersedes the Phase 1 design; that record is now at `archive/phase1/FULL_PT.md`.

**Status: archiving is done, no code work has started.** Begin at §6, task T1.

**The key clarification:** task boundaries **stay**. What changes at a boundary is the *physics*
(friction, damping, …) instead of the reward sign. That resolves both open blockers from the first
draft — EWC works unchanged, and forward/backward transfer keep their standard definitions.

---

## 1. The design

### Phase 2a — boundary-based dynamics change (this job)

| | |
|---|---|
| **Environment** | HalfCheetah only. Point-mass dropped entirely. |
| **Non-stationarity** | At each task switch, **physics parameters change**. Reward function is fixed (always "run forward"). |
| **Parameters** | Start with `["damping", "friction"]`. Then a harder variant changing **all** of them (§2.4). |
| **Agents** | `vanilla`, `ewc`, `pt` — where `pt` is the current `pt_full`: split actor **and** split critic. |
| **Seeds** | 5 per method. |
| **Task boundaries** | Observable, as in the paper's *semi-continual* setting. |

3 agents × 5 seeds = **15 runs**, ≈3.5–4 h locally at 7-way parallelism. No remote box needed.

### Phase 2b — boundary-free (the job after 2a works)

Remove boundaries entirely: continuous drift plus **online EWC** (periodic Fisher accumulation on
a fixed interval). Do not start this until 2a is producing clean numbers — it changes two things
at once otherwise.

---

## 2. Resolved from the first draft

### 2.1 EWC — no longer a blocker ✅

With boundaries retained, `PPOEWC.on_task_switch()` fires exactly as it always has, the Fisher
accumulates normally, and EWC is a real baseline. **No change to `ppo_ewc.py` for Phase 2a.**

Carry forward to 2b: online EWC needs periodic Fisher consolidation behind a new
`ewc_consolidate_every`, ideally matched to `pt`'s consolidation period *k* so neither method gets
a cadence advantage.

*(For the record, this is why it mattered: with no boundaries the Fisher is never accumulated, the
penalty is identically zero, and EWC becomes vanilla PPO — we measured p = 1.000, identical to the
decimal, in Phase 1.)*

### 2.2 Forward / backward transfer — no longer a blocker ✅

With a discrete task sequence, the standard Lopez-Paz & Ranzato definitions apply directly. Tasks
are the physics configurations between boundaries.

Build the evaluation matrix `R[i, j]` = mean return on **task j** after finishing **task i**,
frozen policy, no exploration noise, 10 episodes per cell:

- **BWT** = mean over `j < N` of `R[N, j] − R[j, j]` — retention on physics already seen.
- **FWT** = mean over `j > 1` of `R[j−1, j] − b_j` — zero-shot competence on physics not yet seen,
  where `b_j` is a random-init policy's return on task *j*.

**One requirement this imposes on the task sequence: it must revisit.** BWT is meaningless if
every task is new. Use a cyclic set of physics settings — e.g. multipliers `[1.0, 1.6, 0.6, 1.6,
0.6]` over 5 phases, so tasks 2 and 4 are the same physics, as are 3 and 5. This mirrors the
reward-switch design that Phase 1 used and keeps BWT well-defined.

Implementation: `evaluate_transfer_matrix()` in `utils/metrics.py`, plus the ability to set the
env to a named task's physics for evaluation without disturbing training.

### 2.3 Plot (c) — one figure, and this is the one 🎯

You asked for simplicity, and I agree one figure is better here. **Use the ablation.**

> **Three bars: `vanilla` · `pt` with the permanent frozen · `pt` (both components live).**

It reads instantly and it decomposes the question exactly:
- bar 1 → bar 2 = what having a **split at all** buys
- bar 2 → bar 3 = what the **permanent actually learning** buys

That is precisely "the effect of having 2 components," and no explanation is needed to read it.

**Keep two diagnostics as logged numbers, not figures** — one sentence each in the text:

- `probe/decay_gain` — at each consolidation: evaluate, decay μ_T, evaluate again, **no gradient
  step in between**. Any change is behaviour caused by the decomposition alone. This is our
  insurance: with the shrink control dropped, it is the only thing that can distinguish "the
  decomposition works" from "the decay works". Costs one extra eval per boundary.
- `diag/actor_perm_trans_corr` — if this sits near −1 the two components are cancelling, and
  the ablation figure will be flat. Better seen on day one than after the sweep.

### 2.4 Parameters — default first, then a real hypothesis ✅

Run 1: `["damping", "friction"]`. Both are read straight out of the MuJoCo model each step, so no
derived quantities go stale.

Run 2 (the harder variant): change **all** of `["damping", "friction", "mass", "armature"]` at each
switch. `mass` triggers a `mj_setConst` inertia refresh — already handled in `_apply`.

**This is not a fishing expedition, and it is worth framing as a hypothesis in the thesis.** The
original paper's Appendix C.3 states its own boundary condition:

> "our approach is beneficial in the **big world - small agent** setup where the computation budget
> of the agent is very small compared to the complexity of the environment. When the agent's
> capacity is large relative to the complexity of the environment, there's no additional benefit."

Harder dynamics = a more complex environment relative to a fixed-capacity agent. So the paper
**predicts** PT should do better in run 2 than run 1. That makes this a directional test of the
paper's own claim rather than an extra sweep — and it is a genuinely good experiment.

Avoid `dof_frictionloss`: all zeros in this model, so multiplicative change is a no-op.

---

## 3. New code needed

The one thing that does not exist yet.

**`envs/drift_half_cheetah.py` needs a `schedule="step"` mode.** Today `multiplier(t)` only
produces `sin` or `linear` — both continuous. Phase 2a needs the multiplier **held constant within
a task and changed at the boundary**:

```python
# schedule="step": piecewise-constant physics, one setting per task.
# task_multipliers cycles, so tasks repeat and BWT stays well-defined.
task_multipliers: [1.0, 1.6, 0.6, 1.6, 0.6]
```

Small change — a branch in `multiplier()` plus a `set_task(i)` that selects the entry. Keep `sin`
and `linear` intact; Phase 2b uses them.

**Add a test** asserting the physics actually change at a boundary and stay fixed between them.
A silently-constant environment would look like a working experiment and produce three identical
agents — this is exactly the failure mode that cost a week in Phase 1.

---

## 4. Files to MODIFY

| file | change |
|---|---|
| `agents/ppo_pt_full.py` → **rename `agents/ppo_pt.py`** | class `PPOPTFull` → `PPOPT`. This becomes *the* PT agent. |
| `agents/__init__.py` | drop the old `PPOPT`; register the renamed class as `"pt"`; remove `"pt_full"`. |
| `agents/ppo_base.py` | **remove** `policy_shrink_every`, `policy_shrink_factor`, `critic_shrink`, `shrink_flush_optim` and the shrink block in `post_update` (~lines 63–70, 148–170). Keep `mu_l2_coef`, `log_std_init`, `freeze_log_std`. |
| `agents/ppo_vanilla.py` | `post_update` reverts to a no-op once the base hook is empty. |
| `agents/ppo_ewc.py` | **no change for 2a.** `ewc_consolidate_every` in 2b. |
| `envs/drift_half_cheetah.py` | add `schedule="step"` + `set_task(i)` (§3). |
| `train.py` | remove `point_tasks` / `point_drift` modes and imports (lines 34–35); default `env_mode: drift`; route `set_task` to the drift wrapper at boundaries; wire the transfer-matrix eval. |
| `utils/metrics.py` | add `evaluate_transfer_matrix()`. `BoundaryReturnTracker` / `JumpstartTracker` still apply — boundaries exist. |
| `configs/default.yaml` | `env_mode: drift`, `schedule: step`, `drift_targets: ["damping","friction"]`, `task_multipliers`, 5 seeds. |
| `configs/ppo_pt_full.yaml` → **rename `configs/ppo_pt.yaml`** (overwrites the old). |
| **new** `configs/phase2_hard.yaml` | the all-parameters variant (§2.4 run 2). |
| `plots/plot_compare.py` | basis for plot (a), return vs timestep. |
| **new** `plots/make_phase2_figures.py` | the three figures: (a) return curves, (b) FWT/BWT, (c) the ablation bars. |
| `tests/test_paper_fidelity.py`, `test_pt_full_agent.py`, `test_split_actor.py` | update imports/agent key for the rename. These three are the PT tests worth keeping. |
| `tests/test_drift_env.py` | extend for `schedule="step"`. |
| `CLAUDE.md`, `README.md` | rewrite — both currently describe reward-switch as the design. |

---

## 5. Archiving — DONE ✅

Completed and pushed. `main` is tagged **`phase1-final`** at the last Phase 1 commit; everything
below is recoverable with `git checkout phase1-final -- <path>` or by moving files back.

**Archived to `archive/phase1/`** — 205 files:

| group | count |
|---|---:|
| `configs/` — 130 `stage*`, 23 `abl_*`, 11 Phase 1 overlays | 164 |
| `scripts/` — stage runners, ablation runners, sweeps, analysis | 22 |
| `plots/` — Phase 1 and pre-Phase-1 figure generators | 11 |
| `docs/` — `FULL_PT.md`, `FINDINGS.md`, `REINVESTIGATION.md`, `PT_REFERENCE_MAPPING.md`, `VASTAI_SETUP.md`, `figures_full_pt_guide.md`, `MEETING_BRIEF.md`, `SESSION_LOG.md` | 8 |
| `tests/test_simple_drift_plot.py` (tested an archived plot script) | 1 |

`archive/phase1/README.md` records what the supervisors' decision on the shrinkage baseline does
and does not invalidate — several Phase 1 measurements stand independently of it and may still be
cited. **Read it before reusing any Phase 1 number.**

**Deliberately NOT archived:**
- `plots/figures_pt_full/` — committed images, still the evidence for Phase 1.
- `PT_full.md` — the specification, not a result.
- `envs/directional_half_cheetah.py` — **`drift_half_cheetah.py` imports `make_base_env` from it.**
  The file stays; only the reward-flip *usage* is being dropped.

**Left in place until the code work happens** (archiving them now would break imports — they go in
tasks T2/T4 below): `agents/ppo_pt.py`, `envs/mock_continual.py`, `envs/simple_drift.py`, and the
tests that cover them.

**Working tree now:** 8 configs, 5 scripts, 2 plot modules, 5 markdown files. **76 tests pass.**

---

## 6. The task list — start here

Each task leaves the test suite green. Run `python -m pytest src_continuous_control/tests -q` from
the **parent** directory after every one.

### T1 — `schedule="step"` in the drift env  ⬅ **do this first**
The only new science; everything else is plumbing.
`envs/drift_half_cheetah.py` currently produces only continuous `sin` / `linear` multipliers.
Phase 2a needs the multiplier **held constant within a task and changed at the boundary**:
```yaml
schedule: step
task_multipliers: [1.0, 1.6, 0.6, 1.6, 0.6]     # cycles, so tasks 2/4 and 3/5 repeat
```
Add a branch in `multiplier()` and a `set_task(i)` that selects the entry. Keep `sin`/`linear`
intact — Phase 2b uses them.
**Add a test** asserting the physics differ between tasks and stay fixed within one. A silently
constant env looks exactly like a working experiment and yields three identical agents.

### T2 — rename `pt_full` → `pt`
- `agents/ppo_pt_full.py` → `agents/ppo_pt.py`, class `PPOPTFull` → `PPOPT`.
- **Archive the old critic-only agent first** (`agents/ppo_pt.py` → `archive/phase1/`).
- `agents/__init__.py`: register the renamed class as `"pt"`, remove `"pt_full"`.
- `configs/ppo_pt_full.yaml` → `configs/ppo_pt.yaml` (overwrites the old).
- Update the tests that import the old agent: `conftest.py`, `test_online_updates.py`,
  `test_optim_state_reset.py`, `test_paper_fidelity.py`, `test_pt_full_agent.py`,
  `test_task_switching.py`. Archive `tests/test_pt_consolidation.py` (it tests the old agent).

### T3 — strip the shrinkage control from `ppo_base.py`
Remove `policy_shrink_every`, `policy_shrink_factor`, `critic_shrink`, `shrink_flush_optim` and the
whole shrink block in `post_update` (~lines 63–70 and 148–170). `PPOVanilla.post_update` reverts to
a no-op. **Keep** `mu_l2_coef`, `log_std_init`, `freeze_log_std`.

### T4 — retire the point-mass benchmark
Remove the `point_tasks` / `point_drift` env modes from `train.py` and its imports (lines 34–35),
then archive `envs/mock_continual.py`, `envs/simple_drift.py`, `configs/mock_continual.yaml`,
`configs/simple_drift.yaml`, `tests/test_simple_drift.py`.

### T5 — route `set_task` to the drift wrapper
`train.py:_set_env_task` currently searches for `DirectionalHalfCheetah`. It must call the drift
wrapper's `set_task(i)` instead, so a boundary changes physics rather than reward sign.
`envs/directional_half_cheetah.py` stays (T-note above), but its `set_task` is no longer used.

### T6 — transfer matrix
`evaluate_transfer_matrix()` in `utils/metrics.py`: for each pair (i, j), mean return of the
policy after task *i* evaluated on task *j*'s physics — frozen policy, no exploration noise,
10 episodes per cell. Then BWT and FWT per §2.2. Add a test on a 2×2 case.

### T7 — configs
`configs/default.yaml`: `env_mode: drift`, `schedule: step`, `drift_targets: ["damping","friction"]`,
`task_multipliers`, 5 phases, 5 seeds.
New `configs/phase2_hard.yaml`: all four drift targets, for §2.4 run 2.
`configs/ppo_ewc.yaml`: unchanged for 2a.

### T8 — pre-flight, before any long run
Cheap checks that would each have saved days in Phase 1:
1. **Smoke test** — 3 agents × 1 seed × 60k steps. All three train; EWC's penalty is non-zero;
   `actor_absorbed_frac > 0.01` for `pt`; physics differ between tasks.
2. **Dynamic range** — one full-length vanilla run. **If its return varies by less than ~20% across
   the task sequence the benchmark cannot separate the methods** and the amplitude must go up.
   Do not skip this.
3. **Parameter parity** — print the parameter count of all three agents. Phase 1 found a published
   config giving PT 13.9× the baseline's parameters.
4. **σ parity** — assert the realised `log_std` is identical across agents before the sweep.

### T9 — the sweep
3 agents × 5 seeds, ≈4 h locally at 7-way parallelism. Pin threads
(`OMP_NUM_THREADS=1` etc.) or throughput collapses.

### T10 — figures
New `plots/make_phase2_figures.py`, reading raw per-seed pickles so nothing is transcribed:
**(a)** return vs timestep, boundaries marked · **(b)** FWT/BWT · **(c)** the three-bar ablation
(§2.3). Load the `dataviz` skill before writing chart code.

### T11 — the harder variant
Repeat T9/T10 with `phase2_hard.yaml`. This is the directional test of the paper's own
"big world – small agent" claim (§2.4).

---

## 7. Risks worth stating now

**Check the dynamic range before the full sweep.** Phase 1's drift benchmark was saturated — every
agent sat at 96–99% of the ceiling, and a "significant" result evaporated once the task was made
harder. **If vanilla's return varies less than ~20% across the task sequence, the benchmark cannot
separate the methods.** One run, and it would have saved a week last time.

**5 seeds is thin.** Exact two-sided Mann-Whitney with 5 vs 5 has a floor of p = 2/252 = **0.0079** —
significance is reachable, but only where the two groups barely overlap. Several Phase 1 effects
that landed near p ≈ 0.04 would be invisible at n = 5. If a headline lands near the boundary, add
seeds rather than report it.

**Dropping the shrink control removes our only explanation of the Phase 1 gain.** A reasonable
call — it is a heuristic without theory. But if `pt` beats `vanilla` in Phase 2, we cannot say
whether that is the decomposition or the decay unless something plays that role.
**`probe/decay_gain` (§2.3) is the cheap insurance** — inside the agent, no extra arm, no heuristic
added to vanilla. Strongly recommend keeping it wired.

**`pt` has never been tuned for dynamics change.** Every ρ and *k* in use was chosen against the
reward-switch benchmark. Budget a small sweep (ρ ∈ {0.25, 0.5, 0.75} at fixed *k*) before treating
any Phase 2 number as final.

**Physics change may be an easier problem than reward change.** A reward flip inverts the optimal
policy; a friction change perturbs it. If all three agents end up close together, that is a
benchmark-difficulty finding, not a null result about PT — and the hard variant (§2.4) is the
answer to it.

---

## 8. Open questions

1. Task sequence: confirm `[1.0, 1.6, 0.6, 1.6, 0.6]` over 5 phases, or give different
   multipliers / a different number of phases. **The sequence must revisit** or BWT is undefined.
2. Amplitude: is ±60% of nominal friction/damping the right size? This is the single knob that
   decides whether the benchmark can separate the methods.
3. Transfer matrix: 10 episodes per cell — enough, or more?
4. Does the thesis want the Phase 1 reduction result at all? It affects whether `FULL_PT.md` is
   archived-and-forgotten or archived-and-cited.
