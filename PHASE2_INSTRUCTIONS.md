# Phase 2 — instructions

Decisions from the supervisor meeting, turned into a work plan, with the follow-up clarifications
folded in. This supersedes the experimental design in `FULL_PT.md`; that document becomes part of
the Phase 1 archive.

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
| **Parameters** | Start with `["damping", "friction"]`. Then a harder variant changing **all** of them (§5). |
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

## 5. Files to ARCHIVE — for your approval

Nothing deleted. Proposal: move to `archive/` in one commit, so history is preserved and Phase 1
stays reproducible by checking out the previous tag.

### 5a. Old PT implementation
```
agents/ppo_pt.py                      the critic-only PT agent
configs/ppo_pt.yaml  pt_fixed.yaml  pt_paper.yaml  pt_paper_asym.yaml
configs/pt_slow.yaml  pt_klambda.yaml  pt_zeroperm.yaml  vanilla_paper.yaml
tests/test_pt_consolidation.py
plots/make_consolidation_figure.py  plot_consolidation_insitu.py  plot_consolidation_internals.py
```

### 5b. Point-mass benchmark
```
envs/mock_continual.py  envs/simple_drift.py
configs/mock_continual.yaml  simple_drift.yaml
tests/test_simple_drift.py  test_simple_drift_plot.py
plots/plot_mock_continual.py  plot_simple_drift.py
```

### 5c. Reward-switch benchmark
```
envs/directional_half_cheetah.py      the +1/−1 reward flip
tests/test_task_switching.py
```
⚠️ **Check first:** `train.py:29` imports it and `_set_env_task` traverses for it. Must stay until
`set_task` is re-routed to the drift wrapper (§4).

### 5d. Phase 1 reduction study
```
configs/stage*.yaml                                       130 files
scripts/run_stage1.sh  run_stage2.sh  run_stage3.sh  run_stages_4_7.sh
scripts/run_stage12_halfcheetah.sh  run_stage14_halfcheetah.sh  run_on_vastai.sh
scripts/analyze_all_stages.py  analyze_stage1.py  gen_stage1_configs.py
plots/make_pt_full_figures.py
```
**Keep `plots/figures_pt_full/`** — the figures are committed and are the evidence for the Phase 1
result. Archiving the generator means they can no longer be regenerated, which is acceptable only
because the images themselves are in git.

### 5e. Older ablation rounds
```
configs/abl_*.yaml                                        23 files
scripts/run_ablation.sh … run_ablation7.sh                7 files
scripts/hp_focused_sweep.py  hp_sweep_expanded.py
scripts/run_singletask_baseline.sh  run_vanilla_5seeds.py  run_all_5seeds_eval.py
plots/make_reinvestigation_figures.py  plot_drift_and_r7.py  analyze_final_comparison.py
plots/plot_singletask_live.py  plot_thesis_figures.py
configs/cleanrl_match.yaml  continual_fast.yaml  drift_fast.yaml  drift_twoscale.yaml
```

### 5f. Results / report documents — **all of them**
```
FULL_PT.md                  Phase 1 record, §1–§27
figures_full_pt_guide.md    the figure walkthrough
MEETING_BRIEF.md            (untracked) plain-language brief
SESSION_LOG.md              (untracked) handoff log
FINDINGS.md                 pre-reduction results, several since retracted
REINVESTIGATION.md          the investigation preceding Phase 1
PT_REFERENCE_MAPPING.md     mapping to the old critic-only implementation
VASTAI_SETUP.md             remote-box notes, no longer needed
```

**Recommend keeping two out of the archive:**
- **`PT_full.md`** — the specification. Not a result; still the reference for what `pt` must do.
- **`README.md`, `CLAUDE.md`** — to be rewritten, not archived.

⚠️ **Before archiving `FULL_PT.md`:** it is the only record of the Phase 1 negative result, and
that result is still likely to appear in the thesis (the reduction, the benchmark critique, the
actor-critic cancellation). Archiving is fine; **losing track of it is not.** Suggest
`archive/phase1/` with a one-page `archive/README.md` saying what is in there and why.

### Summary

| group | files |
|---|---:|
| 5a old PT | 13 |
| 5b point-mass | 8 |
| 5c reward switch | 2 |
| 5d Phase 1 study | ~140 |
| 5e old ablations | ~40 |
| 5f documents | 8 |
| **total** | **~211 of 355 tracked** |

Leaves ~44 source files.

---

## 6. Order of work

Each step leaves the test suite green.

1. **Tag `main` as `phase1-final`.** Everything below is irreversible without it.
2. **Branch `phase2`.**
3. **`schedule="step"`** in the drift env, plus its test (§3). Do this *first* — it is the only new
   science, and everything else is plumbing.
4. **Rename** `pt_full` → `pt`; delete the old agent; update the registry and three PT tests.
5. **Strip** the shrink code from `ppo_base.py`.
6. **Re-route `set_task`** in `train.py` to the drift wrapper; then `directional_half_cheetah.py`
   is safe to archive.
7. **Archive** the approved lists in one commit, with `archive/README.md`.
8. **Rewrite** `CLAUDE.md`, `README.md`, `configs/default.yaml`.
9. **Transfer-matrix eval** + test.
10. **Smoke test** — 3 agents × 1 seed × 60k steps. Confirm: all three train; EWC's penalty is
    non-zero; `absorbed_frac > 0.01` for `pt`; the physics actually differ between tasks.
11. **Dynamic-range check** (§7) — one vanilla run, full length.
12. **Full sweep** — 3 × 5 seeds, ≈4 h.
13. **Figures** (a), (b), (c).
14. **Hard variant** (§2.4 run 2), same protocol.

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
