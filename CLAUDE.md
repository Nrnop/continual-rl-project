# CLAUDE.md — src_continuous_control

Guidance for Claude Code when working in this repo. Read this first.

## What this project is

A **continuous-control extension of the Permanent–Transient (PT) value decomposition**
(Anand & Precup 2023, *Prediction and Control in Continual Reinforcement Learning*) to a
**PPO actor-critic on MuJoCo HalfCheetah**. It is the codebase behind Nura Nabipour's BIHE
bachelor's thesis, *Permanent and Transient Representations for Continual RL under Smooth
Environment Non-Stationarity*.

**Proposal vs. code — read this before trusting the proposal.** The written proposal
(`NuraNabipour-Revised Proposal.pdf`) describes *smooth, Lipschitz-continuous drift of
environment dynamics* (friction/mass drifting every step). The **code does not do that yet.**
What is actually implemented is **discrete task-switching**: `DirectionalHalfCheetah` flips
the sign of the forward-velocity reward term (+1 = run forward, −1 = run backward) at fixed
step boundaries (`--switch`). This mirrors the reward-sign flip in the reference `PT_DQN_half.py`.
So the current experiment is *boundary-based directional non-stationarity*, not smooth dynamics
drift. Treat the proposal as direction-of-travel, not spec — the thesis is not final and scope
may change. When something in the proposal and the code disagree, **the code is the ground truth.**

**Current plan (decided):** stay with discrete directional task-switching for now and get good
results with it first. Smooth Lipschitz dynamics drift (a per-step friction/mass drift wrapper)
is a *later* addition, only once the PT mechanism is clearly working — not a current task.

## How to run

**Run from the PARENT directory** `e:/update-single task + videos/`, invoking the package by
its module path. The default `results_dir`/`runs_dir` in `configs/default.yaml` are relative
(`src_continuous_control/results`, `src_continuous_control/runs`); running from *inside* this
folder makes those paths create a **nested `src_continuous_control/` folder** — that is exactly
what the stray nested folder was. Always `cd` to the parent first.

```bash
cd "e:/update-single task + videos"

# Vanilla single-critic PPO baseline
python -m src_continuous_control.train --agent vanilla --seed 0

# Dual-timescale PT-PPO (the contribution)
python -m src_continuous_control.train --agent pt --seed 0

# EWC baseline
python -m src_continuous_control.train --agent ewc --seed 0

# Quick smoke test (tiny run, no logging backends)
python -m src_continuous_control.train --agent pt --seed 0 \
    --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb --no-tb
```

Multi-seed sweeps live in `scripts/` (`run_all.sh`, `run_full_scale_parallel.sh`, etc.).
Comparison plots: `python -m src_continuous_control.plots.plot_compare --seeds 0 1 2 3 4`.

### Environment
- Isolated venv, **gymnasium ≥ 0.29** + **mujoco ≥ 3.1** pip wheel (native on Windows, no
  `mujoco-py`). Deps in `requirements_continuous.txt`. Do **not** reuse any baseline venv pinned
  to gymnasium 0.28.1 — HalfCheetah-v5 needs the newer API.
- CUDA auto-detected; falls back to CPU.

## Configuration system

Layered, later overrides earlier:
`configs/default.yaml` ← `configs/ppo_<agent>.yaml` ← `--config <overlay>.yaml` ← CLI flags.

- **`default.yaml`** — shared PPO + env defaults.
- **`ppo_vanilla.yaml` / `ppo_pt.yaml` / `ppo_ewc.yaml`** — per-agent knobs (loaded by agent name).
- **Overlays** (`--config`): `cleanrl_match` (validated single-env PPO recipe),
  `continual_fast` (8× async vector envs for throughput), `pt_fixed` (corrected PT on the
  validated base). Overlays load *after* the agent YAML so agent-specific keys survive.

## Architecture

| Path | Purpose |
|------|---------|
| `train.py` / `__main__.py` | Entry point: config merge, vectorized env, train loop, eval, logging |
| `envs/directional_half_cheetah.py` | HalfCheetah wrapper; `set_task(±1)` flips reward direction; `make_vector_env` |
| `models/actor.py` | `GaussianActor` (used by ALL agents). `SplitActor` also exists but is unused — PT is critic-only |
| `models/critic.py` | `VanillaCritic` and `SplitCritic` (V = V_perm + V_trans, two separate trunks) |
| `agents/ppo_base.py` | Abstract PPO core: rollout, GAE-λ, clipped-surrogate update, LR anneal |
| `agents/ppo_vanilla.py` | Single-critic baseline |
| `agents/ppo_pt.py` | Dual-timescale split critic + periodic consolidation + transient decay |
| `agents/ppo_ewc.py` | Online EWC (diagonal Fisher penalty on the actor) |
| `utils/` | `seeding`, `buffers` (RolloutBuffer + ConsolidationBuffer), `logger` (TB+W&B+pickle), `metrics` |
| `plots/` | `plot_compare` (mean±CI return curves, boundary-drop bars), `plot_singletask_live` |
| `scripts/` | Multi-seed / full-scale shell + python runners |
| `tests/` | pytest suite (buffers, online updates, task switching, logging/plotting) |

Agents register in `agents/__init__.py` (`AGENTS = {"vanilla", "pt", "ewc"}`). All three share
the `PPOBase` loop; each implements `get_value`, `critic_loss`, `post_update`, and the critic
optimizer plumbing hooks.

### The PT mechanism (the actual contribution)
- **Transient critic `θ_T`** (fast, `lr_trans`, Adam): every PPO update, regresses
  `V_perm.detach() + V_trans → returns` — learns the residual above a *frozen* permanent baseline.
- **Permanent critic `θ_P`** (slow, `lr_perm`, SGD): **not** trained on returns each step.
  Every `k` updates it *consolidates* by regressing `V_perm → old_V_perm + (1-decay)·V_trans.detach()`
  over a rolling `ConsolidationBuffer` of visited states (mirrors `train_P_Net`), then the
  transient head is **decayed** (`θ_T ← decay · θ_T`). This target is **value-preserving for any
  decay**: `V = V_perm + V_trans` is unchanged across consolidation (no drift). `decay=0` = hard reset.
- **PT is critic-only** — the actor is the same single `GaussianActor` as vanilla/EWC.
- At a task boundary (`consolidate_on_switch`, default on) it **consolidates first, then decays** —
  locking the just-learned task value into `θ_P` so the acting value doesn't lurch at the boundary.
- **Why θ_P is consolidation-only:** regressing `V_perm` directly on returns every step
  double-counts against the acting value `V = V_perm + V_trans` and (no target net, no value
  normalization) caused divergence (`critic_loss ~2e5`). `pt_fixed.yaml` is the corrected recipe.

### Metrics that matter (supervisor's framing)
The thesis question is *how violently the value function lurches at a task switch and how much
return collapses right after*. `utils/metrics.py` provides `ValueDriftProbe` (mean `|ΔV|` on
fixed probe states across a boundary) and `BoundaryReturnTracker` (pre-switch plateau minus
post-switch trough). PT should spike less than vanilla. There is also a zero-momentum offline
eval from standstill (`_run_offline_eval` in `train.py`).

## Testing

```bash
cd "e:/update-single task + videos"
python -m pytest src_continuous_control/tests -q
```

## Repo conventions & gotchas

- **Training artifacts are NOT tracked** and are regenerable: `checkpoints/`, `runs/`,
  `runs_singletask/`, `results/`, `results_singletask/`, `numeric_logs_csv/`, `*.pt`, `*.pkl`,
  `wandb/`. Only source, configs, `README.md`, `requirements`, and `plots/` figures are committed.
  See `.gitignore`. (These were force-pushed once by mistake and later purged from history.)
- **Don't run training from inside this folder** — see the nested-folder gotcha above.
- Env wrappers are traversed defensively in `train.py` (`_set_env_task`, `_find_normalize_obs`)
  because `RecordVideo` / vector wrappers hide the `DirectionalHalfCheetah`. Reuse those helpers
  rather than reaching for `.env` directly.
- Windows + `AsyncVectorEnv` needs picklable env factories — that's why
  `_make_single_directional` is module-level, not a closure. Keep it that way.
- True (un-normalized) episodic return is read from `RecordEpisodeStatistics` *before* reward
  normalization and via `info["directional_reward"]`; keep return logging honest this way.

## Working style here

- This is a research/thesis codebase — favor small, legible, well-commented changes that match
  the existing heavy-comment style. Preserve the vanilla/PT/EWC apples-to-apples symmetry.
- Don't commit or push unless asked. Artifacts stay out of git.
