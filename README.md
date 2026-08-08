# src_continuous_control

A **continuous-control extension of the Permanent–Transient (PT) value decomposition** from
Anand & Precup (2023), *Prediction and Control in Continual Reinforcement Learning*, applied to a
**PPO actor-critic on MuJoCo HalfCheetah**. Thesis codebase (BIHE, B.Eng.):
*Permanent and Transient Representations for Continual RL under Smooth Environment Non-Stationarity.*

## What this is

We port the value-based PT idea `V = V_perm + V_trans` into continuous-control PPO and study it
under continual (non-stationary) conditions:

- **Legacy `pt` is applied to the critic** (the value function), not the policy:

  ```
  V(s) = V_perm(s; θ_P) + V_trans(s; θ_T)
  ```

  - **Transient critic `θ_T`** — fast (`lr_trans`, Adam): trained every PPO update to fit the
    residual `returns − V_perm.detach()` above the frozen permanent baseline.
  - **Permanent critic `θ_P`** — slow (`lr_perm`, SGD): *not* trained on returns each step. Every
    `k` updates it **consolidates**, regressing `V_perm → old_V_perm + (1−decay)·V_trans.detach()`
    over a buffer of visited states, then the transient is decayed (`θ_T ← decay·θ_T`). This target
    is **value-preserving for any `decay`**: the acting value `V` doesn't drift across consolidation.
  - At a task boundary the agent **consolidates first, then decays** — locking the just-learned task
    value into `θ_P` so `V` doesn't lurch at the switch.

- **Non-stationarity** comes from `DirectionalHalfCheetah`: the forward-velocity reward term flips
  sign at fixed step boundaries — `task = +1` rewards running **forward**, `task = −1` rewards
  **backward**. Control cost stays task-invariant (shared physics); only the direction flips.

- **CPU-friendly `point_drift` benchmark**: `DriftingPointMass` controls a one-dimensional point
  mass toward the origin while an unobserved drag coefficient follows a smooth cyclic schedule.
  The reward is fixed and quadratic, there are no task boundaries, and the two-dimensional
  observation is just position and velocity. It is intended for fast end-to-end debugging before
  spending compute on MuJoCo.

- **Comparison agents** (same environment and seeds; `pt_full` also tests policy transfer):

  | Agent | What it is |
  |-------|-----------|
  | `vanilla` | Standard single-critic PPO — the baseline |
  | `pt` | Dual-timescale split critic + consolidation — **the contribution** |
  | `ewc` | Online Elastic Weight Consolidation (diagonal Fisher penalty) — regularization baseline |
  | `pt_full` | Dual-timescale split actor mean + critic with KL prior and rho consolidation |

> **Scope note.** Both regimes are implemented: `DirectionalHalfCheetah` provides discrete reward
> task switches, while `LipschitzDriftHalfCheetah` and `DriftingPointMass` provide boundary-free
> smooth dynamics drift. The current PT-full confirmation table is the discrete HalfCheetah study;
> the smooth-drift configurations require a fresh run with the current PT-full agent before their
> results can be used as final evidence. See [PT_full.md](PT_full.md).

## Install

Use an isolated virtualenv with a **modern** gymnasium (HalfCheetah-v5 needs ≥ 0.29):

```bash
python -m venv .venv_cc
source .venv_cc/Scripts/activate      # Windows (Git Bash);  Linux/Mac: .venv_cc/bin/activate
pip install -r src_continuous_control/requirements_continuous.txt
```

The `mujoco` pip wheel runs natively on Windows (no `mujoco-py`).

## Run

> **Run from the *parent* directory** (`…/` that contains `src_continuous_control/`). The default
> `results_dir`/`runs_dir` are relative, so running from *inside* the package creates a stray nested
> `src_continuous_control/` folder.

```bash
cd ..    # into the folder that contains src_continuous_control/

# Baseline / contribution / regularization baseline
python -m src_continuous_control.train --agent vanilla --config continual_fast --seed 0
python -m src_continuous_control.train --agent pt      --config continual_fast --seed 0
python -m src_continuous_control.train --agent ewc     --config continual_fast --seed 0
python -m src_continuous_control.train --agent pt_full --config continual_fast --seed 0

# Simple CPU drift demo (run both arms, then plot the learning diagnostics)
python -m src_continuous_control.train --agent vanilla --config simple_drift --seed 0 \
  --results-dir results/simple_drift_demo --runs-dir runs/simple_drift_demo --no-wandb --no-tb
python -m src_continuous_control.train --agent pt_full --config simple_drift --seed 0 \
  --results-dir results/simple_drift_demo --runs-dir runs/simple_drift_demo --no-wandb --no-tb
python -m src_continuous_control.plots.plot_simple_drift \
  --results-dir results/simple_drift_demo --agents vanilla pt_full --seeds 0

# Smooth HalfCheetah physics drift (fixed reward, no task boundaries)
python -m src_continuous_control.train --agent vanilla --config drift --seed 0 \
  --no-wandb --no-tb --results-dir results/drift_vanilla
python -m src_continuous_control.train --agent pt_full --config drift --seed 0 \
  --lr-perm 0.0003 --lr-perm-actor 0.0003 --perm-optimizer adam \
  --rho 0.5 --k 16 --consolidation-epochs 3 \
  --no-wandb --no-tb --results-dir results/drift_pt_full

# Single-task sanity check (no switching) — does PPO reach a healthy positive HalfCheetah return?
python -m src_continuous_control.train --agent pt --config pt_fixed \
    --disable-task-switch --total-steps 1000000 --no-wandb --no-tb

# Quick smoke test (tiny run, no logging backends)
python -m src_continuous_control.train --agent pt --seed 0 \
    --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb --no-tb
```

Per-seed curves are written to `results/*.pkl`; TensorBoard events to `runs/`; W&B (unless
`--no-wandb`) to project `pt-continuous-control`. Multi-seed sweeps live in `scripts/`.

### Configuration

Layered — later overrides earlier: `configs/default.yaml` ← `configs/ppo_<agent>.yaml`
← `--config <overlay>.yaml` ← CLI flags. Key overlays:

| Overlay | Purpose |
|---------|---------|
| `cleanrl_match` | Validated single-env PPO recipe (obs/reward normalization, LR anneal, Adam eps 1e-5) |
| `continual_fast` | The full continual experiment, 8× async vector envs for throughput |
| `ppo_pt_full` | Actor+critic PT-PPO defaults; sweep rho, KL coefficient, k, and permanent LRs |
| `simple_drift` | Tiny CPU point-mass benchmark with cyclic hidden drag drift |
| `drift`, `drift_fast`, `drift_twoscale` | HalfCheetah damping/friction drift at slow, fast, and two-timescale rates |
| `pt_fixed` | Corrected PT on the validated base |

The CleanRL-style **observation and reward normalization** (running mean/std, clipped to ±10) is
what makes HalfCheetah reach positive return; without it PPO returns stay negative.

## Compare & plot

```bash
python -m src_continuous_control.plots.plot_compare --seeds 0 1 2 3 4
```

Produces overlaid mean ± CI return curves with task-boundary markers, plus boundary-drop and
recovery-time charts, under `plots/figures/`.

## Layout

| Path | Purpose |
|------|---------|
| `envs/directional_half_cheetah.py` | HalfCheetah wrapper; `set_task(±1)`; vectorized + normalized env factories |
| `envs/drift_half_cheetah.py` | Boundary-free smooth HalfCheetah physics drift |
| `models/actor.py` | `GaussianActor` and `SplitGaussianActor` (separate permanent/transient means) |
| `envs/simple_drift.py` | CPU-friendly drifting point-mass environment and vector factory |
| `models/critic.py` | `VanillaCritic` and `SplitCritic` (`V_perm + V_trans`, two separate trunks) |
| `agents/ppo_base.py` | Shared PPO core: vectorized rollout, GAE-λ, clipped-surrogate update, LR anneal |
| `agents/ppo_vanilla.py` | Single-critic baseline |
| `agents/ppo_pt.py` | Dual-timescale split critic + value-preserving consolidation + decay |
| `agents/ppo_pt_full.py` | Dual-timescale split actor mean + critic, KL prior, and rho consolidation |
| `agents/ppo_ewc.py` | Online EWC (diagonal Fisher penalty on the actor) |
| `utils/` | seeding, buffers (rollout + state-only consolidation), unified logger (TB+W&B+pickle), boundary metrics |
| `plots/` | `plot_compare` (comparative charts), `plot_singletask_live` |
| `scripts/` | Multi-seed / full-scale run scripts |
| `train.py` | Entry point (config resolution + training loop) |
| `plots/plot_simple_drift.py` | Learning curve, drift-clock, energy-balance, and consolidation plots |
| `plots/analyze_final_comparison.py` | Reproducible PT-full versus vanilla result tables and figures |
| `PT_full.md` | Implementation, experiment results, limitations, and next-run protocol |

## Testing

```bash
python -m pytest tests -q
```

Fast, CPU-only unit tests (buffers, agent updates, task switching, logging/plotting) — no MuJoCo
training required.
