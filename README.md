# src_continuous_control

Continuous-control extension of the Permanent–Transient (PT) decomposition from
Anand & Precup (2023), *Prediction and Control in Continual Reinforcement Learning*.

This module is **fully isolated**: it does not modify, import, or depend on any file in the
baseline repository. The baseline (`control/`, `prediction_semi_crl/`) is preserved untouched as
the reference implementation. We only *copied patterns* from it — chiefly the `train_T_Net` /
`train_P_Net` update equations in `control/minatar_crl/PT_DQN_half.py`.

## What this is

We port the value-based PT decomposition `Q = Q_perm + Q_trans` into a **continuous-control
actor-critic (PPO)** on a MuJoCo **DirectionalHalfCheetah** task (forward ↔ backward = reward-sign
flip at fixed boundaries). The critic is split into a **dual-timescale state value**:

```
V(s) = V_perm(s; θ_P) + V_trans(s; θ_T)
```

- **Transient critic** `θ_T`: updated aggressively every PPO update, learning rate `α_T`.
- **Permanent critic** `θ_P`: consolidated conservatively every `k` updates (and at task
  boundaries), learning rate `α_P ≪ α_T` (SGD). It regresses `V_perm → V_trans.detach() + old_V_perm`,
  mirroring the paper's `train_P_Net`.
- After each consolidation the transient critic is **decayed** (`θ_T ← decay · θ_T`; `decay=0` ≈ reset).

The goal (supervisor feedback): keep the combined `V` from lurching at task switches — `V_perm`
anchors the shared locomotion physics, `V_trans` absorbs the directional shift — minimizing the
performance drop at the boundary while raising the overall return.

We benchmark the PT agent against a **vanilla single-critic PPO on the same environment and seeds**
(the same apples-to-apples comparison the paper uses for PT_DQN vs DQN).

## Install (use an isolated venv — do not reuse the baseline's gymnasium 0.28.1)

```bash
python -m venv .venv_cc
# Windows (Git Bash):
source .venv_cc/Scripts/activate
# Linux/Mac:
# source .venv_cc/bin/activate

pip install -r src_continuous_control/requirements_continuous.txt
```

The `mujoco` pip wheel runs natively on Windows (no `mujoco-py` needed).

## Run

```bash
# Vanilla single-critic PPO baseline
python -m src_continuous_control.train --agent vanilla --seed 0

# Dual-timescale PT-PPO (the contribution)
python -m src_continuous_control.train --agent pt --seed 0

# Quick smoke test (tiny run, frequent switch)
python -m src_continuous_control.train --agent pt --seed 0 \
    --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb
```

Per-seed return curves are written to `src_continuous_control/results/*.pkl`; TensorBoard event
files to `src_continuous_control/runs/`; W&B (if enabled) to project `pt-continuous-control`.

## Compare & plot

```bash
# After running both agents over several seeds:
python -m src_continuous_control.plots.plot_compare --seeds 0 1 2 3 4
```

Produces an overlaid mean ± CI return curve with task-boundary markers, plus a boundary-drop bar
chart, under `src_continuous_control/plots/figures/`.

## Layout

| Path | Purpose |
|------|---------|
| `envs/directional_half_cheetah.py` | HalfCheetah wrapper; `set_task(±1)`; directional reward |
| `models/actor.py` | Gaussian MLP policy (continuous actions) |
| `models/critic.py` | `SplitCritic` (V_perm + V_trans) and `VanillaCritic` |
| `agents/ppo_base.py` | Shared PPO core (rollout, GAE-λ, clipped actor loss) |
| `agents/ppo_vanilla.py` | Single-critic baseline |
| `agents/ppo_pt.py` | Dual-timescale split critic + consolidation + decay |
| `utils/` | seeding, buffers, unified logger (TB+W&B+pickle), metrics |
| `train.py` | Entry point (config + CLI) |
| `plots/plot_compare.py` | Comparative charts |
| `scripts/` | Multi-seed run shell scripts |
