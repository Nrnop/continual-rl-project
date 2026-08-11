# PT-full: Implementation, Results, and Next Experiments

## Executive summary

This repository contains a full permanent-transient PPO agent, `pt_full`, that decomposes both the
policy mean and value function into permanent and transient components. It was compared with
vanilla PPO on real MuJoCo `HalfCheetah-v5` under four discrete directional task switches.

The current 3-seed confirmation does **not** show a PT-full performance win. PT has a modestly
smaller measured boundary drop, but vanilla PPO has much higher return and much stronger early
post-switch recovery. The PT mechanism is active: every PT run performed 94 consolidation cycles
and transferred a large fraction of the transient component. Activation of the mechanism alone
did not produce a control benefit on this benchmark.

This conclusion is specific to the tested directional task and three seeds. The repository also
contains smooth, boundary-free drift environments and proxy sweep tools for the next stage. Smooth
drift results from older experiments must be treated as historical until rerun with the current
`pt_full` architecture; the older drift configs explicitly warn that some prior tables used a
removed shared-trunk PT variant.

## What was added

### PT-full agent

`agents/ppo_pt_full.py` implements the full actor-plus-critic decomposition:

- `SplitGaussianActor`: permanent and transient policy means with a shared frozen `log_std`.
- `SplitCritic`: independent permanent and transient value trunks with
  `V(s) = V_perm(s) + V_trans(s)`.
- Online PPO updates train the transient components while the permanent components are protected
  from ordinary PPO updates.
- Every `k` updates, a state-only consolidation buffer regresses the permanent components toward
  the old permanent output plus `rho` times the old transient output.
- The transient value output is decayed in output space, preserving the intended transfer split;
  transient optimizer state is flushed after consolidation.
- Permanent learning rates use Robbins-Monro-style annealing over consolidation events.
- Actor consolidation, critic consolidation, KL-to-permanent regularization, and detailed
  absorption/alignment diagnostics are recorded.

`configs/ppo_pt_full.yaml` contains the agent defaults. The final confirmation overrides were:

| Setting | PT-A | PT-B |
|---|---:|---:|
| critic `lr_perm` | `1e-4` | `3e-4` |
| actor `lr_perm_actor` | `3e-4` | `3e-4` |
| `rho` | `0.5` | `0.5` |
| `k` | `16` | `16` |
| `rm_power` | `0.6` | `0.6` |
| `kl_prior_coef` | `0.01` | `0.01` |

### Training and measurement corrections

The tracked changes also make the comparison protocol reproducible and measurable:

- frozen observation/reward normalizer statistics after warmup;
- training normalizer statistics copied to offline evaluation;
- isolated evaluation RNG so evaluation cannot alter later training trajectories;
- vectorized rollout buffers storing permanent and transient value estimates separately;
- corrected GAE using the sum of the two value components;
- boundary windows measured in whole vectorized PPO updates;
- jumpstart measurements in the first 20 updates after a switch;
- retention MSE with `perm_init` and zero controls to detect inert permanents;
- consolidation loss traces and per-cycle absorption records;
- unit tests for normalizer freezing, split actor behavior, consolidation, and smooth point drift.

## Hyperparameter tuning already completed

The proxy search was staged because a single HalfCheetah seed is noisy:

1. `scripts/hp_sweep_expanded.py` searched 81 combinations over `lr_perm`, `rho`, KL
   coefficient, and consolidation cadence, then confirmed the top configurations on multiple
   seeds.
2. `scripts/hp_focused_sweep.py` tested six configurations: `lr_perm` in
   `{1e-4, 2e-4, 3e-4}` and `k` in `{8, 16}`, with `rho=0.5`, using five seeds.
3. The proxy results narrowed the real confirmation to the two candidates above.

The proxy ranking was high variance. It was useful for narrowing the search, not for claiming a
winner. The real study changed only the critic permanent learning rate between PT-A and PT-B, so
it is a confirmation of two candidates, not an exhaustive real-environment search.

Reproduce the proxy stages with:

```bash
PYTHONPATH=. .venv/bin/python scripts/hp_sweep_expanded.py \
  --grid-updates 120 --multi-updates 300 --top-k 5

PYTHONPATH=. .venv/bin/python scripts/hp_focused_sweep.py \
  --total-updates 300 --num-seeds 5
```

The raw proxy outputs are written under the ignored `results/` directory.

## Real HalfCheetah confirmation

All nine runs used:

- MuJoCo `HalfCheetah-v5`;
- eight asynchronous vector environments and 256 rollout steps per environment;
- 3,072,000 aggregate environment steps;
- four switches at steps 614,400, 1,228,800, 1,843,200, and 2,457,600;
- task sequence `+1 -> -1 -> +1 -> -1 -> +1`;
- matched normalization, PPO, evaluation, and seeds across methods.

The reward changes from forward to backward velocity. The physics, observation space, action space,
and policy input remain the same. This is genuine continual task switching, but it is not smooth
physics drift.

### Results

Values are means plus sample standard deviation across three seeds. `final return` is the mean EMA
return over the last 100 PPO updates, not the entire fifth phase.

| Method | Final return | Boundary drop | 20-update jumpstart | Retention MSE, permanent | Retention MSE, full |
|---|---:|---:|---:|---:|---:|
| PT-A, `lr_perm=1e-4` | `-490.8 +/- 39.8` | `234.5 +/- 9.8` | `14.8 +/- 68.4` | `41.84 +/- 11.74` | `42.08 +/- 11.51` |
| PT-B, `lr_perm=3e-4` | `-407.9 +/- 255.2` | `257.8 +/- 37.8` | `78.3 +/- 99.5` | `22.79 +/- 10.30` | `22.90 +/- 10.49` |
| Vanilla PPO | `832.6 +/- 234.0` | `291.9 +/- 65.2` | `1088.8 +/- 334.6` | `3.02 +/- 2.02` | `3.02 +/- 2.02` |

The method-level reading is:

- PT-A has the smallest boundary-drop mean, but the advantage is local and does not translate into
  higher return.
- Neither PT candidate has a jumpstart advantage. Vanilla recovers more strongly in the measured
  post-switch window.
- PT-A and PT-B are not inert: mean critic absorption was approximately `0.864` and `0.967`, with
  94 consolidation cycles per run.
- The retention values do not support a useful PT retention claim here. The permanent component
  does not beat the initial/zero controls consistently, and the symmetric reward flip gives the
  permanent component a weak task-average target.

The final artifacts are generated by `plots/analyze_final_comparison.py` and live under
`plots/figures/final_comparison/` after running the analysis. The per-seed table is
`run_metrics.csv`; the complete raw result pickles remain in the ignored `results/` tree.

## Smooth continual environments

Two smooth, boundary-free environments are ready.

### CPU point-mass drift

`envs/simple_drift.py` implements `DriftingPointMass`. The reward stays fixed while an unobserved
drag coefficient follows a bounded sinusoidal schedule. Episode resets do not reset the global drift
clock. `configs/simple_drift.yaml`, `plots/plot_simple_drift.py`, and
`tests/test_simple_drift.py` provide the runnable demo and checks.

### MuJoCo HalfCheetah physics drift

`envs/drift_half_cheetah.py` implements `LipschitzDriftHalfCheetah`. Damping and friction are
rescaled smoothly from their nominal values, the reward is fixed, and there is no task index or
boundary callback. The repository provides three regimes:

- `configs/drift.yaml`: slow single-timescale drift;
- `configs/drift_fast.yaml`: ten-times-faster single-timescale drift;
- `configs/drift_twoscale.yaml`: a slow trend plus a fast fluctuation.

`tests/test_drift_env.py` checks smoothness, boundedness, clock behavior, non-compounding physics
updates, and reward preservation.

The existing `FINDINGS.md` contains historical smooth-drift results, but the current drift configs
warn that those tables used a removed shared-trunk PT variant. They should not be presented as final
evidence for the current `pt_full` agent. The next run must use the current agent and explicit
overrides, for example:

```bash
# Vanilla reference
python -m src_continuous_control.train --agent vanilla --config drift --seed 0 \
  --no-wandb --no-tb --results-dir results/drift_vanilla

# Current PT-full settings, with no boundary signal
python -m src_continuous_control.train --agent pt_full --config drift --seed 0 \
  --lr-perm 0.0003 --lr-perm-actor 0.0003 --perm-optimizer adam \
  --rho 0.5 --k 16 --consolidation-epochs 3 \
  --no-wandb --no-tb --results-dir results/drift_pt_full
```

Repeat those commands for `drift_fast` and `drift_twoscale`, and use matched seeds. The important
comparison is continuous tracking error and return by drift segment, not boundary drop.

## Recommended next experiment program

Further work should be structured, not an open-ended search for a favorable seed.

### 1. Current-agent smooth-drift rerun

Run vanilla, PT-full, and optionally EWC on `drift`, `drift_fast`, and `drift_twoscale` with the
same seeds and 5 seeds initially. Use the point-mass drift as a cheap smoke test before launching
MuJoCo. Report return by segment, drift multiplier, value/policy tracking diagnostics, and PT
absorption. Increase to 10 seeds before making a fine-grained claim.

### 2. Asymmetric discrete tasks

The symmetric `+1/-1` reward flip cancels the task-discriminative term in the task average. Run the
existing asymmetric task configuration (`configs/pt_paper_asym.yaml`, for example tasks
`[1.0, -0.5]`) with current `pt_full` and vanilla. This is the cleanest test of whether the
permanent component helps when its theoretical average target contains task information.

### 3. Mechanism-aware tuning

Use a pre-registered staged search:

- stationary no-switch gate: PT-full must be comparable to vanilla before continual claims;
- mechanism gate: absorption and alignment must show that the permanent actually learns;
- performance gate: compare final return, jumpstart, boundary-free tracking, and retention against
  initial/zero controls;
- final confirmation: at least 10 seeds for close PT-versus-vanilla effects.

A practical next grid is `lr_perm` and `lr_perm_actor` on a logarithmic scale, `rho` in
`{0.25, 0.5, 0.75, 1.0}`, `k` in `{4, 8, 16, 32}`, and KL coefficient in
`{0, 1e-3, 1e-2}`. Search the cheap point-mass first, then promote only a few mechanism-valid
configurations to each smooth HalfCheetah regime.

### 4. Other smooth environments

The point-mass and HalfCheetah drift environments are ready now. Walker2d, Hopper, or another
MuJoCo task would require a generic drift wrapper plus task-specific validation of model fields,
reward info, and episode behavior before it is a fair comparison. Do not treat merely changing
`env_id` as a valid second environment without those checks.

## Validation and cleanup

The retained tests cover the PT-full construction/update/consolidation path, split actor behavior,
normalizer freezing, buffer invariants, simple smooth drift, and plotting. Run them from the
repository root with:

```bash
.venv/bin/python -m pytest tests -q
```

Generated raw results stay under ignored `results/` paths. The bundled paper PDF, interim plots,
and the superseded small sweep driver were removed from the worktree; reproducible scripts and
final analysis outputs are the files that remain.
