---
name: run-experiment
description: Launch a training run for this continuous-control PT project (vanilla/pt/ewc PPO on DirectionalHalfCheetah). Use whenever asked to run, train, validate, smoke-test, or reproduce an experiment. Encodes the correct invocation, config overlays, and the run-from-parent-directory gotcha.
---

# Running an experiment

Training entry point is the `src_continuous_control` package (`train.py`). Agents: `vanilla`
(baseline), `pt` (the contribution), `ewc` (regularization baseline).

## Golden rules

1. **Run from the PARENT directory** — the one that *contains* `src_continuous_control/`
   (Windows: `e:/update-single task + videos`; Linux/vast.ai: the git clone dir, e.g.
   `continual-rl-project`). The default `results_dir`/`runs_dir` are relative; running from inside
   the package writes a stray nested `src_continuous_control/` folder. Always `cd` to the parent first.
2. **Never run training from inside the package folder.**
3. Long runs: launch with `run_in_background: true` (Bash tool) and report the run id, or wrap in a
   `timeout` for a bounded smoke run. Don't block for minutes. Over SSH (vast.ai), use `tmux`.
4. Use `--no-wandb --no-tb` for quick/local runs so nothing is logged to external backends unless
   the user wants tracking. Never `--save-checkpoints` unless asked (they're large).
5. **Compute is CPU-bound** (MuJoCo physics; tiny nets). On a GPU box force CPU with
   `export CUDA_VISIBLE_DEVICES=""` — the GPU doesn't help and may need special PyTorch builds.
   Speedup comes from **more cores + running seeds in parallel**, not the GPU.

## Config overlays (`--config <name>`)

Resolution: `default.yaml` ← `ppo_<agent>.yaml` ← `--config overlay` ← CLI flags.

- `cleanrl_match` — validated single-env PPO recipe (obs+reward normalization, LR anneal, Adam eps 1e-5).
- `continual_fast` — the full continual experiment, 8× async vector envs (throughput).
- `pt_fixed` — corrected PT on the validated base.

Normalization (in `cleanrl_match`/`continual_fast`/`pt_fixed`) is what makes HalfCheetah reach
positive return — a run without it will look "broken" (negative returns) but isn't.

## Recipes

```bash
cd <PARENT of src_continuous_control>   # Windows: "e:/update-single task + videos"  ·  Linux: clone dir

# Smoke test (~seconds, no backends)
python -m src_continuous_control.train --agent pt --seed 0 \
    --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb --no-tb

# Single-task validation — the honest "is PPO healthy on HalfCheetah?" test (positive return expected)
python -m src_continuous_control.train --agent vanilla --config cleanrl_match \
    --disable-task-switch --total-steps 1000000 --no-wandb --no-tb --seed 0

# Full continual run, one agent, one seed
python -m src_continuous_control.train --agent pt --config continual_fast --seed 0

# Multi-seed sweeps
bash src_continuous_control/scripts/run_all.sh                 # sequential, seeds 0-4, vanilla+pt

# Parallel sweep on a many-core box (e.g. vast.ai 24 threads): all 3 agents per seed at once
export CUDA_VISIBLE_DEVICES=""
for S in 0 1 2 3 4; do
  for AG in vanilla pt ewc; do
    python -u -m src_continuous_control.train --agent $AG --config continual_fast \
        --seed $S --no-wandb --no-tb > sweeplogs/${AG}_seed${S}.log 2>&1 &
  done
  wait     # finish this seed's 3 agents before the next; ~80 min/wave, ~7 h total for 5 seeds
done
```

## After a run

- Curves: `results/*.pkl` (`*_returns`, `*_ep_returns`, `*_eval_returns`, `*_velocities`).
- TensorBoard: `runs/`. W&B: project `pt-continuous-control` (unless `--no-wandb`).
- Watch the printed `return=` (true, un-normalized episodic return) and `critic_loss` — a healthy
  HalfCheetah climbs to hundreds/thousands positive; a diverging critic (`~1e5`) means something's wrong.
- To compare/plot, hand off to the `plot-results` skill.

## Verifying the code without full training

For algorithm-correctness checks (e.g. PT value-preservation) prefer a tiny in-process Python
script over a training run — construct the agent on CPU with a mock config and assert on shapes /
value math. Set `PYTHONPATH` to the parent dir and `PYTHONIOENCODING=utf-8` (source has `θ`/`Δ`).
The unit tests in `src_continuous_control/tests` (`pytest -q`) are the first line of defense.
