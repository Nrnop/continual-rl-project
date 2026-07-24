# Running this project on vast.ai (or any Linux GPU box)

Setup + full-sweep playbook for `src_continuous_control`. Target box we planned for:
**AMD Ryzen 9 3900X, 24 threads, 64 GB RAM, ~54 GB disk, RTX 5080** ($0.205/hr).

## TL;DR — the box's CPU is the asset, not the GPU

Training is **CPU-bound**: the wall-clock is almost entirely MuJoCo `env.step()` physics; the
policy/critic are tiny `[64,64]` nets. So:

- **Force CPU.** `export CUDA_VISIBLE_DEVICES=""`. The RTX 5080 (Blackwell, sm_120) would need a
  special PyTorch build (`cu128`); the default wheel likely lacks sm_120 kernels, and the GPU
  wouldn't speed this up anyway (transfer overhead can make it slower).
- **Speedup = parallelism.** 24 threads → run multiple seeds/agents at once. That turns the full
  5-seed sweep from ~20 h sequential into **~7 h**.

## 1. Clone (after the work is pushed)

The GitHub remote must already contain the fixed code (PT fixes, tests, configs, docs). If it only
has the old baseline commit, you'll clone the negative-return version — push first from the dev machine.

```bash
git clone https://github.com/Nrnop/continual-rl-project.git
cd continual-rl-project            # this is the PARENT of src_continuous_control/  -> run everything from here
```

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src_continuous_control/requirements_continuous.txt
```

- The `mujoco` pip wheel runs **headless for training** with no extra system packages.
- Only if you use `--render` (videos): `export MUJOCO_GL=egl` and `apt-get install -y libgl1 libosmesa6`.
- Linux uses `fork` for `AsyncVectorEnv` (faster, no Windows spawn/pickle issues).

## 3. Sanity-gate BEFORE the long sweep

```bash
export CUDA_VISIBLE_DEVICES=""
python -m pytest src_continuous_control/tests -q            # expect 15 passed
# 30-second smoke run — confirm it trains and critic_loss is small (no divergence):
python -u -m src_continuous_control.train --agent pt --config continual_fast \
    --disable-task-switch --total-steps 6000 --n-steps 256 --no-wandb --no-tb --no-eval --seed 0
```

## 4. Full experiment — 3 agents × 5 seeds (the headline result)

Run **inside tmux** so an SSH disconnect doesn't kill hours of compute. All 3 agents per seed run
in parallel (3 × 8 envs ≈ 24 threads); seeds go sequentially.

```bash
tmux new -s sweep
cd continual-rl-project
export CUDA_VISIBLE_DEVICES=""
mkdir -p sweeplogs

for S in 0 1 2 3 4; do
  echo "=== seed $S: vanilla+pt+ewc in parallel  $(date) ==="
  for AG in vanilla pt ewc; do
    python -u -m src_continuous_control.train --agent $AG --config continual_fast \
        --seed $S --no-wandb --no-tb > sweeplogs/${AG}_seed${S}.log 2>&1 &
  done
  wait     # finish this seed's 3 agents before starting the next
done
echo "SWEEP DONE $(date)"
```

- Detach with `Ctrl-b d`; reattach with `tmux attach -t sweep`.
- Config `continual_fast` = 3.07 M steps, directional switch every 614 400 steps, 8 async envs,
  normalized. It loads each agent's YAML first, so PT/EWC knobs survive.
- **Estimate:** ~80 min/wave × 5 ≈ **7 h**, ≈ **$1.45**.
- If sps looks low (24 workers + 3 mains slightly oversubscribes 24 threads), either run **2 agents
  in parallel** instead of 3, or add `--num-envs 6` to each run.
- Watch progress: `grep return= sweeplogs/pt_seed0.log | tail`. Healthy HalfCheetah climbs from
  negative to positive within each phase; `critic_loss` stays small (a `~1e5` value = something wrong).

## 5. Get results back

Result `.pkl`s land in `src_continuous_control/results/` (gitignored). Either:

```bash
# plot on the box, then copy just the figures:
python -m src_continuous_control.plots.plot_compare --seeds 0 1 2 3 4
# from your local machine:
scp -r -P <ssh_port> root@<host>:continual-rl-project/src_continuous_control/plots/figures ./
```

or `git add -f src_continuous_control/plots/figures && git commit && git push` to ship figures via git.

## Notes

- Don't use `--save-checkpoints` (large; not needed). Keep `--no-wandb`.
- `.claude/skills/` (`/run-experiment`, `/plot-results`) travels with the repo. `graphify` is a
  user-global skill and won't be present here — not needed for runs.
