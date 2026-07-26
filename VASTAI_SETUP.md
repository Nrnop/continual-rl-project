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
- **PIN EACH PROCESS TO ONE THREAD — this is not optional.** Torch defaults to a large intra-op
  thread pool per process, so 3 concurrent runs (3 mains × ~13 torch threads + 24 env workers ≈ 63
  threads on 24 cores) thrash and throughput **collapses to ~40 sps** — the full sweep would take
  ~21 h *per seed*. With the pinning below each run gets ~1600 sps (near solo speed even with 3
  concurrent), and the whole sweep finishes in ~3 h:
  ```bash
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  ```
  The `scripts/run_*.sh` runners set this themselves; export it manually for any ad-hoc run.

## 1. Get the code onto the box (and verify it's the FIXED version)

The code is delivered by **copying the project folder over VSCode Remote-SSH** (not `git clone`).
Open the box in VSCode Remote-SSH; the folder that contains `src_continuous_control/` is already
present. **Run everything from that parent folder** (the run-from-parent rule).

**Verify it's the fixed code before doing anything** — a copy made before the fixes is the old,
negative-return version:

- `src_continuous_control/VASTAI_SETUP.md` and `CLAUDE.md` must exist (they only exist in the fixed version).
- After installing deps (§2), `pytest` must reach **15 passed** (§3). If not, the copy is stale —
  re-copy the current local folder and start over.

```bash
cd <folder that CONTAINS src_continuous_control/>     # the parent dir — run everything from here
ls src_continuous_control/VASTAI_SETUP.md              # sanity: should exist
```

> Alternative (only if the box has the repo via git): `git pull` to fast-forward to the latest
> `main`. But the normal path here is the VSCode folder copy above.

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
cd <folder that CONTAINS src_continuous_control/>    # the parent dir
export CUDA_VISIBLE_DEVICES=""
# REQUIRED: without this, concurrent runs oversubscribe the cores and throughput
# collapses from ~1600 sps to ~40 sps (see TL;DR).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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

Result `.pkl`s land in `src_continuous_control/results/` (gitignored). Plot on the box first:

```bash
python -m src_continuous_control.plots.plot_compare --seeds 0 1 2 3 4
# figures -> src_continuous_control/plots/figures/
```

Then pull the figures to your machine. Since you're on **VSCode Remote-SSH**, the easy way is:
right-click `src_continuous_control/plots/figures/` in the VSCode Explorer → **Download**.
(Alternatives: `scp -r -P <ssh_port> root@<host>:<path>/plots/figures ./`, or `git add -f` the
figures and push if the box has git.)

## Notes

- Don't use `--save-checkpoints` (large; not needed). Keep `--no-wandb`.
- `.claude/skills/` (`/run-experiment`, `/plot-results`) travels with the repo. `graphify` is a
  user-global skill and won't be present here — not needed for runs.
