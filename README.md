# Permanent–Transient Representations for Continual RL in Continuous Control

A continuous-control extension of the **Permanent–Transient (PT) value decomposition**
(Anand & Precup, *Prediction and Control in Continual Reinforcement Learning*, NeurIPS 2023) to a
**PPO actor-critic**, tested on MuJoCo HalfCheetah and dm_control cartpole-swingup.

Codebase for Nura Nabipour's BIHE bachelor's thesis.

---

## The question

An RL agent in a changing environment faces the **stability–plasticity dilemma**: keep old
knowledge and adapt slowly, or adapt fast and forget. The PT decomposition answers it by splitting
a learner into two components on different timescales —

- a **permanent** component, slow, holding what stays true across situations;
- a **transient** component, fast, correcting for the situation right now.

The original paper does this to a *value function* in a *value-based* agent (DQN, Q-learning). This
project asks whether the same idea works in an **actor-critic**, where the split is applied to both
the critic and the **policy** itself:

```
V(s) = V_perm(s) + V_trans(s)          μ(s) = μ_perm(s) + μ_trans(s)
```

Splitting the policy is the paper's own suggested extension (§7.1, Limitations and Future Work),
which it did not test.

## The benchmarks

The reward function is always fixed. What changes is the **physics**, so this tests adaptation to a
changing *body and world* rather than a changing goal.

**Two environments**, chosen to be as unalike as possible, because a result on one cannot tell you
whether it is a property of the method or of the environment:

- **HalfCheetah** (6 actuators, 17 observations, a running gait) — ground friction, joint damping,
  and optionally link mass and armature are rescaled.
- **cartpole-swingup** (1 actuator, 5 observations, balancing at a point) — the pole's length and
  mass are rescaled. Its reward is bounded in [0,1] over exactly 1000 steps, so the maximum return
  is **1000 by construction** and every result reads as a percentage of optimal.

**Two kinds of non-stationarity:**

- **Observable boundaries** — physics held constant within a task and changed at a known boundary,
  matching the paper's *semi-continual* setting.
- **Smooth drift** — physics rescaled every step by a sine schedule, with no boundaries, no task
  index and no reset signal. This is the setting the thesis proposal specifies.

## Agents

| agent | what it is |
|---|---|
| `vanilla` | Standard PPO. The baseline. |
| `ewc` | PPO + Elastic Weight Consolidation — a diagonal Fisher penalty anchoring important weights at each boundary. |
| `pt` | The contribution: PPO with a **split actor and split critic**, four networks on two timescales, with periodic consolidation from transient into permanent. |

10 seeds per agent. Medians and exact permutation tests throughout.

## What we measure

- **Return over time** — how each agent copes as the physics change.
- **Forward / backward transfer** — a task×task evaluation matrix. *Backward* asks whether the agent
  still handles physics it met earlier; *forward* asks how well it handles physics it has not yet
  seen.
- **The two-component ablation** — `vanilla` vs `pt` with the permanent frozen vs `pt` intact,
  isolating what the split buys and what the permanent's *learning* buys on top of it.

## Quick start

```bash
# Run from the PARENT directory, always.
cd "e:/update-single task + videos"

pip install -r src_continuous_control/requirements_continuous.txt

# A short smoke test
python -m src_continuous_control.train --agent pt --seed 0 \
    --total-steps 20000 --switch 4000 --no-wandb --no-tb

# One full run (the Phase 2a benchmark is the default; no overlay needed)
python -m src_continuous_control.train --agent pt --seed 0

# Before any long run: the pre-flight gates
python -m src_continuous_control.scripts.preflight
python -m src_continuous_control.scripts.preflight --dynamic-range   # full-length, run once

# The sweep, then the figures
python -m src_continuous_control.scripts.run_phase2_sweep --jobs 7
python -m src_continuous_control.plots.make_phase2_figures --seeds 0 1 2 3 4

# Tests
python -m pytest src_continuous_control/tests -q
```

Requires **gymnasium ≥ 0.29** and the **mujoco ≥ 3.1** pip wheel (native on Windows — no
`mujoco-py`). Training is CPU-bound; pin threads when running seeds in parallel.

## Layout

```
agents/     PPO core + vanilla / EWC / PT
models/     GaussianActor, SplitGaussianActor, VanillaCritic, SplitCritic
envs/       LipschitzDriftHalfCheetah and DriftCartpoleSwingup — the two benchmarks
utils/      seeding, rollout buffers, logging, metrics and probes
plots/      figure generation, one module per study
scripts/    preflight.py (the gates), run_phase2_sweep.py (the runner), report_*.py (the numbers)
tests/      292 tests
configs/    layered YAML: default <- per-agent <- overlay <- CLI
archive/    Phase 1 — see archive/phase1/README.md
```

- **`CLAUDE.md`** — how the repo works, the PT mechanism in detail, and the failure modes this
  project has actually hit.
- **`PHASE2_INSTRUCTIONS.md`** — the current work plan *(local working file, not tracked)*.
- **`PT_SPECIFICATION.md`** — the implementation specification for the PT agent.

## Results

Three write-ups, each self-contained and regenerable from the committed configs:

| file | what it covers | headline |
|---|---|---|
| **`HALFCHEETAH_RESULTS.md`** | HalfCheetah, observable boundaries (~157 runs) | `pt` ties vanilla at matched exploration and loses to EWC |
| **`CARTPOLE_RESULTS.md`** | cartpole-swingup, observable boundaries (95 runs) | `pt` beats vanilla and EWC; the ablation shows it is the permanent's *learning* that does it |
| **`SWITCHRATE_RESULTS.md`** | both environments, phase length varied (120 runs) | a pre-registered prediction that **failed**; both headline results replicate on independent hardware |
| **`DRIFT_RESULTS.md`** | both environments, smooth boundary-free drift (160 runs) | on cartpole with a fast drift component, `pt` beats vanilla and online EWC |

The two environments disagree, and that disagreement is the main finding: `pt` helps on cartpole and
not on HalfCheetah. The measured difference between them is how much competence carries from one
task to the next — 0.56 on cartpole against 0.23 on HalfCheetah, disjoint at p = 1e-5. Six competing
explanations were tested and eliminated; that one survives but was reached by elimination rather than
demonstrated directly. Each report states its own limitations.

## Status

**Phase 2a and 2b complete**, ~530 runs across both environments, both kinds of non-stationarity.

**Phase 1 is complete and archived.** It studied the same PT agent on a reward-switch benchmark
across ~979 runs. Its headline framing was set aside by the supervisors — it rested on a comparison
against a non-standard baseline — but a number of its measurements stand independently of that.
`archive/phase1/README.md` records precisely which, and why.

Training artifacts (`results/`, `runs/`, `*.pkl`, checkpoints) are not tracked; they are
regenerable from the configs and seeds.
