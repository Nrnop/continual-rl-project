# Permanent–Transient Representations for Continual RL in Continuous Control

A continuous-control extension of the **Permanent–Transient (PT) value decomposition**
(Anand & Precup, *Prediction and Control in Continual Reinforcement Learning*, NeurIPS 2023) to a
**PPO actor-critic on MuJoCo HalfCheetah**.

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

## The benchmark

**HalfCheetah with non-stationary physics.** The reward function is fixed — always "run forward" —
but at each task boundary the **dynamics change**: ground friction, joint damping, and optionally
link mass and armature are rescaled. Boundaries are observable, matching the paper's *semi-continual*
setting.

This tests continual adaptation to a changing *body and world* rather than a changing goal.

## Agents

| agent | what it is |
|---|---|
| `vanilla` | Standard PPO. The baseline. |
| `ewc` | PPO + Elastic Weight Consolidation — a diagonal Fisher penalty anchoring important weights at each boundary. |
| `pt` | The contribution: PPO with a **split actor and split critic**, four networks on two timescales, with periodic consolidation from transient into permanent. |

5 seeds per agent. Medians and exact permutation tests throughout.

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
python -m src_continuous_control.train --agent pt_full --seed 0 \
    --total-steps 6000 --n-steps 1000 --switch 2000 --no-wandb --no-tb

# A full run
python -m src_continuous_control.train --agent pt_full --config drift --seed 0

# Tests
python -m pytest src_continuous_control/tests -q
```

Requires **gymnasium ≥ 0.29** and the **mujoco ≥ 3.1** pip wheel (native on Windows — no
`mujoco-py`). Training is CPU-bound; pin threads when running seeds in parallel.

## Layout

```
agents/     PPO core + vanilla / EWC / PT
models/     GaussianActor, SplitGaussianActor, VanillaCritic, SplitCritic
envs/       LipschitzDriftHalfCheetah — the non-stationary benchmark
utils/      seeding, rollout buffers, logging, metrics and probes
plots/      figure generation
tests/      76 tests
configs/    layered YAML: default <- per-agent <- overlay <- CLI
archive/    Phase 1 — see archive/phase1/README.md
```

- **`CLAUDE.md`** — how the repo works, the PT mechanism in detail, and the failure modes this
  project has actually hit.
- **`PHASE2_INSTRUCTIONS.md`** — the current work plan *(local working file, not tracked)*.
- **`PT_SPECIFICATION.md`** — the implementation specification for the PT agent.

## Status

**Phase 2, in progress.** The benchmark is being changed from a reward-sign switch to a dynamics
change, and the agent set reduced to the three above.

**Phase 1 is complete and archived.** It studied the same PT agent on a reward-switch benchmark
across ~979 runs. Its headline framing was set aside by the supervisors — it rested on a comparison
against a non-standard baseline — but a number of its measurements stand independently of that.
`archive/phase1/README.md` records precisely which, and why.

Training artifacts (`results/`, `runs/`, `*.pkl`, checkpoints) are not tracked; they are
regenerable from the configs and seeds.
