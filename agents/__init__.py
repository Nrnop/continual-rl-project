"""Agent registry.

`pt` is the full split agent — split actor AND split critic, two separate networks each. It was
registered as `pt_full` through Phase 1, alongside a legacy critic-only `pt`; Phase 2 keeps only
this one and gives it the plain name. The legacy agent is at
`archive/phase1/agents/ppo_pt_critic_only.py` and must not be imported.
"""
from .ppo_vanilla import PPOVanilla
from .ppo_pt import PPOPT
from .ppo_ewc import PPOEWC

AGENTS = {
    "vanilla": PPOVanilla,
    "pt": PPOPT,
    "ewc": PPOEWC,
}

__all__ = ["PPOVanilla", "PPOPT", "PPOEWC", "AGENTS"]
