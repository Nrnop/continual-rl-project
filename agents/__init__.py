"""Agent registry.

`pt` is the split agent: split actor and split critic, each made of two separate networks.
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
