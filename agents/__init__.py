"""Agent registry."""
from .ppo_vanilla import PPOVanilla
from .ppo_pt import PPOPT
from .ppo_ewc import PPOEWC

AGENTS = {
    "vanilla": PPOVanilla,
    "pt": PPOPT,
    "ewc": PPOEWC,
}

__all__ = ["PPOVanilla", "PPOPT", "PPOEWC", "AGENTS"]
