"""Agent registry."""
from .ppo_vanilla import PPOVanilla
from .ppo_pt import PPOPT
from .ppo_pt_full import PPOPTFull
from .ppo_ewc import PPOEWC

AGENTS = {
    "vanilla": PPOVanilla,
    "pt": PPOPT,
    "pt_full": PPOPTFull,
    "ewc": PPOEWC,
}

__all__ = ["PPOVanilla", "PPOPT", "PPOPTFull", "PPOEWC", "AGENTS"]
